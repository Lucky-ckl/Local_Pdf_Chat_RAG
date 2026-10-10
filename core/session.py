"""会话状态管理（WBS v1.0 P3.1）。

职责：维护 ``session_id`` 与最近 ``SESSION_MAX_TURNS``（默认 5）轮的
``(question, answer, citations)``，供多轮改写（P3.2）与上下文压缩（P3.3）使用。

设计边界（WBS v1.0 §13 禁止项）：
- 会话状态**仅存内存**，面向单用户本地运行；进程重启后会话丢失属**预期行为**。
- 不引入 Redis / 数据库 / 任何外部会话存储。
- 会话超长时按 ``SESSION_MAX_TURNS`` 截断（丢弃最旧轮次），不无限增长。
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import SESSION_MAX_TURNS


@dataclass
class Turn:
    """单轮对话记录。"""

    question: str
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)


class Session:
    """单个会话：持有有界的历史轮次。"""

    def __init__(self, session_id: str, max_turns: int):
        self.session_id = session_id
        self.max_turns = max(1, int(max_turns))
        self.turns: List[Turn] = []

    def append(self, question: str, answer: str, citations: Optional[List[Dict[str, Any]]] = None) -> None:
        """追加一轮；超过 ``max_turns`` 时丢弃最旧轮次（P3.1 截断要求）。"""
        self.turns.append(Turn(question=question, answer=answer, citations=list(citations or [])))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def history(self) -> List[Turn]:
        """返回按时间升序的历史轮次副本（最旧在前）。"""
        return list(self.turns)

    def __len__(self) -> int:
        return len(self.turns)


class SessionStore:
    """内存会话仓库（线程安全的最简实现）。"""

    def __init__(self, max_turns: Optional[int] = None):
        self.max_turns = max_turns if max_turns is not None else SESSION_MAX_TURNS
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, session_id: Optional[str] = None) -> str:
        """显式创建会话；未给定 id 时生成随机 id。"""
        sid = session_id or uuid.uuid4().hex
        with self._lock:
            self._sessions[sid] = Session(sid, self.max_turns)
        return sid

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """存在则返回，不存在则以该 id（或新 id）创建。"""
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            sid = session_id or uuid.uuid4().hex
            session = Session(sid, self.max_turns)
            self._sessions[sid] = session
            return session

    def append_turn(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> Session:
        session = self.get_or_create(session_id)
        with self._lock:
            session.append(question, answer, citations)
        return session

    def history(self, session_id: Optional[str]) -> List[Turn]:
        session = self.get(session_id) if session_id else None
        return session.history() if session else []

    def clear(self, session_id: Optional[str] = None) -> None:
        """清除指定会话；``session_id`` 为空时清空全部会话。"""
        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)


# 模块级单例：单进程本地运行使用（WBS §13：不引入 Redis / 外部存储）。
session_store = SessionStore()
