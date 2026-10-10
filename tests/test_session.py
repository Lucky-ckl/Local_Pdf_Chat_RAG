"""P3.1 会话状态结构单测：创建 / 读取 / 截断 / 清理。"""

from core.session import SessionStore


def test_session_create_read_and_truncate_drops_oldest():
    store = SessionStore(max_turns=3)
    store.create("s1")
    assert store.get("s1").session_id == "s1"

    for i in range(5):
        store.append_turn("s1", f"q{i}", f"a{i}")

    turns = store.history("s1")
    assert len(turns) == 3  # 超出轮数时截断
    assert [t.question for t in turns] == ["q2", "q3", "q4"]  # 最旧轮次被丢弃
    assert turns[-1].answer == "a4"


def test_get_or_create_generates_id_and_stores_citations():
    store = SessionStore(max_turns=5)
    session = store.get_or_create(None)
    assert session.session_id
    store.append_turn(session.session_id, "q", "a", [{"index": 0, "source": "x.pdf"}])
    assert store.history(session.session_id)[0].citations == [{"index": 0, "source": "x.pdf"}]


def test_get_or_create_reuses_existing_session():
    store = SessionStore(max_turns=5)
    store.append_turn("fixed", "q1", "a1")
    again = store.get_or_create("fixed")
    assert len(again.history()) == 1
    assert again.session_id == "fixed"


def test_history_for_unknown_session_is_empty():
    store = SessionStore(max_turns=2)
    assert store.history("missing") == []
    assert store.history(None) == []


def test_clear_single_and_all():
    store = SessionStore(max_turns=2)
    store.append_turn("a", "q", "a")
    store.append_turn("b", "q", "a")
    store.clear("a")
    assert store.history("a") == []
    assert len(store.history("b")) == 1
    store.clear()
    assert store.history("b") == []
