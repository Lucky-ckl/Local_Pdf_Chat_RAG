/* 知识库问答前端逻辑：原生手写，无框架、无构建链。
   只做用户可见的事：提问、流式回答、引用来源、同步知识库目录。 */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX_TURNS = 6;
  const THEME_KEY = "rag_theme";
  const state = { sessionId: null, busy: false, turns: [] };

  // ── 配色 ────────────────────────────────────────────────
  function applyTheme(theme) {
    document.body.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
    $("theme-toggle").textContent = theme === "dark" ? "☀️" : "🌙";
  }
  applyTheme(localStorage.getItem(THEME_KEY) || "light");
  $("theme-toggle").addEventListener("click", () =>
    applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark"));

  // ── 工具 ────────────────────────────────────────────────
  const escapeHtml = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function linkify(text) {
    return escapeHtml(text).replace(/\[(\d+)\]/g, (_m, n) => `<span class="cite">[${n}]</span>`);
  }

  function sessionId() {
    if (!state.sessionId) {
      state.sessionId = (crypto.randomUUID && crypto.randomUUID()) ||
        `sess-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
    return state.sessionId;
  }

  function scrollDown() {
    const log = $("chat-log");
    log.scrollTop = log.scrollHeight;
  }

  // ── 欢迎页 ──────────────────────────────────────────────
  function showWelcome() {
    $("chat-log").innerHTML = `
      <div class="welcome">
        <span class="big">🐾</span>
        <h2>你好呀，我是你的知识库小助手～</h2>
        <p>资料已经放在本地，直接问我问题就可以啦。</p>
        <p>如果有新的 PDF，把它放进 <code>data/documents</code>，再点一下「同步资料」就行。</p>
      </div>`;
  }

  // ── 资料同步（扫描固定目录，只新增、不清空） ─────────────
  function setHint(text) { $("kb-hint").textContent = text; }

  async function syncKnowledgeBase(showResult) {
    const btn = $("sync-btn");
    if (showResult) {
      setHint("正在同步资料…");
      btn.disabled = true;
    }
    try {
      const res = await fetch("/api/kb/refresh", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      const chunks = data.total_chunks || 0;
      if (chunks === 0) {
        setHint("还没有资料：把 PDF 放进 data/documents，再点「同步资料」");
      } else {
        setHint(`资料已就绪（${chunks} 个片段），随时可以提问～`);
      }
      if (showResult) {
        const added = (data.added || []).length;
        const failed = (data.files || []).filter((f) => f.status !== "success");
        if (added) setHint(`新加入 ${added} 个文档，资料已就绪，随时可以提问～`);
        else if (failed.length) setHint(`有 ${failed.length} 个文档没能处理，请确认文件格式是否受支持`);
      }
    } catch (e) {
      setHint("同步失败：" + e.message);
    } finally {
      btn.disabled = false;
    }
  }

  // ── 对话 ────────────────────────────────────────────────
  function trimTurns() {
    while (state.turns.length > MAX_TURNS) {
      const old = state.turns.shift();
      for (const node of [old.userEl, old.el]) {
        if (node && node.parentNode) node.parentNode.removeChild(node);
      }
    }
  }

  function newTurn(question) {
    const userEl = document.createElement("div");
    userEl.className = "msg msg-user";
    userEl.innerHTML = `<div class="pip">🙂</div><div class="bubble">${escapeHtml(question)}</div>`;

    const el = document.createElement("div");
    el.className = "msg msg-assistant";
    el.innerHTML = `<div class="pip">🐾</div>
      <div class="bubble">
        <div class="body"><span class="dots"><i></i><i></i><i></i></span></div>
        <div class="cites"></div>
        <div class="note"></div>
      </div>`;

    const log = $("chat-log");
    log.appendChild(userEl);
    log.appendChild(el);
    const turn = {
      el, userEl, question, answer: "", citations: [],
      bodyEl: el.querySelector(".body"),
      citesEl: el.querySelector(".cites"),
      noteEl: el.querySelector(".note"),
    };
    state.turns.push(turn);
    trimTurns();
    scrollDown();
    return turn;
  }

  function renderAnswer(turn, streaming) {
    turn.bodyEl.innerHTML = linkify(turn.answer) + (streaming ? '<span class="cursor"></span>' : "");
    scrollDown();
  }

  function renderCitations(turn) {
    turn.citesEl.innerHTML = turn.citations.map((c) => {
      const page = c.page > 0 ? ` · 第 ${c.page} 页` : "";
      const title = `来源：${c.source || ""}`;
      return `<span class="cite-chip" title="${escapeHtml(title)}">`
        + `[${c.index}] ${escapeHtml(c.source || "文档")}${page}</span>`;
    }).join("");
  }

  function handleEvent(turn, name, data) {
    if (name === "delta") {
      turn.answer += (data && data.text) || "";
      renderAnswer(turn, true);
    } else if (name === "citation") {
      turn.citations = (data && data.citations) || [];
      renderCitations(turn);
    } else if (name === "error") {
      turn.noteEl.className = "note err";
      turn.noteEl.textContent = "抱歉，这次没能回答成功，请稍后再试～";
    }
  }

  async function readStream(turn, response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        let name = "";
        let payload = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) name = line.slice(6).trim();
          else if (line.startsWith("data:")) payload += line.slice(5).trim();
        }
        if (!name || !payload) continue;
        let data = {};
        try { data = JSON.parse(payload); } catch (_e) { data = {}; }
        handleEvent(turn, name, data);
      }
    }
  }

  async function ask(event) {
    if (event && event.preventDefault) event.preventDefault();
    if (state.busy) return;
    const box = $("question");
    const question = box.value.trim();
    if (!question) return;

    state.busy = true;
    $("ask-btn").disabled = true;
    box.value = "";
    box.style.height = "auto";

    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();
    const turn = newTurn(question);

    try {
      const res = await fetch("/api/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          session_id: sessionId(),
          enable_web_search: $("web-search").checked,
        }),
      });
      if (!res.ok) throw new Error(res.statusText);
      await readStream(turn, res);
      renderAnswer(turn, false);
      if (!turn.answer) {
        turn.noteEl.className = "note err";
        turn.noteEl.textContent = "这次没有拿到回答，请再试一次～";
      }
    } catch (e) {
      renderAnswer(turn, false);
      turn.noteEl.className = "note err";
      turn.noteEl.textContent = "网络或服务出了点小状况：" + e.message;
    } finally {
      state.busy = false;
      $("ask-btn").disabled = false;
      box.focus();
    }
  }

  // ── 绑定 ────────────────────────────────────────────────
  $("ask-form").addEventListener("submit", ask);
  $("clear-btn").addEventListener("click", () => {
    state.sessionId = null;
    state.turns = [];
    showWelcome();
  });
  $("sync-btn").addEventListener("click", () => syncKnowledgeBase(true));

  const box = $("question");
  box.addEventListener("input", () => {
    box.style.height = "auto";
    box.style.height = Math.min(box.scrollHeight, 160) + "px";
  });
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });

  showWelcome();
  syncKnowledgeBase(false);
})();
