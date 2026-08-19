"use strict";
// Minimal vanilla-JS client for the Stateful AI Agent demo UI. No build step.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const sid = () => ($("sid").value.trim() || "demo");

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

// --- tabs --------------------------------------------------------------------
document.querySelectorAll(".tabs button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    b.classList.add("active");
    $(b.dataset.tab).classList.add("active");
  });
});

// --- chat --------------------------------------------------------------------
const scrollLog = () => { $("log").scrollTop = $("log").scrollHeight; };

function bubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text || "";
  $("log").appendChild(div);
  scrollLog();
  return div;
}

// Expandable "sources" panel: retrieved memory (context) + any tool calls.
function sourcesPanel(bubbleEl, context, tools) {
  const nCtx = (context || []).length, nTool = (tools || []).length;
  if (!nCtx && !nTool) return;
  const det = document.createElement("details");
  det.className = "sources";
  const sum = document.createElement("summary");
  sum.textContent = `${nCtx} memory source(s)` + (nTool ? ` · ${nTool} tool call(s)` : "");
  det.appendChild(sum);
  (tools || []).forEach((t) => {
    const d = document.createElement("div");
    d.className = "item";
    d.innerHTML = `<div class="meta"><span class="pill">tool</span><span>${esc(t.name)}</span></div>` +
      `<div><code>${esc(JSON.stringify(t.args))}</code> → ${esc(t.result)}</div>`;
    det.appendChild(d);
  });
  (context || []).forEach((c) => {
    const s = c.rerank_score ?? c.score;
    const sc = typeof s === "number" ? ` · score ${s.toFixed(4)}` : "";
    const d = document.createElement("div");
    d.className = "item";
    d.innerHTML = `<div class="meta"><span class="pill">${esc(c.role || "?")}</span>` +
      `<span>turn ${esc(c.turn)}${sc}</span></div><div>${esc(c.content)}</div>`;
    det.appendChild(d);
  });
  bubbleEl.appendChild(det);
  scrollLog();
}

// Parse an SSE byte stream, invoking onEvent(type, data) per event.
async function readSSE(resp, onEvent) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let type = "message", data = "";
      raw.split("\n").forEach((line) => {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      });
      let parsed; try { parsed = JSON.parse(data); } catch (_) { parsed = data; }
      onEvent(type, parsed);
    }
  }
}

async function sendChat() {
  const msg = $("chatInput").value.trim();
  if (!msg) return;
  $("chatInput").value = "";
  $("chatSend").disabled = true;
  bubble("user", msg);
  const reply = bubble("assistant", "");
  let text = "", ctx = [], tools = [];
  try {
    const resp = await fetch("/chat/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: sid(), message: msg }),
    });
    if (!resp.ok || !resp.body) throw new Error(`/chat/stream → ${resp.status}`);
    await readSSE(resp, (type, data) => {
      if (type === "context") ctx = data || [];
      else if (type === "tool") tools.push(data);
      else if (type === "token") { text += data; reply.textContent = text; scrollLog(); }
      else if (type === "done") { if (data.reply) reply.textContent = text = data.reply; }
      else if (type === "error") throw new Error(String(data));
    });
    if (!text) reply.textContent = "(no response)";
    sourcesPanel(reply, ctx, tools);
  } catch (e) {
    reply.innerHTML = `<span style="color:#B00020">${esc(e.message)}</span>`;
  } finally {
    $("chatSend").disabled = false;
    $("chatInput").focus();
  }
}
$("chatSend").addEventListener("click", sendChat);
$("chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

// --- result rendering --------------------------------------------------------
function itemHTML(it, scoreKey) {
  const score = it[scoreKey];
  const scoreStr = typeof score === "number" ? score.toFixed(4) : "";
  return `<div class="item">
    <div class="meta">
      <span class="pill">${esc(it.role || "?")}</span>
      <span>turn ${esc(it.turn)}</span>
      ${scoreStr ? `<span>score ${scoreStr}</span>` : ""}
    </div>
    <div>${esc(it.content)}</div>
  </div>`;
}
function renderList(el, items, scoreKey) {
  if (!items || !items.length) { el.innerHTML = `<div class="empty">No results.</div>`; return; }
  el.innerHTML = items.map((it) => itemHTML(it, scoreKey)).join("");
}

// --- search ------------------------------------------------------------------
async function runSearch() {
  const query = $("searchInput").value.trim();
  if (!query) return;
  $("searchSend").disabled = true;
  $("searchOut").innerHTML = `<div class="empty">Searching…</div>`;
  try {
    const d = await post("/search", { session_id: sid(), query, mode: $("searchMode").value });
    renderList($("searchOut"), d.results, "score");
  } catch (e) {
    $("searchOut").innerHTML = `<div class="empty" style="color:#B00020">${esc(e.message)}</div>`;
  } finally { $("searchSend").disabled = false; }
}
$("searchSend").addEventListener("click", runSearch);
$("searchInput").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });

// --- rerank ------------------------------------------------------------------
async function runRerank() {
  const query = $("rerankInput").value.trim();
  if (!query) return;
  $("rerankSend").disabled = true;
  $("rerankBefore").innerHTML = $("rerankAfter").innerHTML = `<div class="empty">Reranking…</div>`;
  try {
    const d = await post("/rerank", { session_id: sid(), query, top_k: 10 });
    renderList($("rerankBefore"), d.before, "score");
    renderList($("rerankAfter"), d.after, "rerank_score");
  } catch (e) {
    const err = `<div class="empty" style="color:#B00020">${esc(e.message)}</div>`;
    $("rerankBefore").innerHTML = $("rerankAfter").innerHTML = err;
  } finally { $("rerankSend").disabled = false; }
}
$("rerankSend").addEventListener("click", runRerank);
$("rerankInput").addEventListener("keydown", (e) => { if (e.key === "Enter") runRerank(); });

// --- load recent history for the current session on start / session change ---
async function loadHistory() {
  $("log").innerHTML = "";
  try {
    const r = await fetch(`/memory/${encodeURIComponent(sid())}?n=20`);
    if (!r.ok) return;
    const d = await r.json();
    (d.recent || []).forEach((m) => bubble(m.role === "user" ? "user" : "assistant", m.content));
  } catch (_) { /* best-effort */ }
}
$("sid").addEventListener("change", loadHistory);
loadHistory();
