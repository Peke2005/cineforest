const API = (window.API_URL || "").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
const messages = $("messages");
const form = $("form");
const input = $("input");
const sendBtn = $("send");

const COLORS = { Mala: "var(--mala)", Regular: "var(--regular)", Buena: "var(--buena)", Excelente: "var(--excelente)" };
const SUGGESTIONS = [
  "¿Qué puedes hacer?",
  "Predice la calidad de Inception",
  "Predice una película de terror de 2010 de 95 minutos",
  "Top 10 de ciencia ficción de los 90",
  "Películas de Christopher Nolan",
  "Recomiéndame algo parecido a Toy Story",
  "Resumen del dataset",
  "¿Qué variables son más importantes?",
  "¿Qué es el bagging?",
];

/* ---------- Markdown mínimo (seguro: se escapa antes de formatear) ---------- */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|\s)_([^_]+)_(?=\s|$|[.,])/g, "$1<em>$2</em>");
}
function markdown(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "", list = null;
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^\s*[-•]\s+(.*)/))) {
      if (list !== "ul") { close(); html += "<ul>"; list = "ul"; }
      html += `<li>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^\s*\d+\.\s+(.*)/))) {
      if (list !== "ol") { close(); html += "<ol>"; list = "ol"; }
      html += `<li>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^&gt;\s?(.*)/))) {
      close(); html += `<blockquote>${inline(m[1])}</blockquote>`;
    } else if (line.trim() === "") {
      close();
    } else {
      close(); html += `<p>${inline(line)}</p>`;
    }
  }
  close();
  return html;
}

/* ---------- Mensajes ---------- */
function addMessage(role, html) {
  const el = document.createElement("div");
  el.className = `msg msg--${role}`;
  el.innerHTML = `<div class="avatar">${role === "user" ? "🧑" : "🌲"}</div><div class="bubble">${html}</div>`;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function votesChart(pred) {
  const total = pred.n_trees;
  const rows = Object.entries(pred.votes).map(([cat, n]) => `
    <div class="vrow">
      <span>${cat}</span>
      <div class="track"><div class="fill" style="width:${(n / total) * 100}%;background:${COLORS[cat]}"></div></div>
      <span class="num">${n}</span>
    </div>`).join("");
  return `<div class="votes"><div class="title">🗳️ Voto mayoritario de los ${total} árboles</div>${rows}</div>`;
}

function renderBot(data) {
  let body = markdown(data.answer || "");
  if (data.prediction) body += votesChart(data.prediction);
  if (data.poster) {
    body = `<div class="movie"><img src="${escapeHtml(data.poster)}" alt="Póster" loading="lazy" onerror="this.remove()"><div>${body}</div></div>`;
  }
  if (data.intent) {
    body += `<div class="meta">Intención detectada: <code>${escapeHtml(data.intent)}</code> · confianza ${(data.confidence * 100).toFixed(0)}%</div>`;
  }
  return body;
}

async function ask(text) {
  addMessage("user", `<p>${escapeHtml(text)}</p>`);
  const typing = addMessage("bot", `<div class="typing"><span></span><span></span><span></span></div>`);
  sendBtn.disabled = true;
  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    typing.querySelector(".bubble").innerHTML = renderBot(data);
  } catch (err) {
    typing.querySelector(".bubble").innerHTML =
      `<p>⚠️ No he podido conectar con la API (${escapeHtml(err.message)}). Si el servidor de Render estaba dormido tarda ~50 s en despertar: vuelve a intentarlo.</p>`;
  } finally {
    sendBtn.disabled = false;
    messages.scrollTop = messages.scrollHeight;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  ask(text);
});

const chips = $("chips");
SUGGESTIONS.forEach((s) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = s;
  b.addEventListener("click", () => ask(s));
  chips.appendChild(b);
});

/* ---------- Panel lateral con métricas (/info) ---------- */
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const num = (v) => v.toLocaleString("es-ES");

async function loadInfo(attempt = 1) {
  const status = $("status");
  try {
    const res = await fetch(`${API}/info`);
    if (!res.ok) throw new Error(res.status);
    const info = await res.json();
    status.textContent = "API en línea (Render)";
    status.className = "status status--ok";
    $("m-rf").textContent = pct(info.metrics.random_forest_accuracy);
    $("m-oob").textContent = pct(info.metrics.random_forest_oob_score);
    $("m-dt").textContent = pct(info.metrics.decision_tree_accuracy);
    $("m-trees").textContent = info.n_trees;
    const max = info.feature_importances[0][1];
    $("importances").innerHTML = info.feature_importances.slice(0, 6).map(([n, v]) =>
      `<li><div class="row"><code>${escapeHtml(n)}</code><span>${pct(v)}</span></div><div class="bar"><i style="width:${(v / max) * 100}%"></i></div></li>`).join("");
    const s = info.stats;
    $("stats").innerHTML = `
      <li>🎬 <b>${num(s.n_movies)}</b> películas</li>
      <li>📺 <b>${num(s.n_series)}</b> series</li>
      <li>💬 <b>${num(s.n_comments)}</b> comentarios</li>
      <li>👥 <b>${num(s.n_users)}</b> usuarios</li>
      <li>🎦 <b>${num(s.n_theaters)}</b> cines</li>`;
  } catch {
    if (attempt < 6) {
      status.textContent = `Despertando el servidor de Render… (intento ${attempt})`;
      setTimeout(() => loadInfo(attempt + 1), 10000);
    } else {
      status.textContent = "No se puede conectar con la API";
      status.className = "status status--error";
    }
  }
}

addMessage("bot", markdown(
  "¡Hola! 👋 Soy **CineForest**, una IA local entrenada con **Random Forest** sobre el dataset de películas *mflix*.\n" +
  "Pregúntame por películas, directores, actores o rankings, o pídeme que **prediga la calidad** de una película. Pulsa una sugerencia para empezar."
));
loadInfo();
