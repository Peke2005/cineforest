const API = (window.API_URL || "").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
const thread = $("thread");
const form = $("form");
const input = $("input");
const sendBtn = $("send");

const COLORS = { Mala: "var(--mala)", Regular: "var(--regular)", Buena: "var(--buena)", Excelente: "var(--excelente)" };
const ICON_BOT = '<svg viewBox="0 0 24 24"><path d="M12 2 5 12h4l-4 6h6v4h2v-4h6l-4-6h4z" fill="currentColor"/></svg>';

const STARTERS = [
  { ico: "🌲", t: "Predice con el bosque", s: "Predice la calidad de Inception", q: "Predice la calidad de Inception" },
  { ico: "🏆", t: "Rankings", s: "Top 10 de ciencia ficción de los 90", q: "Top 10 de ciencia ficción de los 90" },
  { ico: "💡", t: "Recomendaciones", s: "Algo parecido a Toy Story", q: "Recomiéndame algo parecido a Toy Story" },
  { ico: "🎬", t: "Directores", s: "Películas de Christopher Nolan", q: "Películas de Christopher Nolan" },
  { ico: "🎞️", t: "Ficha de película", s: "Háblame de El Padrino", q: "Háblame de El Padrino" },
  { ico: "🧪", t: "Predicción a medida", s: "Terror de 2010, 95 minutos", q: "Predice una película de terror de 2010 de 95 minutos" },
];
const QUICK = [
  "Mejores películas de terror de los 80", "Películas con Tom Hanks", "Películas más populares",
  "Recomiéndame una comedia", "Películas japonesas de animación", "Resumen del dataset",
  "¿Qué variables son más importantes?", "¿Qué es el bagging?", "Cines en California",
];

/* ---------- Utilidades ---------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString("es-ES");
const pct = (v) => `${(v * 100).toFixed(1)}%`;

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|\s)_([^_]+)_(?=\s|$|[.,])/g, "$1<em>$2</em>");
}
function markdown(text) {
  let html = "", list = null;
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of esc(text).split("\n")) {
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

function posterImg(url, alt, cls = "") {
  if (!url) return "";
  return `<img class="${cls}" src="${esc(url)}" alt="${esc(alt)}" loading="lazy" referrerpolicy="no-referrer" onerror="this.remove()">`;
}

/* ---------- Bloques visuales ---------- */
function carousel(movies) {
  return `<div class="carousel">${movies.map((m, i) => `
    <button class="card" type="button" data-q="Háblame de &quot;${esc(m.title)}&quot;" style="animation-delay:${i * 45}ms">
      <div class="card__poster">
        <div class="ph">${esc(m.title)}</div>
        ${posterImg(m.poster, m.title)}
        <span class="card__rank">${i + 1}</span>
        ${m.rating != null ? `<span class="card__rating">★ ${m.rating.toFixed(1)}</span>` : ""}
      </div>
      <div class="card__title">${esc(m.title)}</div>
      <div class="card__sub">${[m.year, ...(m.genres || [])].filter(Boolean).map(esc).join(" · ")}</div>
      ${m.extra ? `<span class="card__extra">${esc(m.extra)}</span>` : ""}
    </button>`).join("")}</div>`;
}

function filmCard(d, poster) {
  const line = [d.year, d.runtime && `${d.runtime} min`, d.rated, (d.countries || []).slice(0, 2).join(", ")].filter(Boolean);
  const scores = [
    d.rating != null && `<div class="score score--imdb"><b>★ ${d.rating.toFixed(1)}</b><small>IMDb · ${fmt(d.votes || 0)} votos</small></div>`,
    d.metacritic != null && `<div class="score"><b>${d.metacritic}</b><small>Metacritic</small></div>`,
    `<div class="score"><b>${d.wins}</b><small>Premios · ${d.nominations} nominac.</small></div>`,
    d.comments ? `<div class="score"><b>${d.comments}</b><small>Comentarios mflix</small></div>` : "",
  ].filter(Boolean).join("");
  return `
    <article class="film" style="--poster:url('${esc(poster || "")}')">
      <div>${poster ? posterImg(poster, d.title, "film__poster") : ""}</div>
      <div>
        <h3>${esc(d.title)}</h3>
        <div class="film__line">${line.map((x) => `<span>${esc(x)}</span>`).join("")}</div>
        <div class="chips">${(d.genres || []).map((g) => `<span class="chip">${esc(g)}</span>`).join("")}</div>
        <div class="scores">${scores}</div>
        ${d.plot ? `<p class="film__plot">${esc(d.plot)}</p>` : ""}
        <div class="film__people">
          ${d.directors?.length ? `<div><b>Dirección:</b> ${d.directors.map(esc).join(", ")}</div>` : ""}
          ${d.cast?.length ? `<div><b>Reparto:</b> ${d.cast.map(esc).join(", ")}</div>` : ""}
        </div>
      </div>
    </article>`;
}

function forestVote(p) {
  const order = ["Mala", "Regular", "Buena", "Excelente"];
  const trees = order.flatMap((c) => Array(p.votes[c] || 0).fill(c));
  return `
    <div class="forest">
      <div class="forest__head">
        <span class="forest__title">Voto de los ${p.n_trees} árboles</span>
        <span class="verdict" style="background:${COLORS[p.category]}">${esc(p.category)} · ${Math.round(p.confidence * 100)}%</span>
      </div>
      <div class="trees">${trees.map((c, i) => `<i style="background:${COLORS[c]};animation-delay:${i * 6}ms"></i>`).join("")}</div>
      ${order.map((c) => `
        <div class="vrow">
          <span>${c}</span>
          <div class="track"><div class="fill" data-w="${((p.votes[c] || 0) / p.n_trees) * 100}" style="background:${COLORS[c]}"></div></div>
          <span class="n">${p.votes[c] || 0}</span>
        </div>`).join("")}
      <div class="forest__foot">🌳 Un solo árbol de decisión diría: <b>${esc(p.tree_category)}</b></div>
    </div>`;
}

function renderBot(data) {
  let html = "";
  if (data.detail) {
    html += filmCard(data.detail, data.poster);
  } else {
    html += `<div class="text">${markdown(data.answer || "")}</div>`;
    if (data.poster && !data.movies) {
      html = `<div class="film" style="--poster:url('${esc(data.poster)}');grid-template-columns:120px 1fr">${posterImg(data.poster, "", "film__poster")}<div>${html}</div></div>`;
    }
  }
  if (data.movies?.length) html += carousel(data.movies);
  if (data.prediction) html += forestVote(data.prediction);
  if (data.intent) {
    html += `<div class="meta"><span class="tag">Intención: <b>${esc(data.intent.replace("_", " "))}</b></span>
      <span class="tag">Confianza: <b>${Math.round(data.confidence * 100)}%</b></span></div>`;
  }
  return html;
}

/* ---------- Conversación ---------- */
function addMsg(role, html) {
  const el = document.createElement("div");
  el.className = `msg msg--${role}`;
  el.innerHTML = role === "user"
    ? `<div class="bubble">${html}</div>`
    : `<div class="avatar">${ICON_BOT}</div><div class="body">${html}</div>`;
  thread.appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}

let busy = false;
async function ask(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  document.body.classList.add("chatting");
  addMsg("user", esc(text));
  const bot = addMsg("bot", `<div class="typing"><span></span><span></span><span></span></div>`);
  const body = bot.querySelector(".body");
  const slow = setTimeout(() => {
    body.insertAdjacentHTML("beforeend", `<div class="wake">Despertando el servidor de Render (el plan gratuito duerme tras un rato sin uso)…</div>`);
  }, 4000);
  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    body.innerHTML = renderBot(data);
    requestAnimationFrame(() => body.querySelectorAll(".fill").forEach((f) => (f.style.width = `${f.dataset.w}%`)));
    setStatus("ok");
  } catch (err) {
    body.innerHTML = `<div class="text"><p>⚠️ No he podido conectar con la IA en Render (${esc(err.message)}). Vuelve a intentarlo en unos segundos.</p></div>`;
  } finally {
    clearTimeout(slow);
    busy = false;
    sendBtn.disabled = false;
    bot.scrollIntoView({ behavior: "smooth", block: "start" });
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  input.value = "";
  ask(text);
});
document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-q]");
  if (el) ask(el.dataset.q.replace(/&quot;/g, '"'));
});

$("starters").innerHTML = STARTERS.map((s, i) => `
  <button class="starter" type="button" data-q="${esc(s.q)}" style="animation-delay:${120 + i * 60}ms">
    <span class="ico">${s.ico}</span><b>${esc(s.t)}</b><span>${esc(s.s)}</span>
  </button>`).join("");
$("quick").innerHTML = QUICK.map((q) => `<button type="button" data-q="${esc(q)}">${esc(q)}</button>`).join("");

/* ---------- Panel del modelo ---------- */
const panel = $("panel"), scrim = $("scrim");
function togglePanel(open) {
  panel.classList.toggle("open", open);
  panel.setAttribute("aria-hidden", String(!open));
  scrim.hidden = !open;
}
$("open-panel").addEventListener("click", () => togglePanel(true));
$("close-panel").addEventListener("click", () => togglePanel(false));
scrim.addEventListener("click", () => togglePanel(false));
document.addEventListener("keydown", (e) => e.key === "Escape" && togglePanel(false));

function setStatus(state, text) {
  const pill = $("status");
  pill.className = `pill ${state}`;
  pill.querySelector("span").textContent = text || (state === "ok" ? "IA en línea" : state === "err" ? "Sin conexión" : "Despertando servidor…");
}

function buildWall(posters) {
  const wall = $("wall");
  const cols = 9, per = 6;
  const list = posters.filter((p) => p.poster);
  let html = "";
  for (let c = 0; c < cols; c++) {
    html += `<div class="col">`;
    for (let r = 0; r < per; r++) {
      const p = list[(c * per + r * 5) % list.length];
      html += `<img src="${esc(p.poster)}" alt="" loading="lazy" referrerpolicy="no-referrer" onerror="this.style.visibility='hidden'">`;
    }
    html += `</div>`;
  }
  wall.innerHTML = html;
  requestAnimationFrame(() => wall.classList.add("ready"));
}

async function loadInfo(attempt = 1) {
  try {
    const res = await fetch(`${API}/info`);
    if (!res.ok) throw new Error(res.status);
    const info = await res.json();
    setStatus("ok");
    const m = info.metrics;
    $("k-rf").textContent = pct(m.random_forest_accuracy);
    $("k-oob").textContent = pct(m.random_forest_oob_score);
    $("k-dt").textContent = pct(m.decision_tree_accuracy);
    $("k-trees").textContent = info.n_trees;
    $("k-intent").textContent = "TF-IDF + RF";
    const max = info.feature_importances[0][1];
    $("imp").innerHTML = info.feature_importances.slice(0, 8).map(([n, v]) => `
      <li><div class="row"><code>${esc(n)}</code><span>${pct(v)}</span></div>
      <div class="bar"><i style="width:${(v / max) * 100}%"></i></div></li>`).join("");
    const dist = info.class_distribution || {};
    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    $("dist").innerHTML = ["Mala", "Regular", "Buena", "Excelente"].map((c) =>
      `<div style="flex:${dist[c] || 0};background:${COLORS[c]}" title="${c}: ${fmt(dist[c] || 0)}">${(dist[c] / total) > .14 ? c : ""}</div>`).join("");
    const s = info.stats;
    $("facts").innerHTML = [
      [s.n_movies, "películas"], [s.n_series, "series"], [s.n_comments, "comentarios"],
      [s.n_users, "usuarios"], [s.n_theaters, "cines"], [m.train_size, "para entrenar"],
    ].map(([n, l]) => `<li><b>${fmt(n)}</b>${l}</li>`).join("");
    if (info.posters?.length) buildWall(info.posters);
  } catch {
    if (attempt < 8) {
      setStatus("", "Despertando servidor…");
      setTimeout(() => loadInfo(attempt + 1), 8000);
    } else {
      setStatus("err");
    }
  }
}

loadInfo();
input.focus();
