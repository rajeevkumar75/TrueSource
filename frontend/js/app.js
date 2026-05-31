let appStatus = null;
let imageFile = null;
let videoFile = null;

const HISTORY_KEY = "truesource_scan_history";

// ── Utilities ──
function toast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3500);    
}

function saveHistory(type, label, confidence) {
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  history.unshift({ type, label, confidence, time: new Date().toLocaleString() });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 20)));
  renderHistory();
}

function renderHistory() {
  const el = document.getElementById("history-list");
  if (!el) return;
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  if (!history.length) {
    el.innerHTML = `<p class="placeholder-text">No scans yet.</p>`;
    return;
  }
  el.innerHTML = history.slice(0, 8).map((h) => `
    <div class="history-item">
      <div style="display:flex;align-items:center;gap:10px">
        <span class="type-badge">${h.type}</span>
        <span class="verdict">${h.label}</span>
      </div>
      <span class="history-meta">${(h.confidence * 100).toFixed(0)}% · ${h.time}</span>
    </div>`).join("");
}

function confidenceRing(pct, color) {
  const r = 36;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  return `
    <div class="confidence-ring">
      <svg width="88" height="88" viewBox="0 0 88 88">
        <circle class="bg" cx="44" cy="44" r="${r}"/>
        <circle class="fill" cx="44" cy="44" r="${r}" stroke="${color}"
          stroke-dasharray="${circ}" stroke-dashoffset="${offset}"/>
      </svg>
      <div class="pct">${pct.toFixed(0)}%</div>
    </div>`;
}

function verdictClass(label) {
  const l = (label || "").toLowerCase();
  if (l.includes("fake") || l === "ai") return "fake";
  return "real";
}

function probClass(label) {
  const l = (label || "").toLowerCase();
  if (l.includes("fake")) return "fake";
  if (l === "ai") return "ai";
  if (l === "human") return "human";
  return "real";
}

function renderResult(containerId, data, extra = "") {
  const el = document.getElementById(containerId);
  const pct = data.confidence * 100;
  const vClass = verdictClass(data.predicted_label);
  const color = vClass === "fake" ? "#fb7185" : "#34d399";
  const probs = Object.entries(data.class_probabilities || {}).map(([label, value]) => {
    const p = (value * 100).toFixed(1);
    return `
      <div class="prob-row">
        <div class="prob-label"><span>${label}</span><span>${p}%</span></div>
        <div class="prob-bar"><div class="prob-fill ${probClass(label)}" style="width:${p}%"></div></div>
      </div>`;
  }).join("");

  el.innerHTML = `
    <div class="result-card">
      <div class="result-header">
        ${confidenceRing(pct, color)}
        <div>
          <div class="result-verdict ${vClass}">${data.predicted_label}</div>
          <div class="result-meta">
            Decision: <strong>${data.decision}</strong><br/>
            Model: <strong>${data.model || "—"}</strong>
            · <strong>${data.inference_ms ?? "—"}ms</strong>
            ${extra}
          </div>
        </div>
      </div>${probs}
    </div>`;
}

// ── Status ──
function renderModelGrid(status) {
  const el = document.getElementById("model-grid");
  if (!el) return;
  const models = status.models || {};
  const cards = [
    { key: "image", icon: "🖼", tab: "image", label: "Image Model", desc: "Deepfake photo detection" },
    { key: "video", icon: "🎬", tab: "video", label: "Video Model", desc: "Frame-level video analysis" },
    { key: "text", icon: "📝", tab: "text", label: "Text Model", desc: "Human vs AI classifier" },
  ];
  el.innerHTML = cards.map((c) => {
    const m = models[c.key] || {};
    return `
      <div class="model-card" data-tab="${c.tab}">
        <div class="model-card-header">
          <div class="model-card-icon ${c.key}">${c.icon}</div>
          <div class="status-dot ${m.ready ? "ready" : ""}"></div>
        </div>
        <h3>${c.label}</h3>
        <p>${c.desc}</p>
        <div class="model-type">${m.ready ? m.type + " · Ready" : "Not found"}</div>
      </div>`;
  }).join("");
  el.querySelectorAll(".model-card").forEach((card) => {
    card.addEventListener("click", () => switchTab(card.dataset.tab));
  });
}

function updateStatusUI(status) {
  const loaded = status.models_loaded || {};
  const sysEl = document.getElementById("system-info");
  if (sysEl) {
    sysEl.innerHTML = `
      <p><strong>Device:</strong> ${status.device}</p>
      <p><strong>Classes:</strong> ${status.class_names.join(", ")}</p>
      <p><strong>Image loaded:</strong> ${loaded.image ? "Yes" : "No"}</p>
      <p><strong>Text loaded:</strong> ${loaded.text ? "Yes" : "No"}</p>
      <p><strong>AI threshold:</strong> ${status.models?.text?.ai_threshold ?? "—"}</p>`;
  }

  const badge = document.getElementById("live-badge");
  const liveText = document.getElementById("live-text");
  const heroDevice = document.getElementById("hero-device");
  const heroStatus = document.getElementById("hero-status");
  const contactStatus = document.getElementById("contact-model-status");

  const allReady = status.image_model && status.text_model && loaded.image && loaded.text;
  const partial = status.image_model || status.text_model;

  if (heroDevice) heroDevice.textContent = status.device.toUpperCase();
  if (heroStatus) heroStatus.textContent = allReady ? "Online" : partial ? "Partial" : "Offline";
  if (contactStatus) contactStatus.textContent = allReady ? "All models online" : "Some models missing";

  if (badge && liveText) {
    badge.classList.toggle("offline", !partial);
    liveText.textContent = allReady ? "Models connected" : partial ? "Partial connection" : "Models missing";
  }

  if (status.models?.text?.ai_threshold) {
    const t = status.models.text.ai_threshold;
    const input = document.getElementById("text-threshold");
    const label = document.getElementById("text-thresh-val");
    if (input) input.value = t;
    if (label) label.textContent = t.toFixed(2);
  }
}

async function loadStatus() {
  try {
    try { await API.warmup(); } catch (_) {}
    appStatus = await API.status();
    renderModelGrid(appStatus);
    updateStatusUI(appStatus);
  } catch (error) {
    toast(error.message, "error");
  }
}

// ── Navigation ──
function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${tab}`));
  scrollToSection("analyze");
}

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setupNavigation() {
  document.querySelectorAll('.site-nav a, .footer-links a, a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (!href || !href.startsWith("#")) return;
      e.preventDefault();
      const id = href.slice(1);
      scrollToSection(id);
      document.getElementById("site-nav")?.classList.remove("open");
    });
  });

  document.querySelectorAll(".feature-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      switchTab(link.dataset.tab);
    });
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  document.getElementById("mobile-toggle")?.addEventListener("click", () => {
    document.getElementById("site-nav")?.classList.toggle("open");
  });

  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".site-nav .nav-link");
  window.addEventListener("scroll", () => {
    document.getElementById("site-header")?.classList.toggle("scrolled", window.scrollY > 40);
    let current = "home";
    sections.forEach((sec) => {
      if (window.scrollY >= sec.offsetTop - 120) current = sec.id;
    });
    navLinks.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === `#${current}`));
  });
}

function setupSliders() {
  [["img-threshold","img-thresh-val"],["vid-threshold","vid-thresh-val"],["text-threshold","text-thresh-val"]].forEach(([id, labelId]) => {
    const input = document.getElementById(id);
    const label = document.getElementById(labelId);
    if (!input || !label) return;
    input.addEventListener("input", () => { label.textContent = parseFloat(input.value).toFixed(2); });
  });
}

function setupFileDrop(dropId, inputId, onFile) {
  const drop = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  if (!drop || !input) return;
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("dragover");
    if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) onFile(input.files[0]); });
}

function setupImage() {
  const preview = document.getElementById("img-preview");
  const btn = document.getElementById("img-analyze");
  setupFileDrop("img-drop", "img-input", (file) => {
    imageFile = file;
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
    btn.disabled = false;
  });
  btn.addEventListener("click", async () => {
    if (!imageFile || !appStatus?.image_model) { toast(appStatus?.image_model ? "Select a file" : "Image model missing", "error"); return; }
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Analyzing…';
    const form = new FormData();
    form.append("file", imageFile);
    form.append("fake_threshold", document.getElementById("img-threshold").value);
    try {
      const data = await API.predictImage(form);
      renderResult("img-result", data);
      saveHistory("image", data.predicted_label, data.confidence);
      toast(`Image: ${data.predicted_label}`);
    } catch (error) {
      document.getElementById("img-result").innerHTML = `<div class="alert alert-error">${error.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Analyze image →"; }
  });
}

function setupVideo() {
  const preview = document.getElementById("vid-preview");
  const btn = document.getElementById("vid-analyze");
  setupFileDrop("vid-drop", "vid-input", (file) => {
    videoFile = file;
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
    btn.disabled = false;
  });
  btn.addEventListener("click", async () => {
    if (!videoFile || !appStatus?.video_model) { toast("Video model missing", "error"); return; }
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Analyzing…';
    const form = new FormData();
    form.append("file", videoFile);
    form.append("fake_threshold", document.getElementById("vid-threshold").value);
    form.append("max_frames", document.getElementById("vid-frames").value);
    form.append("frame_aggregation", document.getElementById("vid-aggregation").value);
    form.append("mean_max_weight", "0.45");
    try {
      const data = await API.predictVideo(form);
      renderResult("vid-result", data, `<br/>Frames: <strong>${data.total_frames_used}</strong>`);
      saveHistory("video", data.predicted_label, data.confidence);
      toast(`Video: ${data.predicted_label}`);
    } catch (error) {
      document.getElementById("vid-result").innerHTML = `<div class="alert alert-error">${error.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Analyze video →"; }
  });
}

function setupText() {
  document.getElementById("text-analyze").addEventListener("click", async () => {
    const text = document.getElementById("text-input").value.trim();
    const btn = document.getElementById("text-analyze");
    if (!text) { toast("Enter text first", "error"); return; }
    if (!appStatus?.text_model) { toast("Text model missing", "error"); return; }
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Analyzing…';
    try {
      const data = await API.predictText(text, document.getElementById("text-threshold").value);
      renderResult("text-result", data, `<br/>Words: <strong>${data.word_count}</strong>`);
      saveHistory("text", data.predicted_label, data.confidence);
      toast(`Text: ${data.predicted_label}`);
    } catch (error) {
      document.getElementById("text-result").innerHTML = `<div class="alert alert-error">${error.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Analyze text →"; }
  });
}

function setupContact() {
  document.getElementById("contact-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    toast("Thanks! We'll get back to you soon.");
    e.target.reset();
  });
}

(async () => {
  setupNavigation();
  setupSliders();
  setupImage();
  setupVideo();
  setupText();
  setupContact();
  renderHistory();
  await loadStatus();
})();
