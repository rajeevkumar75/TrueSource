let appStatus = null;
let imageFile = null;
let videoFile = null;

const HISTORY_KEY = "truesource_scan_history";
const THEME_KEY = "truesource_theme";

// ── Theme ──
function initTheme() {
  const toggle = document.getElementById("theme-toggle");
  const saved = localStorage.getItem(THEME_KEY);
  const root = document.documentElement;
  
  if (saved === "light") {
    root.setAttribute("data-theme", "light");
  }
  
  toggle?.addEventListener("click", () => {
    const current = root.getAttribute("data-theme");
    const next = current === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    localStorage.setItem(THEME_KEY, next);
  });
}

// ── Utilities ──
function toast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(10px)';
    setTimeout(() => el.remove(), 300);
  }, 3500);
}

function saveHistory(type, label, confidence, fullData) {
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  history.unshift({ type, label, confidence, time: new Date().toLocaleString(), data: fullData });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 30)));
  renderHistory();
}

function renderHistory() {
  const el = document.getElementById("history-list");
  if (!el) return;
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  if (!history.length) {
    el.innerHTML = `<div class="empty-state sm"><p>No scans recorded yet.</p></div>`;
    return;
  }
  
  el.innerHTML = history.slice(0, 15).map((h, i) => {
    const lClass = (h.label || "").toLowerCase().includes("fake") ? "fake" : 
                   (h.label || "").toLowerCase() === "ai" ? "ai" : 
                   (h.label || "").toLowerCase() === "human" ? "human" : "real";
                   
    return `
    <div class="hist-item" data-idx="${i}">
      <div class="hist-left">
        <span class="hist-type ${h.type}">${h.type}</span>
        <span class="hist-label ${lClass}">${h.label}</span>
      </div>
      <div class="hist-right">
        <span class="hist-conf">${(h.confidence * 100).toFixed(1)}%</span>
        <span class="hist-time">${h.time.split(",")[0]}</span>
      </div>
    </div>`;
  }).join("");
  
  // Attach click handlers to replay history
  el.querySelectorAll(".hist-item").forEach(item => {
    item.addEventListener("click", () => {
      const h = history[item.dataset.idx];
      switchTab(h.type);
      
      let containerId;
      if (h.type === "image") containerId = "img-result";
      if (h.type === "video") containerId = "vid-result";
      if (h.type === "text") containerId = "text-result";
      
      if (h.data.results && Array.isArray(h.data.results)) {
         renderBatchResult(h.data);
      } else {
         renderResult(containerId, h.data, h.type, true);
      }
      
      document.getElementById(`${h.type}-export`)?.classList.remove("hidden");
      window.currentResult = h.data; // for export
      scrollToSection("analyze");
    });
  });
}

function probClass(label) {
  const l = (label || "").toLowerCase();
  if (l.includes("fake")) return "fake";
  if (l === "ai") return "ai";
  if (l === "human") return "human";
  return "real";
}

function exportJSON(data, type) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `truesource_${type}_report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Result Rendering ──
function confidenceRing(pct, color) {
  const r = 42;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  return `
    <div class="conf-ring">
      <svg viewBox="0 0 96 96">
        <circle class="bg" cx="48" cy="48" r="${r}"/>
        <circle class="fill" cx="48" cy="48" r="${r}" stroke="${color}"
          stroke-dasharray="${circ}" stroke-dashoffset="${offset}"/>
      </svg>
      <div class="pct">${pct.toFixed(0)}<span style="font-size:0.6em;color:var(--text-faint)">%</span></div>
    </div>`;
}

function renderResult(containerId, data, type, isHistory = false) {
  const el = document.getElementById(containerId);
  const pct = data.confidence * 100;
  
  const vClass = probClass(data.predicted_label);
  let color = "#10b981"; // emerald
  if (vClass === "fake") color = "#f43f5e"; // rose
  if (vClass === "ai") color = "#f97316"; // orange
  if (vClass === "human") color = "#0ea5e9"; // sky
  
  const probs = Object.entries(data.class_probabilities || {}).map(([label, value]) => {
    const p = (value * 100).toFixed(1);
    return `
      <div class="prob-row">
        <div class="prob-head"><span>${label}</span><span>${p}%</span></div>
        <div class="prob-track">
          <div class="prob-fill ${probClass(label)}" data-width="${p}" style="width: 0%"></div>
        </div>
      </div>`;
  }).join("");

  let extra = "";
  if (type === "video" && data.total_frames_used) extra = `<br/>Frames Analyzed: <strong>${data.total_frames_used}</strong>`;
  if (type === "text" && data.word_count) extra = `<br/>Word Count: <strong>${data.word_count}</strong>`;
  
  const historyTag = isHistory ? `<span class="hist-type" style="background:rgba(255,255,255,0.1);color:var(--text-dim);font-size:0.65rem;margin-left:12px;">Viewing History</span>` : "";

  el.innerHTML = `
    <div class="result-card">
      <div class="result-hero">
        ${confidenceRing(pct, color)}
        <div class="result-info">
          <div class="result-verdict ${vClass}">${data.predicted_label}</div>
          <div class="result-meta">
            Model: <strong>${data.model || "—"}</strong>
            · Latency: <strong>${data.inference_ms ?? "—"}ms</strong>
            ${extra}
            ${historyTag}
          </div>
        </div>
      </div>
      <div class="prob-list">
        ${probs}
      </div>
    </div>`;
    
  // Trigger bar transition
  setTimeout(() => {
    el.querySelectorAll('.prob-fill').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 50);
}

function renderBatchResult(data) {
  const el = document.getElementById("text-result");
  const paragraphs = data.results || [];
  
  const aiCount = paragraphs.filter(p => p.predicted_label === "AI").length;
  
  let html = `
    <div class="result-card" style="padding-bottom:16px;margin-bottom:20px;border-bottom:1px solid var(--border)">
      <h3 style="font-family:'Space Grotesk';margin-bottom:8px">Batch Analysis Complete</h3>
      <p style="color:var(--text-dim);font-size:0.9rem">
        Analyzed <strong>${data.total_paragraphs}</strong> paragraphs. 
        <strong>${aiCount}</strong> flagged as AI-generated.
      </p>
    </div>
  `;
  
  html += paragraphs.map((p, i) => {
    const isAi = p.predicted_label === "AI";
    const vClass = isAi ? "ai-verdict" : "human-verdict";
    const cClass = isAi ? "ai-card" : "human-card";
    
    return `
      <div class="batch-result-card ${cClass}">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
          <span style="font-size:0.75rem;color:var(--text-faint);font-family:'JetBrains Mono'">Para ${i+1}</span>
          <span class="batch-verdict ${vClass}">${p.predicted_label} (${(p.confidence*100).toFixed(1)}%)</span>
        </div>
        <div class="batch-para">${p.text}</div>
      </div>
    `;
  }).join("");
  
  el.innerHTML = html;
}

// ── Status ──
function renderModelGrid(status) {
  const el = document.getElementById("model-grid");
  if (!el) return;
  const models = status.models || {};
  const cards = [
    { key: "image", icon: "image", tab: "image", label: "Image Model", desc: "ResNet18 visual deepfake detector" },
    { key: "video", icon: "video", tab: "video", label: "Video Model", desc: "Temporal frame sequence analysis" },
    { key: "text",  icon: "text",  tab: "text",  label: "Text Model",  desc: "DistilBERT AI writing classifier" },
  ];
  
  el.innerHTML = cards.map(c => {
    const m = models[c.key] || {};
    const icSvg = c.key === "image" ? `<svg viewBox="0 0 24 24"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>`
                : c.key === "video" ? `<svg viewBox="0 0 24 24"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>`
                : `<svg viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>`;
    
    return `
      <div class="model-card" data-tab="${c.tab}">
        <div class="model-card-top">
          <div class="model-card-ic ${c.key}">${icSvg}</div>
          <div class="status-indicator ${m.ready ? "ready" : ""}">
            <div class="dot"></div>
            <span class="txt">${m.ready ? "Online" : "Offline"}</span>
          </div>
        </div>
        <h3>${c.label}</h3>
        <p>${c.desc}</p>
        <div class="model-card-arch">${m.ready ? m.type : "Model unavailable"}</div>
      </div>`;
  }).join("");
  
  el.querySelectorAll(".model-card").forEach(card => {
    card.addEventListener("click", () => switchTab(card.dataset.tab));
  });
}

function updateStatusUI(status) {
  const loaded = status.models_loaded || {};
  
  // Fill system panel
  const sysEl = document.getElementById("system-info");
  if (sysEl) {
    sysEl.innerHTML = `
      <div class="sys-row"><span class="sys-key">Compute</span><span class="sys-val">${status.device.toUpperCase()}</span></div>
      <div class="sys-row"><span class="sys-key">Img Model</span><span class="sys-val ${loaded.image?'ready':'not-ready'}">${loaded.image ? "Loaded" : "Missing"}</span></div>
      <div class="sys-row"><span class="sys-key">Txt Model</span><span class="sys-val ${loaded.text?'ready':'not-ready'}">${loaded.text ? "Loaded" : "Missing"}</span></div>
      <div class="sys-row"><span class="sys-key">Framework</span><span class="sys-val">PyTorch / Flask</span></div>
    `;
  }

  const allReady = status.image_model && status.text_model && loaded.image && loaded.text;
  const partial = status.image_model || status.text_model;

  // Hero section updates
  const heroDevice = document.getElementById("hero-device");
  if (heroDevice) heroDevice.textContent = status.device.toUpperCase();
  
  const pill = document.getElementById("live-pill");
  const pillText = document.getElementById("live-pill-text");
  if (pill && pillText) {
    pill.className = "live-pill " + (allReady ? "" : partial ? "partial" : "offline");
    pillText.textContent = allReady ? "All Models Online" : partial ? "Partial Connection" : "System Offline";
  }
}

async function loadStatus() {
  try {
    try { await API.warmup(); } catch (_) {}
    appStatus = await API.status();
    renderModelGrid(appStatus);
    updateStatusUI(appStatus);
  } catch (error) {
    toast("Failed to connect to backend", "error");
  }
}

// ── Navigation ──
function switchTab(tab) {
  document.querySelectorAll(".tab-pill").forEach(b => {
    const isAct = b.dataset.tab === tab;
    b.classList.toggle("active", isAct);
    b.setAttribute("aria-selected", isAct);
  });
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("hidden", p.id !== `tab-${tab}`));
}

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setupNavigation() {
  document.querySelectorAll('.site-nav a, .footer-nav a, .hero-ctas a, .feat-link').forEach(link => {
    link.addEventListener("click", e => {
      const href = link.getAttribute("href");
      if (!href || !href.startsWith("#")) return;
      e.preventDefault();
      
      const id = href.slice(1);
      scrollToSection(id);
      
      document.getElementById("site-nav")?.classList.remove("open");
      document.getElementById("mobile-toggle")?.classList.remove("open");
      
      if (link.classList.contains("feat-link") && link.dataset.tab) {
        switchTab(link.dataset.tab);
      }
    });
  });

  document.querySelectorAll(".tab-pill").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  const mobToggle = document.getElementById("mobile-toggle");
  mobToggle?.addEventListener("click", () => {
    mobToggle.classList.toggle("open");
    document.getElementById("site-nav")?.classList.toggle("open");
  });

  const header = document.getElementById("site-header");
  const navLinks = document.querySelectorAll(".site-nav .nav-link");
  const sections = document.querySelectorAll("section[id]");
  
  window.addEventListener("scroll", () => {
    header?.classList.toggle("scrolled", window.scrollY > 40);
    
    let current = "home";
    sections.forEach(sec => {
      if (window.scrollY >= sec.offsetTop - 120) current = sec.id;
    });
    navLinks.forEach(l => l.classList.toggle("active", l.getAttribute("href") === `#${current}`));
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
  
  // Remove existing listeners if any
  const newDrop = drop.cloneNode(true);
  drop.parentNode.replaceChild(newDrop, drop);
  const newInput = input.cloneNode(true);
  input.parentNode.replaceChild(newInput, input);
  
  newDrop.addEventListener("click", () => newInput.click());
  newDrop.addEventListener("dragover", e => { e.preventDefault(); newDrop.classList.add("dragover"); });
  newDrop.addEventListener("dragleave", () => newDrop.classList.remove("dragover"));
  newDrop.addEventListener("drop", e => {
    e.preventDefault(); newDrop.classList.remove("dragover");
    if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
  });
  
  newInput.addEventListener("change", () => {
    if (newInput.files.length) onFile(newInput.files[0]);
    newInput.value = ""; // reset
  });
}

// ── Image Analyzer ──
function setupImage() {
  const previewWrap = document.getElementById("img-preview-wrap");
  const preview = document.getElementById("img-preview");
  const btn = document.getElementById("img-analyze");
  const exportBtn = document.getElementById("img-export");
  const urlBtn = document.getElementById("img-url-btn");
  const urlInput = document.getElementById("img-url-input");
  
  let currentFile = null;
  let currentUrl = null;
  
  function resetImg() {
    currentFile = null; currentUrl = null;
    previewWrap.classList.add("hidden");
    preview.src = "";
    btn.disabled = true;
    exportBtn.classList.add("hidden");
  }

  setupFileDrop("img-drop", "img-input", file => {
    resetImg();
    currentFile = file;
    preview.src = URL.createObjectURL(file);
    previewWrap.classList.remove("hidden");
    btn.disabled = false;
    urlInput.value = "";
  });
  
  document.getElementById("img-remove")?.addEventListener("click", resetImg);
  
  urlBtn?.addEventListener("click", () => {
    const val = urlInput.value.trim();
    if (!val) return toast("Enter a valid URL", "error");
    resetImg();
    currentUrl = val;
    preview.src = val;
    previewWrap.classList.remove("hidden");
    btn.disabled = false;
  });
  
  urlInput?.addEventListener("keypress", e => {
    if (e.key === "Enter") urlBtn.click();
  });
  
  exportBtn?.addEventListener("click", () => {
    if (window.currentResult) exportJSON(window.currentResult, "image");
  });

  btn?.addEventListener("click", async () => {
    if (!currentFile && !currentUrl) return toast("Select an image first", "error");
    
    const origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Analyzing…';
    exportBtn.classList.add("hidden");
    
    try {
      const threshold = document.getElementById("img-threshold").value;
      let data;
      
      if (currentFile) {
        const form = new FormData();
        form.append("file", currentFile);
        form.append("fake_threshold", threshold);
        data = await API.predictImage(form);
      } else {
        data = await API.predictImageUrl(currentUrl, threshold);
      }
      
      renderResult("img-result", data, "image");
      saveHistory("image", data.predicted_label, data.confidence, data);
      
      window.currentResult = data;
      exportBtn.classList.remove("hidden");
      toast(`Analyzed: ${data.predicted_label}`);
    } catch (err) {
      document.getElementById("img-result").innerHTML = `<div class="alert alert-error">${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = origHtml;
    }
  });
}

// ── Video Analyzer ──
function setupVideo() {
  const previewWrap = document.getElementById("vid-preview-wrap");
  const preview = document.getElementById("vid-preview");
  const btn = document.getElementById("vid-analyze");
  const exportBtn = document.getElementById("vid-export");
  
  let currentFile = null;

  function resetVid() {
    currentFile = null;
    previewWrap.classList.add("hidden");
    preview.src = "";
    btn.disabled = true;
    exportBtn.classList.add("hidden");
  }

  setupFileDrop("vid-drop", "vid-input", file => {
    resetVid();
    currentFile = file;
    preview.src = URL.createObjectURL(file);
    previewWrap.classList.remove("hidden");
    btn.disabled = false;
  });
  
  document.getElementById("vid-remove")?.addEventListener("click", resetVid);
  
  exportBtn?.addEventListener("click", () => {
    if (window.currentResult) exportJSON(window.currentResult, "video");
  });

  btn?.addEventListener("click", async () => {
    if (!currentFile) return toast("Select a video first", "error");
    
    const origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Analyzing Frames…';
    exportBtn.classList.add("hidden");
    
    try {
      const form = new FormData();
      form.append("file", currentFile);
      form.append("fake_threshold", document.getElementById("vid-threshold").value);
      form.append("max_frames", document.getElementById("vid-frames").value);
      form.append("frame_aggregation", document.getElementById("vid-aggregation").value);
      
      const data = await API.predictVideo(form);
      renderResult("vid-result", data, "video");
      saveHistory("video", data.predicted_label, data.confidence, data);
      
      window.currentResult = data;
      exportBtn.classList.remove("hidden");
      toast(`Analyzed: ${data.predicted_label}`);
    } catch (err) {
      document.getElementById("vid-result").innerHTML = `<div class="alert alert-error">${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = origHtml;
    }
  });
}

// ── Text Analyzer ──
function setupText() {
  const input = document.getElementById("text-input");
  const btn = document.getElementById("text-analyze");
  const exportBtn = document.getElementById("text-export");
  
  const wCount = document.getElementById("word-count");
  const cCount = document.getElementById("char-count");
  
  let mode = "single"; // single | batch
  let batchParagraphs = [];
  
  // Word counter
  input?.addEventListener("input", () => {
    const val = input.value;
    cCount.textContent = `${val.length} chars`;
    const words = val.trim() ? val.trim().split(/\s+/).length : 0;
    wCount.textContent = `${words} words`;
  });
  
  // Mode switcher
  document.querySelectorAll(".mode-btn").forEach(mb => {
    mb.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
      mb.classList.add("active");
      mode = mb.dataset.mode;
      
      document.getElementById("txt-single").classList.toggle("hidden", mode !== "single");
      document.getElementById("txt-batch").classList.toggle("hidden", mode !== "batch");
    });
  });
  
  // Batch file drop
  setupFileDrop("txt-drop", "txt-input", file => {
    const reader = new FileReader();
    reader.onload = e => {
      const text = e.target.result;
      // split by double newline
      batchParagraphs = text.split(/\n\s*\n/).map(s => s.trim()).filter(s => s.length > 20);
      
      const sum = document.getElementById("batch-summary");
      sum.classList.remove("hidden");
      sum.innerHTML = `Loaded <strong>${file.name}</strong>. Found <strong>${batchParagraphs.length}</strong> valid paragraphs.`;
    };
    reader.readAsText(file);
  });
  
  exportBtn?.addEventListener("click", () => {
    if (window.currentResult) exportJSON(window.currentResult, "text");
  });

  btn?.addEventListener("click", async () => {
    let textToAnalyze = "";
    
    if (mode === "single") {
      textToAnalyze = input.value.trim();
      if (!textToAnalyze) return toast("Enter text first", "error");
    } else {
      if (!batchParagraphs.length) return toast("Upload a valid .txt file first", "error");
    }
    
    const origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Analyzing Text…';
    exportBtn.classList.add("hidden");
    
    try {
      const threshold = document.getElementById("text-threshold").value;
      
      if (mode === "single") {
        const data = await API.predictText(textToAnalyze, threshold);
        renderResult("text-result", data, "text");
        saveHistory("text", data.predicted_label, data.confidence, data);
        window.currentResult = data;
        
      } else {
        // Batch processing
        const results = [];
        for (let i = 0; i < batchParagraphs.length; i++) {
           btn.innerHTML = `<div class="spinner"></div> Para ${i+1}/${batchParagraphs.length}…`;
           try {
             const res = await API.predictText(batchParagraphs[i], threshold);
             results.push({ ...res, text: batchParagraphs[i] });
           } catch (e) {
             console.error("Batch item failed", e);
           }
        }
        
        if (!results.length) throw new Error("Batch failed");
        
        const aiCount = results.filter(r => r.predicted_label === "AI").length;
        const mainLabel = aiCount > (results.length / 2) ? "AI" : "Human";
        const avgConf = results.reduce((acc, r) => acc + r.confidence, 0) / results.length;
        
        const summaryData = {
          predicted_label: mainLabel,
          confidence: avgConf,
          total_paragraphs: results.length,
          results: results
        };
        
        renderBatchResult(summaryData);
        saveHistory("text", `Batch (${mainLabel})`, avgConf, summaryData);
        window.currentResult = summaryData;
      }
      
      exportBtn.classList.remove("hidden");
      toast("Text analysis complete");
    } catch (err) {
      document.getElementById("text-result").innerHTML = `<div class="alert alert-error">${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = origHtml;
    }
  });
}

// ── Init ──
(async () => {
  initTheme();
  setupNavigation();
  setupSliders();
  setupImage();
  setupVideo();
  setupText();
  
  document.getElementById("clear-history-btn")?.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
    toast("History cleared");
  });
  
  renderHistory();
  await loadStatus();
})();
