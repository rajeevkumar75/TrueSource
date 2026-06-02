// ── API Client ──────────────────────────────────
const API = {
  async request(path, options = {}) {
    const res = await fetch(path, {
      ...options,
      headers: {
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
      },
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  },

  status()             { return this.request("/api/status"); },
  warmup()             { return this.request("/api/models/warmup", { method: "POST" }); },
  predictImage(form)   { return this.request("/api/predict/image",  { method: "POST", body: form }); },
  predictVideo(form)   { return this.request("/api/predict/video",  { method: "POST", body: form }); },
  predictImageUrl(url, threshold) {
    return this.request("/api/predict/image-url", {
      method: "POST",
      body: JSON.stringify({ url, fake_threshold: Number(threshold) }),
    });
  },
  predictText(text, threshold) {
    const payload = { text };
    if (threshold !== undefined && threshold !== null) payload.ai_threshold = Number(threshold);
    return this.request("/api/predict/text", { method: "POST", body: JSON.stringify(payload) });
  },
};
