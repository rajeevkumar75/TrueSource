const API = {
  async request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
      },
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || `Request failed (${response.status})`);
    }
    return data;
  },

  status() {
    return this.request("/api/status");
  },

  warmup() {
    return this.request("/api/models/warmup", { method: "POST" });
  },

  predictImage(formData) {
    return this.request("/api/predict/image", { method: "POST", body: formData });
  },

  predictVideo(formData) {
    return this.request("/api/predict/video", { method: "POST", body: formData });
  },

  predictText(text, aiThreshold) {
    const payload = { text };
    if (aiThreshold !== undefined && aiThreshold !== null && aiThreshold !== "") {
      payload.ai_threshold = Number(aiThreshold);
    }
    return this.request("/api/predict/text", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
