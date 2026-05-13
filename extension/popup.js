const DEFAULTS = {
  cloudUrl: "http://127.0.0.1:8080",
  deviceId: "local-pc",
  token: "",
};

const $ = (id) => document.getElementById(id);

function setStatus(message) {
  $("status").textContent = message || "";
}

function normalizeUrl(url) {
  return String(url || "").trim().replace(/\/+$/, "");
}

async function loadSettings() {
  const saved = await chrome.storage.sync.get(DEFAULTS);
  $("cloud-url").value = saved.cloudUrl || DEFAULTS.cloudUrl;
  $("device-id").value = saved.deviceId || DEFAULTS.deviceId;
  $("token").value = saved.token || "";
}

async function saveSettings() {
  const data = {
    cloudUrl: normalizeUrl($("cloud-url").value) || DEFAULTS.cloudUrl,
    deviceId: $("device-id").value.trim() || DEFAULTS.deviceId,
    token: $("token").value.trim(),
  };
  await chrome.storage.sync.set(data);
  setStatus("Configuração salva.");
  return data;
}

async function settings() {
  const saved = await chrome.storage.sync.get(DEFAULTS);
  return {
    cloudUrl: normalizeUrl(saved.cloudUrl || DEFAULTS.cloudUrl),
    deviceId: saved.deviceId || DEFAULTS.deviceId,
    token: saved.token || "",
  };
}

async function cloudFetch(path, options = {}) {
  const cfg = await settings();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (cfg.token) headers.Authorization = `Bearer ${cfg.token}`;
  const response = await fetch(cfg.cloudUrl + path, {
    ...options,
    headers,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

async function sendCommand(command) {
  const cfg = await settings();
  const data = await cloudFetch("/api/commands", {
    method: "POST",
    body: JSON.stringify({
      device_id: cfg.deviceId,
      command,
    }),
  });
  setStatus(`Comando enviado: ${data.command?.command || command}`);
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadSettings();
  $("save").addEventListener("click", () => saveSettings().catch((err) => setStatus(err.message)));
  $("health").addEventListener("click", async () => {
    try {
      const data = await cloudFetch("/health");
      setStatus(data.ok ? "Cloud API online." : "Cloud API respondeu com erro.");
    } catch (err) {
      setStatus(`Erro: ${err.message}`);
    }
  });
  $("open").addEventListener("click", () => sendCommand("open_tinder").catch((err) => setStatus(`Erro: ${err.message}`)));
  $("start").addEventListener("click", () => sendCommand("start_autoswipe").catch((err) => setStatus(`Erro: ${err.message}`)));
  $("pause").addEventListener("click", () => sendCommand("pause_autoswipe").catch((err) => setStatus(`Erro: ${err.message}`)));
  $("stop").addEventListener("click", () => sendCommand("stop_autoswipe").catch((err) => setStatus(`Erro: ${err.message}`)));
});
