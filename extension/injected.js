(function () {
  const originalFetch = window.fetch.bind(window);
  const NETWORK_CAPTURE_ENABLED = true;
  const NETWORK_CAPTURE_MAX_BODY_CHARS = 250000;
  const BRIDGE_SOURCE = (() => {
    try {
      const src = document.currentScript?.src || "";
      return new URL(src).searchParams.get("bridge") || "__bridge_default";
    } catch (_) {
      return "__bridge_default";
    }
  })();
  const PAGE_MSG_RECS = "r";
  const PAGE_MSG_PROFILE_MAP = "m";
  const PAGE_MSG_NETWORK_CAPTURE = "n";

  function captureNowIso() {
    try {
      return new Date().toISOString();
    } catch (_) {
      return "";
    }
  }

  function captureHeadersToObject(headers) {
    const out = {};
    try {
      if (!headers) return out;
      if (headers instanceof Headers) {
        headers.forEach((value, key) => {
          out[key] = value;
        });
        return out;
      }
      if (Array.isArray(headers)) {
        for (const item of headers) {
          if (Array.isArray(item) && item.length >= 2) out[String(item[0]).toLowerCase()] = String(item[1]);
        }
        return out;
      }
      if (typeof headers === "object") {
        for (const [key, value] of Object.entries(headers)) {
          out[String(key).toLowerCase()] = Array.isArray(value) ? value.join(", ") : String(value);
        }
      }
    } catch (_) {}
    return out;
  }

  function captureTruncateText(text) {
    const value = String(text ?? "");
    if (value.length <= NETWORK_CAPTURE_MAX_BODY_CHARS) {
      return { body: value, body_truncated: false, body_length: value.length };
    }
    return {
      body: value.slice(0, NETWORK_CAPTURE_MAX_BODY_CHARS),
      body_truncated: true,
      body_length: value.length,
    };
  }

  function captureIsLocalServer(url) {
    return /^https?:\/\/(localhost|127\.0\.0\.1):5043\//i.test(String(url || ""));
  }

  function captureIsRecsUrl(url) {
    try {
      const parsed = new URL(String(url || ""), location.href);
      const host = parsed.hostname.toLowerCase();
      const path = parsed.pathname || "";
      if (host !== "api.gotinder.com") return false;
      if (path.includes("/recs/") || path.includes("/v2/recs")) return true;
    } catch (_) {}
    return false;
  }

  function captureIsUsefulTinderUrl(url) {
    try {
      const parsed = new URL(String(url || ""), location.href);
      const host = parsed.hostname.toLowerCase();
      const path = parsed.pathname || "";
      if (host !== "api.gotinder.com") return false;
      if (path.includes("/recs/") || path.includes("/v2/recs")) return true;
      if (/^\/(?:like|pass)\//.test(path)) return true;
      if (path === "/updates" || path.startsWith("/updates/")) return true;
      if (path === "/v2/profile" || path.startsWith("/v2/profile/")) return true;
      if (path === "/v2/fast-match/teaser" || path.startsWith("/v2/fast-match/teaser/")) return true;
    } catch (_) {}
    return false;
  }

  function captureShouldCapture(url) {
    const value = String(url || "");
    return Boolean(
      NETWORK_CAPTURE_ENABLED &&
        value &&
        !captureIsLocalServer(value) &&
        !value.startsWith("chrome-extension:") &&
        captureIsUsefulTinderUrl(value)
    );
  }

  function capturePost(payload) {
    if (!captureShouldCapture(payload?.url)) return;
    try {
      window.postMessage(
        {
          source: BRIDGE_SOURCE,
          kind: PAGE_MSG_NETWORK_CAPTURE,
          payload: {
            schema_version: 1,
            page_url: location.href,
            page_host: location.hostname,
            captured_at: captureNowIso(),
            ...payload,
          },
        },
        "*"
      );
    } catch (_) {}
  }

  function captureFetchUrl(args) {
    try {
      const input = args[0];
      if (typeof input === "string") return input;
      if (input instanceof URL) return input.href;
      return input?.url || "";
    } catch (_) {
      return "";
    }
  }

  function captureFetchMethod(args) {
    try {
      const input = args[0];
      const init = args[1] || {};
      return String(init.method || input?.method || "GET").toUpperCase();
    } catch (_) {
      return "GET";
    }
  }

  function captureFetchHeaders(args) {
    try {
      const input = args[0];
      const init = args[1] || {};
      return {
        ...captureHeadersToObject(input?.headers),
        ...captureHeadersToObject(init.headers),
      };
    } catch (_) {
      return {};
    }
  }

  async function captureBodyFromValue(value) {
    try {
      if (value == null) return {};
      if (typeof value === "string") return captureTruncateText(value);
      if (value instanceof URLSearchParams) return captureTruncateText(value.toString());
      if (value instanceof FormData) {
        const fields = {};
        value.forEach((fieldValue, key) => {
          fields[key] = fieldValue instanceof File ? `[file:${fieldValue.name}:${fieldValue.size}]` : String(fieldValue);
        });
        return { body_json: fields, body_type: "form_data" };
      }
      if (value instanceof Blob) {
        if (value.size > NETWORK_CAPTURE_MAX_BODY_CHARS) {
          return { body_type: "blob", body_length: value.size, body_truncated: true };
        }
        return { body_type: "blob", ...captureTruncateText(await value.text()) };
      }
      if (value instanceof ArrayBuffer) {
        return { body_type: "array_buffer", body_length: value.byteLength, body_truncated: true };
      }
      if (ArrayBuffer.isView(value)) {
        return { body_type: "typed_array", body_length: value.byteLength, body_truncated: true };
      }
      return captureTruncateText(JSON.stringify(value));
    } catch (err) {
      return { body_error: String(err?.message || err) };
    }
  }

  async function captureFetchRequestBody(args) {
    try {
      const init = args[1] || {};
      if (init.body != null) return captureBodyFromValue(init.body);
      const input = args[0];
      if (input instanceof Request && input.method && input.method.toUpperCase() !== "GET") {
        return captureBodyFromValue(await input.clone().text());
      }
    } catch (err) {
      return { body_error: String(err?.message || err) };
    }
    return {};
  }

  function captureIsTextResponse(response) {
    try {
      const ct = response.headers.get("content-type") || "";
      if (!ct) return true;
      return /json|text|javascript|xml|graphql|x-www-form-urlencoded/i.test(ct);
    } catch (_) {
      return false;
    }
  }

  function calcAge(birthDateStr) {
    if (!birthDateStr) return 0;
    try {
      const birth = new Date(birthDateStr);
      const today = new Date();
      let age = today.getFullYear() - birth.getFullYear();
      const m = today.getMonth() - birth.getMonth();
      if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
      return age;
    } catch (_) {
      return 0;
    }
  }

  const patchedFetch = async function (...args) {
    const captureId =
      "fetch-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
    const captureStarted = performance.now();
    const captureUrl = captureFetchUrl(args);
    const shouldCapture = captureShouldCapture(captureUrl);
    const captureMethod = shouldCapture ? captureFetchMethod(args) : "GET";
    const captureHeaders = shouldCapture ? captureFetchHeaders(args) : {};
    const captureRequestBodyPromise = shouldCapture ? captureFetchRequestBody(args) : Promise.resolve({});

    let response;
    try {
      response = await originalFetch(...args);
    } catch (err) {
      if (shouldCapture) {
        captureRequestBodyPromise.then((requestBody) => {
          capturePost({
            capture_type: "fetch",
            request_id: captureId,
            phase: "error",
            url: captureUrl,
            method: captureMethod,
            request_headers: captureHeaders,
            request: requestBody,
            error: String(err?.message || err),
            duration_ms: Math.round(performance.now() - captureStarted),
          });
        });
      }
      throw err;
    }

    if (shouldCapture) {
      const responseClone = response.clone();
      Promise.all([
        captureRequestBodyPromise,
        captureIsTextResponse(responseClone) ? responseClone.text().then(captureTruncateText).catch((err) => ({ body_error: String(err?.message || err) })) : Promise.resolve({ body_omitted: "non_text_response" }),
      ]).then(([requestBody, responseBody]) => {
        capturePost({
          capture_type: "fetch",
          request_id: captureId,
          phase: "complete",
          url: captureUrl,
          method: captureMethod,
          status: response.status,
          status_text: response.statusText,
          ok: response.ok,
          redirected: response.redirected,
          request_headers: captureHeaders,
          response_headers: captureHeadersToObject(response.headers),
          request: requestBody,
          response: responseBody,
          duration_ms: Math.round(performance.now() - captureStarted),
        });
      });
    }

    try {
      const url = typeof args[0] === "string" ? args[0] : args[0]?.url ?? "";

      if (captureIsRecsUrl(url)) {
        const clone = response.clone();
        clone
          .json()
          .then((data) => {
            const results = data?.data?.results ?? [];
            const hasProfiles = results.some((r) => r.type === "user");

            if (!hasProfiles) return;

            // Envia o batch completo para classificação
            window.postMessage(
              { source: BRIDGE_SOURCE, kind: PAGE_MSG_RECS, payload: data },
              "*"
            );

            // Constrói mapa folder_id → {name, tinderId} para identificar
            // o perfil visível na tela pelo URL da foto (mais confiável que ler o DOM)
            const profileMap = {};
            for (const result of results) {
              if (result.type !== "user") continue;
              const user = result.user ?? {};
              const name = user.name ?? "";
              const tinderId = user._id ?? "";
              const age = calcAge(user.birth_date ?? "");

              for (const photo of user.photos ?? []) {
                // Extrai folder_id do padrão: /u/{folder_id}/...
                const extract = (url) => {
                  const m = (url ?? "").match(/\/u\/([A-Za-z0-9]+)\//);
                  return m ? m[1] : null;
                };

                const mainFolder = extract(photo.url);
                if (mainFolder) profileMap[mainFolder] = { name, tinderId, age };

                for (const pf of photo.processedFiles ?? []) {
                  const f = extract(pf.url);
                  if (f) profileMap[f] = { name, tinderId, age };
                }
              }
            }

            window.postMessage(
              { source: BRIDGE_SOURCE, kind: PAGE_MSG_PROFILE_MAP, payload: profileMap },
              "*"
            );
          })
          .catch(() => {});
      }
    } catch (_) {}

    return response;
  };

  // Faz window.fetch.toString() retornar o mesmo que o fetch nativo
  // Isso evita detecção por verificação de integridade da função
  Object.defineProperty(patchedFetch, "toString", {
    value: () => "function fetch() { [native code] }",
    writable: false,
    configurable: false,
  });
  Object.defineProperty(patchedFetch, "name", {
    value: "fetch",
    writable: false,
    configurable: false,
  });

  window.fetch = patchedFetch;
})();
