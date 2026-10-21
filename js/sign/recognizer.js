// Recognizer interface. Every recognizer takes a window of HandFrames and returns a Prediction:
//   { status: 'ok' | 'no_model' | 'not_configured' | 'error',
//     label?: string,            // Thai text for the recognized sign
//     confidence?: number,       // 0..1, NOT a guarantee of correctness
//     alternatives?: [{ label, confidence }],
//     source: 'offline' | 'online' }
// A trained on-device model (e.g. a TFLite/LiteRT model run via a WASM runtime) plugs in by
// implementing predict(); the UI does not change.

export const UNCERTAIN_BELOW = 0.6;

export class OfflineRecognizer {
  constructor(model = null) {
    this.model = model;
  }
  get available() {
    return this.model != null;
  }
  async predict(frames) {
    if (!this.model) return { status: 'no_model', source: 'offline' };
    const res = await this.model.predict(frames);
    return { status: 'ok', source: 'offline', ...res };
  }
}

/**
 * Sends only landmark coordinates (never video) to a server.
 * Request:  POST { version: 1, frames: HandFrame[] }
 * Response: { label, confidence, alternatives? }
 */
export class OnlineRecognizer {
  constructor(endpoint = '', { timeoutMs = 3000, fetchImpl = globalThis.fetch } = {}) {
    this.endpoint = endpoint;
    this.timeoutMs = timeoutMs;
    this.fetch = fetchImpl;
  }
  get available() {
    return Boolean(this.endpoint);
  }
  async predict(frames) {
    if (!this.endpoint) return { status: 'not_configured', source: 'online' };
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const res = await this.fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version: 1, frames }),
        signal: ctrl.signal
      });
      if (!res.ok) return { status: 'error', source: 'online', error: `HTTP ${res.status}` };
      const body = await res.json();
      if (typeof body.label !== 'string') return { status: 'error', source: 'online', error: 'bad response' };
      return { status: 'ok', source: 'online', label: body.label, confidence: Number(body.confidence) || 0, alternatives: body.alternatives ?? [] };
    } catch (err) {
      return { status: 'error', source: 'online', error: err.name === 'AbortError' ? 'timeout' : err.message };
    } finally {
      clearTimeout(timer);
    }
  }
}

/** Picks a recognizer per mode. Hybrid prefers on-device, falls back to online, then reports why neither worked. */
export class RecognizerHub {
  constructor(offline, online) {
    this.offline = offline;
    this.online = online;
    this.mode = 'offline';
  }
  async predict(frames) {
    if (this.mode === 'offline') return this.offline.predict(frames);
    if (this.mode === 'online') return this.online.predict(frames);
    if (this.offline.available) return this.offline.predict(frames);
    if (this.online.available) return this.online.predict(frames);
    return { status: 'no_model', source: 'offline' };
  }
}

export function isUncertain(prediction) {
  return prediction.status === 'ok' && (prediction.confidence ?? 0) < UNCERTAIN_BELOW;
}
