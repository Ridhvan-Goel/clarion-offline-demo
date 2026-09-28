const $ = (s) => document.querySelector(s);
const dropZone = $("#drop");
const fileInput = $("#file");

let busy = false;
let MODE = "kria"; // "kria" | "pc-cpu"
let uiMode = "live"; // "live" | "record" | "upload" -- live mic is the hero, defaults active
let LAST_HEALTH = null;

// Live mic mode state -- declared up here (not next to the functions that use them
// further down) because setUiMode("live") runs at load time via the bootstrap call
// below, and calls resetLiveClip()/stopLive() synchronously; those touch these `let`
// bindings, so they must already be past their TDZ by then, not merely hoisted.
let liveWs = null;
let liveCtx = null;
let liveNode = null;
let liveStream = null;
let livePlayer = null;
let liveSpectrumRAF = null;
const LIVE_CLIP_MAX_SAMPLES = 16000 * 60 * 5;
let liveClipChunks = [];
let liveClipSampleCount = 0;

// The browser's default getUserMedia({audio:true}) applies ITS OWN echo cancellation /
// noise suppression / auto-gain-control to the raw mic signal before anything else
// sees it -- a second, cruder processor stacked in front of the model, trained on
// none of this. Auto-gain "pumping" and the legacy noise gate are a well-known source
// of audible artifacts a downstream enhancer handles badly. Request genuinely raw
// audio instead; the model was trained on (and expects) unprocessed noisy input.
const RAW_AUDIO_CONSTRAINTS = {
  audio: {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 1,
  },
};

const ICON_RADIO = '<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><line x1="12" y1="3" x2="12" y2="13"/><circle cx="12" cy="16" r="1.4" fill="currentColor" stroke="none"/><path d="M8 8a5.5 5.5 0 0 1 8 0M5.3 5.3a9.5 9.5 0 0 1 13.4 0"/></svg>';
const ICON_MIC = '<svg viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor" stroke="none"><path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/></svg>';
const SCENARIO_ICONS = {
  military_radio: ICON_RADIO,
  parade: ICON_MIC,
  parade2: ICON_MIC,
};

function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

function micErrorMessage(e) {
  if (e && (e.name === "NotAllowedError" || e.name === "PermissionDeniedError")) {
    return "Microphone access was denied. Check browser permissions.";
  }
  if (e && e.name === "NotFoundError") {
    return "No microphone was found on this device.";
  }
  if (e && e.name === "NotReadableError") {
    return "The microphone is in use by another application.";
  }
  return "Could not access microphone: " + (e && e.message ? e.message : "unknown error");
}

// ---------------------------------------------------------------------------
// Waveform visualizer -- shared scrolling/static bar renderer. One instance per
// column (input / enhanced); used for BOTH the live/streaming view and, once a clip
// finishes, redrawn as the final full waveform -- never a second, separate canvas.
// Deliberately simple (level bars, not a full spectrogram) -- it exists to give an
// instant, honest "audio is flowing and being processed" visual, not a research plot.
// ---------------------------------------------------------------------------
class WaveformCanvas {
  constructor(canvas, colorVar = "--accent", maxBars = 110) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.maxBars = maxBars;
    this.levels = [];
    this.colorVar = colorVar;
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }
  _color() {
    return getComputedStyle(document.documentElement).getPropertyValue(this.colorVar).trim() || "#5b8cff";
  }
  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = Math.max(1, rect.width);
    this.h = Math.max(1, rect.height || 60);
    this.canvas.width = Math.max(1, this.w * dpr);
    this.canvas.height = Math.max(1, this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._draw();
  }
  push(level) {
    level = Math.max(0, Math.min(1, level));
    // Light temporal smoothing so consecutive bars flow into each other instead of
    // slamming between near-zero (silence) and full-height (any moderately loud
    // frame) -- each incoming chunk is a fairly short window, so raw per-chunk RMS is
    // naturally noisy; this keeps the shape while removing that jitter.
    const prev = this.levels.length ? this.levels[this.levels.length - 1] : level;
    const smoothed = prev * 0.4 + level * 0.6;
    this.levels.push(smoothed);
    if (this.levels.length > this.maxBars) this.levels.shift();
    this._draw();
  }
  setStatic(peaks) {
    this.levels = peaks.slice(0, this.maxBars);
    this._draw();
  }
  reset() {
    this.levels = [];
    this._draw();
  }
  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    if (!this.levels.length) {
      ctx.fillStyle = "rgba(154,161,178,0.45)"; // --dim, dimmed further
      ctx.font = "11px -apple-system,Segoe UI,sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("waiting for audio…", this.w / 2, this.h / 2 + 4);
      ctx.textAlign = "left";
      return;
    }
    const barW = this.w / this.maxBars;
    const mid = this.h / 2;
    ctx.fillStyle = this._color();
    const offset = this.maxBars - this.levels.length;
    for (let i = 0; i < this.levels.length; i++) {
      const x = (offset + i) * barW;
      const barH = Math.max(2, this.levels[i] * this.h * 0.92);
      ctx.fillRect(x, mid - barH / 2, Math.max(1, barW - 1.5), barH);
    }
  }
}

function rmsOf(f32) {
  let sum = 0;
  for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
  return Math.sqrt(sum / f32.length);
}
function rmsToLevel(rms) {
  // Perceptual compression (sqrt-ish curve): linear rms/peak scaling makes quiet
  // frames read as ~0 and anything moderately loud instantly slam to full height.
  // Raising to the power 0.55 lifts the quiet-to-mid range into visible territory
  // while still saturating at genuinely loud content.
  const norm = Math.min(1, rms / 0.2);
  return Math.pow(norm, 0.55);
}

// Real level meter -- same rms values already driving the waveform bars, also drawn as
// a horizontal fill + an actual dBFS readout (20*log10(rms), the standard digital-audio
// convention), not a decorative animation.
function setLevelMeter(fillId, dbId, rms) {
  const fill = $(fillId);
  const db = $(dbId);
  if (!fill || !db) return;
  fill.style.width = `${Math.round(rmsToLevel(rms) * 100)}%`;
  db.textContent = rms > 1e-6 ? `${(20 * Math.log10(rms)).toFixed(0)} dBFS` : "−∞ dBFS";
}

const inWave = new WaveformCanvas($("#inWave"), "--accent");
const outWave = new WaveformCanvas($("#outWave"), "--ok");

// Downsample a decoded AudioBuffer into `n` RMS peaks in [0,1] for a static waveform.
function peaksFromBuffer(buf, n) {
  const data = buf.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / n));
  const peaks = [];
  for (let i = 0; i < n; i++) {
    const start = i * step;
    let sum = 0;
    let count = 0;
    for (let j = start; j < Math.min(data.length, start + step); j++) {
      sum += data[j] * data[j];
      count++;
    }
    peaks.push(count ? rmsToLevel(Math.sqrt(sum / count)) : 0);
  }
  return peaks;
}
async function drawStaticWaveformFromUrl(canvasObj, url) {
  try {
    const buf = await fetch(url).then((r) => r.arrayBuffer());
    const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await tmpCtx.decodeAudioData(buf.slice(0));
    canvasObj.setStatic(peaksFromBuffer(audioBuf, canvasObj.maxBars));
    tmpCtx.close();
  } catch (e) {
    // non-fatal -- the audio player itself still works even if the waveform can't be drawn
  }
}
async function drawStaticWaveformFromFile(canvasObj, file) {
  try {
    const buf = await file.arrayBuffer();
    const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await tmpCtx.decodeAudioData(buf);
    canvasObj.setStatic(peaksFromBuffer(audioBuf, canvasObj.maxBars));
    tmpCtx.close();
  } catch (e) {
    // some recorded formats decode fine server-side via ffmpeg but not in-browser --
    // that's fine, this is a nice-to-have visual, not the actual processing path
  }
}

// ---------------------------------------------------------------------------
// Pipeline stage indicator -- honest simplification: highlights that audio is
// actively moving through the pipeline while something is running, not a
// per-stage-timed animation (those sub-stages aren't separately instrumented).
// ---------------------------------------------------------------------------
function setPipelineActive(active) {
  document.querySelectorAll(".pipeStage").forEach((el) => el.classList.toggle("active", active));
}

// ---------------------------------------------------------------------------
// The one Input/Enhanced audio section's title+state -- reflects real app state only.
// ---------------------------------------------------------------------------
function setAvState(state) {
  const title = $("#avTitle");
  const tag = $("#avStateTag");
  const map = {
    idle: ["LIVE AUDIO", null, null],
    live: ["LIVE AUDIO", "● LIVE", "live"],
    recording: ["LIVE AUDIO", "● RECORDING", "live"],
    processing: ["LIVE AUDIO", "● PROCESSING AUDIO", "live"],
    complete: ["BEFORE / AFTER", "● COMPLETE", "ok"],
    error: [null, "● ERROR", "warn"],
    disconnected: [null, "● DISCONNECTED", "warn"],
  };
  const [t, tg, cls] = map[state] || [null, null, null];
  if (t) title.textContent = t;
  if (tg) {
    tag.textContent = tg;
    tag.className = "tag " + cls;
    tag.classList.remove("hidden");
  } else {
    tag.classList.add("hidden");
  }
}

// Clears the section back to an idle state before a new job/session starts -- output
// player, spectrograms and download link only ever reflect the CURRENT run.
function resetAvSection() {
  outWave.reset();
  $("#outAudio").pause();
  $("#outAudio").classList.add("hidden");
  $("#outAudio").removeAttribute("src");
  $("#dl").classList.add("hidden");
  $("#specSection").classList.add("hidden");
  $("#offlineMetrics").classList.add("hidden");
  setLevelMeter("#inLevelFill", "#inLevelDb", 0);
  setLevelMeter("#outLevelFill", "#outLevelDb", 0);
  setAvState("idle");
}

async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    LAST_HEALTH = h;
    MODE = h.mode === "pc-cpu" ? "pc-cpu" : "kria";
    const s = $("#status");
    const detail = $("#statusDetail");
    const rt = h.runtime || {};
    if (MODE === "pc-cpu") {
      s.className = "pill pill-ok";
      s.textContent = "● SYSTEM READY";
      detail.textContent = [h.cpu, rt.execution_provider, rt.onnxruntime_version ? `ONNX Runtime ${rt.onnxruntime_version}` : null]
        .filter(Boolean).join(" · ");
    } else if (h.kria_online) {
      s.className = "pill pill-ok";
      s.textContent = "● KRIA ONLINE";
      detail.textContent = "ARM CPU · " + (h.kria_host || "");
    } else if (h.local_fallback) {
      s.className = "pill pill-local";
      s.textContent = "● Kria offline — CPU fallback";
      detail.textContent = h.cpu || "";
    } else {
      s.className = "pill pill-wait";
      s.textContent = "● Kria offline";
      detail.textContent = "";
    }
    detail.classList.toggle("hidden", !detail.textContent);
    $("#limits").textContent = h.max_seconds
      ? `WAV / MP3 / M4A — max ${h.max_seconds | 0}s`
      : "WAV / MP3 / M4A — no duration limit";

    const box = $("#sampleBtns");
    box.innerHTML = "";
    (h.samples || []).forEach((sm) => {
      const b = document.createElement("button");
      b.className = "scenarioCard";
      b.innerHTML = `<span class="scenarioIco">${SCENARIO_ICONS[sm.key] || ICON_MIC}</span>
                     <span class="scenarioLabel">${sm.label}</span>
                     <span class="scenarioTry">&#9654; Try</span>`;
      b.onclick = () => runSample(sm.key);
      box.appendChild(b);
    });

    fillTable($("#systemCard"), [
      ["Device", MODE === "kria" ? "Kria KV260" : "This computer"],
      ["Runtime", "ONNX Runtime" + (rt.onnxruntime_version ? ` ${rt.onnxruntime_version}` : "")],
      ["Execution", MODE === "kria" ? "ARM CPU" : (rt.execution_provider || h.cpu || "CPU")],
      ["Model", "Causal CRN + Order-5 Deep Filter"],
      ["Sample rate", "16 kHz"],
      ["Window", "32 ms"],
      ["Hop", "16 ms"],
      ["Look-ahead", "0 ms"],
    ]);
  } catch (e) {
    $("#status").textContent = "● server unreachable";
    $("#statusDetail").textContent = "";
    $("#statusDetail").classList.add("hidden");
  }
}

function setProgress(text) {
  $("#progress").classList.toggle("hidden", text === null);
  if (text !== null) $("#progressText").textContent = text;
  if (text === null) setProgressPct(0);
}
function setProgressPct(pct) {
  const el = $("#progressMeter");
  if (el) el.style.width = `${pct || 0}%`;
}
function showError(msg) {
  const e = $("#error");
  e.classList.remove("hidden");
  e.textContent = msg;
  setAvState("error");
}
function clearOutputs() {
  $("#error").classList.add("hidden");
}

function fillTable(el, rows) {
  el.innerHTML = "";
  rows.forEach(([k, v]) => {
    if (v === undefined || v === null || v === "") return;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
    el.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Real-time performance -- one canonical shape + renderer for both the live-streaming
// telemetry (/ws/live's periodic messages) and a finished job's final stats, so the
// same big numbers/real-time badge logic is never duplicated.
// ---------------------------------------------------------------------------
function statsFromLive(t) {
  return {
    rtf: t.real_time_factor, p50: t.p50_model_ms, p99: t.p99_model_ms,
    hop: t.frame_duration_ms, dropped: t.dropped_chunks, queueWait: null,
  };
}
function statsFromResult(r) {
  const s = r.stats || {};
  return {
    rtf: s.real_time_factor, p50: s.p50_latency_ms, p99: s.p99_latency_ms,
    hop: s.frame_duration_ms, dropped: s.dropped_chunks, queueWait: r.queue_waited_s,
  };
}
function applyStatGrid(c, diagRows) {
  $("#statRtf").textContent = c.rtf != null ? `${c.rtf}×` : "—";
  $("#statP50").textContent = c.p50 != null ? `${c.p50} ms` : "—";
  $("#statP99").textContent = c.p99 != null ? `${c.p99} ms` : "—";
  $("#statHop").textContent = c.hop != null ? `${c.hop} ms` : "—";
  $("#statDropped").textContent = c.dropped != null ? String(c.dropped) : "—";
  $("#statQueue").textContent = c.queueWait != null ? `${c.queueWait} s` : "—";

  const badge = $("#rtBadge");
  if (c.rtf != null) {
    badge.classList.remove("hidden");
    badge.className = "tag " + (c.rtf < 1 ? "ok" : "warn");
    badge.textContent = c.rtf < 1 ? "● REAL-TIME" : "● SLOWER THAN REAL-TIME";
  } else {
    badge.classList.add("hidden");
  }

  if (diagRows) fillTable($("#liveDiagTable"), diagRows);
}

// ---------------------------------------------------------------------------
// Signal Improvement -- the one shared "SIGNAL IMPROVEMENT" visualization, used both
// by a completed clip's (currently always-hidden, since no clean reference exists for
// demo audio) per-clip result AND the held-out evaluation band selector below. Gain is
// always COMPUTED as output-input here, never trusted from a separate stored field, so
// it can never drift out of sync with the two numbers actually shown.
// ---------------------------------------------------------------------------
const SNR_TARGET_DB = 15; // the project's stated output-SNR target, docs/PROGRESS.md

function renderTargetScale(outDb) {
  // Wording/coloring deliberately factual, not pass/fail: a band starting well below
  // 0 dB input that still gains 10+ dB is a real result even when the absolute output
  // lands under the reference line. Amber marks the reference line itself (a target,
  // not a verdict); the actual-value marker is blue (neutral) unless it clears the
  // line, in which case it's teal/ok -- never red, never "warning".
  const min = 0, max = 20;
  const pct = (v) => Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
  const met = outDb >= SNR_TARGET_DB;
  return `
    <div class="targetScale">
      <div class="targetScaleTrack">
        <div class="targetMarker ${met ? "ok" : ""}" style="left:${pct(outDb)}%" title="Output SNR: ${outDb} dB"></div>
        <div class="targetTick" style="left:${pct(SNR_TARGET_DB)}%"></div>
      </div>
      <div class="targetScaleAxis">
        <span>0</span><span>5</span><span>10</span><span>15</span><span>20&nbsp;dB</span>
      </div>
      <div class="targetScaleFoot">
        <span class="hint">${SNR_TARGET_DB} dB reference target</span>
        ${met
          ? `<span class="tag ok">&#10003; Above 15 dB reference</span>`
          : `<span class="hint">Target not reached in this input-SNR band</span>`}
      </div>
    </div>`;
}

function renderSignalImprovement(container, { inDb, outDb, stoi, pesq, showTarget }) {
  const gain = Math.round((outDb - inDb) * 100) / 100;
  const gainSign = gain >= 0 ? "+" : "";
  container.innerHTML = `
    <div class="siHeadline">Signal improvement</div>
    <div class="snrFlow">
      <div class="snrSide"><div class="snrVal">${inDb} dB</div><div class="statLabel">Input SNR</div></div>
      <div class="snrArrow">&#10230;</div>
      <div class="snrSide"><div class="snrVal">${outDb} dB</div><div class="statLabel">Output SNR</div></div>
    </div>
    <div class="snrGainRow">
      <div class="snrGain ${gain >= 0 ? "" : "neg"}">${gainSign}${gain} dB</div>
      <div class="statLabel">SNR gain</div>
    </div>
    ${showTarget ? renderTargetScale(outDb) : ""}
    ${(stoi != null || pesq != null) ? `
      <div class="statGrid" style="grid-template-columns:repeat(2,1fr);max-width:320px;margin:14px auto 0">
        <div class="statTile"><div class="statLabel">STOI</div><div class="statValue">${stoi}</div></div>
        <div class="statTile"><div class="statLabel">PESQ-NB</div><div class="statValue">${pesq}</div></div>
      </div>` : ""}`;
}

function render(r) {
  const s = r.stats || {};
  const p = r.provenance || {};
  const on = p.processed_on || {};
  const rt = p.runtime || {};

  // Input: the local preview player already has this exact audio (same bytes we
  // uploaded), so it's left as-is -- no second player for the same content.
  $("#outAudio").src = r.output_audio;
  $("#outAudio").classList.remove("hidden");
  $("#dl").href = r.output_audio;
  $("#dl").classList.remove("hidden");
  drawStaticWaveformFromUrl(outWave, r.output_audio); // redraw with the true final waveform

  if (r.input_spec && r.output_spec) {
    $("#inSpec").src = r.input_spec;
    $("#outSpec").src = r.output_spec;
    $("#specSection").classList.remove("hidden");
  }

  $("#statQueueLabel").textContent = "Queue wait";
  applyStatGrid(statsFromResult(r), [
    ["Model head", s.head],
    ["Frames processed", s.n_frames],
    ["GRU resets", s.gru_resets],
    ["Connection", "○ last run"],
  ]);

  // Enhancement result (SNR/STOI/PESQ for THIS clip): only rendered if the backend
  // actually supplied reference metrics for this specific clip -- it does not today,
  // since none of the demo/upload/sample audio has a clean reference to measure
  // against (see webdemo/README.md and the "Held-out evaluation" card below, which is
  // a separate, clearly-labeled project-level reference, not this clip's numbers).
  // Never fabricated here.
  const om = $("#offlineMetrics");
  if (r.offline_metrics) {
    const m = r.offline_metrics;
    om.innerHTML = `<div id="offlineMetricsBody"></div>`;
    renderSignalImprovement($("#offlineMetricsBody"), {
      inDb: m.input_snr_db, outDb: m.output_snr_db,
      stoi: m.stoi, pesq: m.pesq_nb, showTarget: true,
    });
    om.classList.remove("hidden");
  } else {
    om.classList.add("hidden");
  }

  fillTable($("#prov"), [
    ["Machine", r.mode === "pc-cpu" ? (on.os || "this computer") : (on.device_tree_model || on.os)],
    ["Host", on.hostname],
    ["Architecture", on.arch],
    ["CPU", on.cpu_model ? `${on.cpu_model} ×${on.cpu_count || "?"}` : null],
    ["Machine ID", on.machine_id],
    ["onnxruntime", rt.onnxruntime_version],
    ["Execution provider", (rt.session_providers || []).join(", ")],
    ["Output SHA-256", p.output && p.output.sha256 ? p.output.sha256.slice(0, 24) + "…" : null],
    ["Produced (UTC)", p.produced_utc],
  ]);

  setAvState("complete");
}

// ---------------------------------------------------------------------------
// Progressive upload -> WS fragment streaming (upload / sample / record / fallback)
// The file is saved (near-instant), then decode+inference+delivery are pipelined
// over /ws/stream_result/{uid} -- first audio arrives after the first decoded chunk,
// not after the entire file has been read.
// ---------------------------------------------------------------------------
async function streamProcess(uploadBodyOrQuery, { pace = "fast", label = "Processing" } = {}) {
  if (busy) return;
  busy = true;
  clearOutputs();
  resetAvSection();
  setAvState("processing");
  setProgress(`${label}…`);
  setPipelineActive(true);
  // Update the live waveform as fragments arrive, but don't play them through
  // speakers -- the enhanced result shouldn't start blasting out automatically the
  // moment it's ready. The user hits play on the Enhanced player once it's populated
  // in render() below. (Live Mic mode is different -- playback there IS the feature.)
  let ws = null;
  try {
    let uploadUrl = "/api/upload";
    const opts = { method: "POST" };
    if (typeof uploadBodyOrQuery === "string") uploadUrl += uploadBodyOrQuery;
    else opts.body = uploadBodyOrQuery;
    const upResp = await fetch(uploadUrl, opts);
    const up = await upResp.json();
    if (!upResp.ok) throw new Error(up.detail || upResp.statusText);

    const q = pace === "realtime" ? "?pace=realtime" : "";
    ws = new WebSocket(wsUrl(`/ws/stream_result/${up.uid}${q}`));
    ws.binaryType = "arraybuffer";

    const finalResult = await new Promise((resolve, reject) => {
      let settled = false;
      ws.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          const rms = rmsOf(new Float32Array(ev.data));
          outWave.push(rmsToLevel(rms));
          setLevelMeter("#outLevelFill", "#outLevelDb", rms);
          return;
        }
        const d = JSON.parse(ev.data);
        if (d.error) { settled = true; reject(new Error(d.error)); return; }
        if (d.done) { settled = true; resolve(d.result); return; }
        setProgress(d.pct != null ? `${label}… ${d.pct}%` : `${label}… (fragment ${d.seq})`);
        setProgressPct(d.pct != null ? d.pct : 0);
      };
      ws.onerror = () => { if (!settled) { settled = true; reject(new Error("streaming connection failed")); } };
      ws.onclose = () => { if (!settled) { settled = true; reject(new Error("connection closed before finishing")); } };
    });
    setProgress(null);
    render(finalResult);
  } catch (e) {
    setProgress(null);
    showError("Could not process this clip:\n" + e.message);
  } finally {
    if (ws) try { ws.close(); } catch (e) { /* already closed */ }
    setPipelineActive(false);
    busy = false;
  }
}

// Local, instant preview of the INPUT audio -- we already have the file/recording in
// the browser before it's even uploaded, so this needs no server round-trip and plays
// independently of (and simultaneously with) the enhanced audio streaming back. This
// is the ONLY input player on the page -- the finished job's input is the same bytes.
let _inPreviewUrl = null;
function setInputPreview(fileBlobOrUrl) {
  const el = $("#inPreviewAudio");
  if (_inPreviewUrl) { URL.revokeObjectURL(_inPreviewUrl); _inPreviewUrl = null; }
  if (fileBlobOrUrl instanceof Blob) {
    _inPreviewUrl = URL.createObjectURL(fileBlobOrUrl);
    el.src = _inPreviewUrl;
  } else {
    el.src = fileBlobOrUrl;
  }
  el.classList.remove("hidden");
}
function clearInputPreview() {
  const el = $("#inPreviewAudio");
  if (_inPreviewUrl) { URL.revokeObjectURL(_inPreviewUrl); _inPreviewUrl = null; }
  el.pause();
  el.removeAttribute("src");
  el.load();
  el.classList.add("hidden");
}

function runFile(f) {
  if (!f) return;
  inWave.reset();
  drawStaticWaveformFromFile(inWave, f);
  setInputPreview(f);
  const fd = new FormData();
  fd.append("file", f);
  streamProcess(fd, { label: "Uploading & processing" });
}
function runSample(key) {
  inWave.reset();
  const url = `/static/samples/${key}.wav`;
  drawStaticWaveformFromUrl(inWave, url);
  setInputPreview(url);
  streamProcess("?sample=" + encodeURIComponent(key), { label: "Processing sample" });
}

dropZone.onclick = () => fileInput.click();
$("#browse").onclick = (e) => { e.stopPropagation(); fileInput.click(); };
fileInput.onchange = () => runFile(fileInput.files[0]);
["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("hover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("hover"); }));
dropZone.addEventListener("drop", (e) => runFile(e.dataTransfer.files[0]));

// ---------------------------------------------------------------------------
// Mode switch
// ---------------------------------------------------------------------------
function setUiMode(mode) {
  if (uiMode === "live" && mode !== "live") stopLive();
  uiMode = mode;
  document.querySelectorAll(".modeBtn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("#uploadPanel").classList.toggle("hidden", mode !== "upload");
  $("#recordPanel").classList.toggle("hidden", mode !== "record");
  $("#livePanel").classList.toggle("hidden", mode !== "live");
  clearOutputs();
  inWave.reset();
  resetAvSection();
  $("#inBaLabel").textContent = mode === "live" ? "Live input" : "Input";
  $("#outBaLabel").textContent = mode === "live" ? "Live enhanced" : "Enhanced";
  if (mode === "live") {
    clearInputPreview(); // nothing to preview -- live mic has no local file
    resetLiveClip();
    $("#liveSpectrumWrap").classList.add("hidden");
  }
}
document.querySelectorAll(".modeBtn").forEach((b) => (b.onclick = () => setUiMode(b.dataset.mode)));
setUiMode("live");

// ---------------------------------------------------------------------------
// Demo mode -- strips the page down to the hero visualizer + live mic button for
// a competition presentation; nothing about the underlying pipeline changes.
// ---------------------------------------------------------------------------
$("#demoModeBtn").onclick = () => {
  const on = document.body.classList.toggle("demo-mode");
  $("#demoModeBtn").textContent = on ? "Exit demo mode" : "Demo mode";
};

// ---------------------------------------------------------------------------
// Record mode: MediaRecorder -> same progressive upload+stream path as a file drop
// ---------------------------------------------------------------------------
let mediaRecorder = null;
let recordedChunks = [];

async function toggleRecord() {
  const btn = $("#recordBtn");
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("Microphone capture needs a secure context (https:// or localhost) — "
      + "this page was opened over plain http from another device.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia(RAW_AUDIO_CONSTRAINTS);
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.textContent = "● Start recording";
      $("#recordStatus").textContent = "";
      const mime = mediaRecorder.mimeType || "audio/webm";
      const ext = mime.includes("ogg") ? "ogg" : mime.includes("mp4") ? "mp4" : "webm";
      const blob = new Blob(recordedChunks, { type: mime });
      inWave.reset();
      drawStaticWaveformFromFile(inWave, blob);
      setInputPreview(blob);
      const fd = new FormData();
      fd.append("file", blob, `recording.${ext}`);
      streamProcess(fd, { label: "Processing your recording" });
    };
    mediaRecorder.start();
    btn.textContent = "■ Stop recording";
    $("#recordStatus").textContent = "Recording…";
    setAvState("recording");
  } catch (e) {
    showError(micErrorMessage(e));
  }
}
$("#recordBtn").onclick = toggleRecord;

// ---------------------------------------------------------------------------
// Live mic mode: AudioWorklet capture -> /ws/live -> StreamingPlayer playback
// (state vars declared up top -- see the note there)
// ---------------------------------------------------------------------------

// Live frequency spectrum -- real AnalyserNode data from the actual enhanced audio
// that's playing (via StreamingPlayer.getFrequencyData()), not a synthetic animation.
function drawLiveSpectrum() {
  const canvas = $("#liveSpectrum");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, rect.width), h = Math.max(1, rect.height || 64);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  ctx.clearRect(0, 0, w, h);
  const data = livePlayer ? livePlayer.getFrequencyData() : null;
  if (data) {
    const okColor = getComputedStyle(document.documentElement).getPropertyValue("--ok").trim();
    const n = data.length;
    const barW = w / n;
    ctx.fillStyle = okColor;
    for (let i = 0; i < n; i++) {
      const barH = Math.max(1, (data[i] / 255) * h);
      ctx.fillRect(i * barW, h - barH, Math.max(1, barW - 1), barH);
    }
  }
  liveSpectrumRAF = requestAnimationFrame(drawLiveSpectrum);
}
function startLiveSpectrum() {
  $("#liveSpectrumWrap").classList.remove("hidden");
  if (!liveSpectrumRAF) drawLiveSpectrum();
}
function stopLiveSpectrum() {
  if (liveSpectrumRAF) { cancelAnimationFrame(liveSpectrumRAF); liveSpectrumRAF = null; }
}

// Save clip -- live audio is otherwise ephemeral (played once, then gone). Captures the
// SAME enhanced samples actually played, bounded to 5 minutes, and offers a real WAV
// download of them -- not a re-synthesis, the literal audio that was heard.
function resetLiveClip() {
  liveClipChunks = [];
  liveClipSampleCount = 0;
  $("#saveClipBtn").classList.add("hidden");
  $("#clipHint").textContent = "Capturing enhanced audio for download…";
}
function captureLiveClip(f32) {
  if (liveClipSampleCount >= LIVE_CLIP_MAX_SAMPLES) return;
  liveClipChunks.push(f32);
  liveClipSampleCount += f32.length;
}
function encodeWavBlob(floatSamples, sampleRate) {
  const n = floatSamples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF"); view.setUint32(4, 36 + n * 2, true); writeStr(8, "WAVE");
  writeStr(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, "data"); view.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, floatSamples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
$("#saveClipBtn").onclick = () => {
  if (!liveClipChunks.length) return;
  const total = new Float32Array(liveClipSampleCount);
  let off = 0;
  for (const c of liveClipChunks) { total.set(c, off); off += c.length; }
  const blob = encodeWavBlob(total, 16000);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `anc_live_clip_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
};

async function startLive() {
  const btn = $("#liveBtn");
  if (liveWs) { stopLive(); return; }
  $("#fallbackOffer").classList.add("hidden");
  clearOutputs();
  inWave.reset();
  resetAvSection();
  resetLiveClip();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("Microphone capture needs a secure context (https:// or localhost) — "
      + "this page was opened over plain http from another device. Use the demo "
      + "laptop's own browser, or a tunnel URL, for live mic.");
    $("#fallbackOffer").classList.remove("hidden");
    return;
  }
  if (!window.AudioWorklet) {
    showError("This browser doesn't support AudioWorklet — try a recent Chrome/Edge/Firefox.");
    $("#fallbackOffer").classList.remove("hidden");
    return;
  }

  try {
    liveStream = await navigator.mediaDevices.getUserMedia(RAW_AUDIO_CONSTRAINTS);
  } catch (e) {
    showError(micErrorMessage(e));
    $("#fallbackOffer").classList.remove("hidden");
    return;
  }

  try {
    liveCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    await liveCtx.audioWorklet.addModule("/static/audio-worklet-capture.js");
    const src = liveCtx.createMediaStreamSource(liveStream);
    liveNode = new AudioWorkletNode(liveCtx, "capture-processor", { processorOptions: { batchSize: 1024 } });
    src.connect(liveNode);
    // liveNode is deliberately NOT connected to liveCtx.destination -- we don't want
    // to hear our own raw mic, only the enhanced audio played back via livePlayer.

    livePlayer = new StreamingPlayer(16000);
    livePlayer.start();
    livePlayer.onLevel = (rms) => { outWave.push(rmsToLevel(rms)); setLevelMeter("#outLevelFill", "#outLevelDb", rms); };

    liveWs = new WebSocket(wsUrl("/ws/live"));
    liveWs.binaryType = "arraybuffer";
    liveWs.onopen = () => {
      $("#liveStatus").textContent = "● Live";
      setAvState("live");
      setPipelineActive(true);
      btn.textContent = "■ Stop live demo";
      btn.classList.add("btnLiveOn");
      $("#heroIcon").classList.add("live");
      startLiveSpectrum();
    };
    liveWs.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        const f32 = new Float32Array(ev.data);
        if (livePlayer) livePlayer.pushFloat32(f32);
        captureLiveClip(f32);
        return;
      }
      const d = JSON.parse(ev.data);
      if (d.error) { showError(d.error); stopLive(); return; }
      if (d.telemetry) {
        const stats = statsFromLive(d.telemetry);
        stats.queueWait = livePlayer ? Number(livePlayer.queuedSeconds().toFixed(2)) : null;
        $("#statQueueLabel").textContent = "Queue depth";
        applyStatGrid(stats, [
          ["GRU resets", d.telemetry.gru_resets],
          ["Audio processed", d.telemetry.audio_seconds_processed != null ? `${d.telemetry.audio_seconds_processed} s` : null],
          ["Connection", "● connected"],
          ["Input sample rate", "16,000 Hz"],
          ["Input channels", "1 (mono)"],
          ["Capture chunk size", "1024 samples (64 ms)"],
        ]);
      }
    };
    liveWs.onclose = () => stopLive();
    liveWs.onerror = () => { showError("Live connection failed"); stopLive(); };

    liveNode.port.onmessage = (ev) => {
      const chunk = ev.data; // Float32Array, already 16kHz mono from the resampled context
      const rms = rmsOf(chunk);
      inWave.push(rmsToLevel(rms));
      setLevelMeter("#inLevelFill", "#inLevelDb", rms);
      if (liveWs && liveWs.readyState === WebSocket.OPEN) liveWs.send(chunk.buffer);
    };
  } catch (e) {
    showError("Could not start live mic: " + e.message);
    stopLive();
  }
}

function stopLive() {
  if (liveNode) { liveNode.port.onmessage = null; try { liveNode.disconnect(); } catch (e) {} liveNode = null; }
  if (liveStream) { liveStream.getTracks().forEach((t) => t.stop()); liveStream = null; }
  if (liveCtx) { try { liveCtx.close(); } catch (e) {} liveCtx = null; }
  if (livePlayer) { livePlayer.stop(); livePlayer = null; }
  if (liveWs) { const ws = liveWs; liveWs = null; try { ws.close(); } catch (e) {} }
  const btn = $("#liveBtn");
  if (btn) { btn.textContent = "● Start live demo"; btn.classList.remove("btnLiveOn"); }
  $("#liveStatus").textContent = "";
  $("#heroIcon").classList.remove("live");
  stopLiveSpectrum();
  if (liveClipChunks.length) {
    const secs = (liveClipSampleCount / 16000).toFixed(1);
    $("#clipHint").textContent = `${secs}s captured`;
    $("#saveClipBtn").classList.remove("hidden");
  } else {
    $("#liveSpectrumWrap").classList.add("hidden");
  }
  setPipelineActive(false);
  // Don't override an error state that's already showing (showError() already set
  // avState to "error" before calling stopLive() in every error path) -- only fall
  // back to idle on a normal stop.
  const errorShown = !$("#error").classList.contains("hidden");
  if (!errorShown) setAvState("idle");
}
$("#liveBtn").onclick = startLive;

// Fallback demo: mic unavailable (permissions/drivers/venue Wi-Fi) -> replay a bundled
// sample through the SAME /ws/stream_result streaming pipeline, paced at real playback
// speed so it still visibly demonstrates streaming, not a different/static demo.
$("#fallbackBtn").onclick = () => {
  inWave.reset();
  drawStaticWaveformFromUrl(inWave, "/static/samples/parade.wav");
  setInputPreview("/static/samples/parade.wav");
  streamProcess("?sample=parade", { pace: "realtime", label: "Fallback demo (streaming pipeline)" });
};

// ---------------------------------------------------------------------------
// Held-out evaluation summary -- the project's real, previously-measured results on
// its held-out validation set (400 real-audio noisy/clean mixture pairs), broken out
// by input SNR band. Source: docs/PROGRESS.md, Phase H breakdown-by-input-SNR-band
// table (2026-09-18 session). Static reference data, not derived from any API call --
// deliberately NOT presented as this session's clip result (see the card's own label
// and the render() gating above, which never conflates the two). Gain/target-met are
// intentionally NOT stored here -- renderSignalImprovement() always derives them from
// inDb/outDb so they can never drift out of sync with the two numbers shown.
// ---------------------------------------------------------------------------
const HELD_OUT_EVAL_BANDS = [
  { band: "-5 to 0 dB", n: 107, inDb: -2.42, outDb: 8.88, stoi: 0.800, pesq: 1.955 },
  { band: "0 to 5 dB", n: 111, inDb: 2.57, outDb: 12.59, stoi: 0.876, pesq: 2.333 },
  { band: "5 to 10 dB", n: 87, inDb: 7.46, outDb: 15.98, stoi: 0.926, pesq: 2.642 },
  { band: "10 to 15 dB", n: 95, inDb: 12.88, outDb: 19.20, stoi: 0.950, pesq: 3.049 },
];
let evalBandIndex = 0;

function selectEvalBand(i) {
  evalBandIndex = i;
  document.querySelectorAll("#evalBandSwitch .modeBtn").forEach((b, bi) => b.classList.toggle("active", bi === i));
  const b = HELD_OUT_EVAL_BANDS[i];
  const wrap = document.createElement("div");
  wrap.innerHTML = `<p class="hint" style="margin:0 0 10px">Input SNR band ${b.band}, n=${b.n} pairs</p>`;
  const el = $("#evalSummary");
  el.innerHTML = "";
  el.appendChild(wrap.firstChild);
  const body = document.createElement("div");
  el.appendChild(body);
  renderSignalImprovement(body, { inDb: b.inDb, outDb: b.outDb, stoi: b.stoi, pesq: b.pesq, showTarget: true });
}

function initEvalBandSwitch() {
  const sw = $("#evalBandSwitch");
  sw.innerHTML = HELD_OUT_EVAL_BANDS.map((b, i) =>
    `<button class="modeBtn" data-i="${i}">${b.band.replace(" to ", "–").replace(" dB", "")}</button>`).join("");
  sw.querySelectorAll(".modeBtn").forEach((btn, i) => (btn.onclick = () => selectEvalBand(i)));
  selectEvalBand(0);
}
initEvalBandSwitch();

// Secondary, optional: output SI-SNR vs. input SNR across the four evaluation points,
// with the 15 dB target as a reference line. Deliberately small/collapsed -- supporting
// evidence for the band selector above, not a dashboard centerpiece.
function drawPerfCurve() {
  const canvas = $("#perfCurve");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, rect.width), h = Math.max(1, rect.height || 170);
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const lineColor = css.getPropertyValue("--line").trim();
  const dimColor = css.getPropertyValue("--dim").trim();
  const accentColor = css.getPropertyValue("--accent").trim();
  const okColor = css.getPropertyValue("--ok").trim();
  const warnColor = css.getPropertyValue("--warn").trim();

  const pad = { l: 34, r: 14, t: 12, b: 26 };
  const xmin = -5, xmax = 15, ymin = 0, ymax = 22;
  const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - ((v - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);

  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();

  // 15 dB target reference line
  ctx.strokeStyle = warnColor;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(pad.l, Y(SNR_TARGET_DB)); ctx.lineTo(w - pad.r, Y(SNR_TARGET_DB));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = warnColor;
  ctx.font = "9px -apple-system,Segoe UI,sans-serif";
  ctx.fillText(`${SNR_TARGET_DB} dB target`, pad.l + 4, Y(SNR_TARGET_DB) - 4);

  // measured points, connected in input-SNR order
  const pts = HELD_OUT_EVAL_BANDS.slice().sort((a, b) => a.inDb - b.inDb);
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((b, i) => { const x = X(b.inDb), y = Y(b.outDb); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke();
  pts.forEach((b) => {
    ctx.fillStyle = b.outDb >= SNR_TARGET_DB ? okColor : accentColor;
    ctx.beginPath(); ctx.arc(X(b.inDb), Y(b.outDb), 4, 0, Math.PI * 2); ctx.fill();
  });

  ctx.fillStyle = dimColor;
  ctx.font = "10px -apple-system,Segoe UI,sans-serif";
  ctx.fillText("Input SNR (dB)", w / 2 - 34, h - 6);
  ctx.save();
  ctx.translate(10, h / 2 + 36);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Output SI-SNR (dB)", 0, 0);
  ctx.restore();
}
$("#perfCurve").closest("details").addEventListener("toggle", (e) => { if (e.target.open) drawPerfCurve(); });
window.addEventListener("resize", () => { if ($("#perfCurve").closest("details").open) drawPerfCurve(); });

// Input and Enhanced are a before/after comparison -- hearing both at once defeats the
// point. Starting either one pauses the other; only one ever plays at a time. Each side
// also gets a visible "▶ Playing" indicator + highlighted column, since two otherwise
// identical-looking native audio players give no clue which one is actually making sound.
function wirePlaybackIndicator(audioEl, tagId, colId) {
  const tag = $(tagId);
  const col = $(colId);
  const setPlaying = (on) => {
    tag.classList.toggle("hidden", !on);
    col.classList.toggle("nowPlaying", on);
  };
  audioEl.addEventListener("play", () => setPlaying(true));
  audioEl.addEventListener("pause", () => setPlaying(false));
  audioEl.addEventListener("ended", () => setPlaying(false));
}
function preventSimultaneousPlayback(a, b) {
  a.addEventListener("play", () => { if (!b.paused) b.pause(); });
  b.addEventListener("play", () => { if (!a.paused) a.pause(); });
}
preventSimultaneousPlayback($("#inPreviewAudio"), $("#outAudio"));
wirePlaybackIndicator($("#inPreviewAudio"), "#inPlayingTag", "#inBaCol");
wirePlaybackIndicator($("#outAudio"), "#outPlayingTag", "#outBaCol");

loadHealth();
setInterval(loadHealth, 15000);

// deep link: /?try=gunshot  auto-runs that sample once on load (kiosk / QR use)
const _try = new URLSearchParams(location.search).get("try");
if (_try) window.addEventListener("load", () => setTimeout(() => { setUiMode("upload"); runSample(_try); }, 400));
