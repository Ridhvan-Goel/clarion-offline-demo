// Shared gapless playback for incoming Float32 PCM chunks at a fixed sample rate.
// Used by both the live-mic panel and the progressive file/record result panel, so
// there is one playback implementation, not two. Classic queued-AudioBufferSourceNode
// pattern: each chunk is scheduled to start exactly when the previous one ends.
class StreamingPlayer {
  constructor(sampleRate = 16000) {
    this.sampleRate = sampleRate;
    this.ctx = null;
    this.nextStartTime = 0;
    this.activeSources = [];
    this.onLevel = null; // optional: (rms:number) => void, for a level meter
    this.analyser = null; // real frequency-domain data, for the live spectrum visualizer
  }

  _ctx() {
    if (!this.ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx({ sampleRate: this.sampleRate });
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.75;
      this.analyser.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  // Uint8Array of frequency-bin magnitudes (0-255), straight from the actual audio
  // that's playing -- null if nothing has started yet. Never synthetic.
  getFrequencyData() {
    if (!this.analyser) return null;
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(data);
    return data;
  }

  start() {
    const ctx = this._ctx();
    this.nextStartTime = ctx.currentTime + 0.05; // small initial jitter buffer
  }

  // float32Array: samples in [-1, 1] at this.sampleRate
  pushFloat32(float32Array) {
    if (!float32Array || !float32Array.length) return;
    const ctx = this._ctx();
    const buf = ctx.createBuffer(1, float32Array.length, this.sampleRate);
    buf.copyToChannel(float32Array, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.analyser);
    const startAt = Math.max(this.nextStartTime, ctx.currentTime);
    src.start(startAt);
    this.nextStartTime = startAt + buf.duration;
    this.activeSources.push(src);
    src.onended = () => {
      this.activeSources = this.activeSources.filter((s) => s !== src);
    };
    if (this.onLevel) {
      let sum = 0;
      for (let i = 0; i < float32Array.length; i++) sum += float32Array[i] * float32Array[i];
      this.onLevel(Math.sqrt(sum / float32Array.length));
    }
  }

  // seconds of audio still queued ahead of the playhead -- a growing number means the
  // server is sending faster than it can be played (shouldn't happen at 16kHz mono,
  // but useful to expose for the telemetry panel).
  queuedSeconds() {
    if (!this.ctx) return 0;
    return Math.max(0, this.nextStartTime - this.ctx.currentTime);
  }

  stop() {
    this.activeSources.forEach((s) => {
      try { s.stop(); } catch (e) { /* already ended */ }
    });
    this.activeSources = [];
    this.nextStartTime = 0;
  }
}

window.StreamingPlayer = StreamingPlayer;
