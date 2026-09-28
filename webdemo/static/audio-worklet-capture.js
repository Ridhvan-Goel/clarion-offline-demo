// AudioWorkletProcessor: runs on the audio render thread, not the main thread.
// Buffers incoming mono samples (already at the AudioContext's sample rate -- the
// context itself is created as new AudioContext({sampleRate:16000}) in app.js, so the
// browser resamples the mic track before it ever reaches here) into fixed-size
// batches and posts them to the main thread, which forwards them over the /ws/live
// WebSocket. Batch size is a multiple of the model's hop (256 samples @ 16kHz = 16ms)
// to keep server-side framing simple; default 1024 = 4 hops = 64ms per message,
// balancing responsiveness against per-message overhead.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.batchSize = opts.batchSize || 1024;
    this.buffer = new Float32Array(this.batchSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0];
      for (let i = 0; i < ch.length; i++) {
        this.buffer[this.offset++] = ch[i];
        if (this.offset >= this.batchSize) {
          this.port.postMessage(this.buffer.slice(0, this.offset));
          this.offset = 0;
        }
      }
    }
    return true; // keep the processor alive
  }
}

registerProcessor("capture-processor", CaptureProcessor);
