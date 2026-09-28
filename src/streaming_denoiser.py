"""Transport-agnostic incremental ring-buffer denoiser -- numpy + onnxruntime only.

This is the ONE place that knows how to turn a live stream of raw audio samples into
a live stream of enhanced audio samples. It knows nothing about WebSockets, FastAPI,
microphones, or files -- callers ("transport adapters": a file adapter, a live-mic
WebSocket adapter, a file-fragment WebSocket adapter, a paced fallback-demo adapter)
all just call `push_samples()`/`flush()` and get numpy arrays back. That separation is
what keeps a future Kria/on-device adapter a small addition instead of a rewrite.

WHY THIS IS DIFFERENT FROM scripts/onnx_denoise.py
`onnx_denoise.py` already steps the ONNX model one STFT frame at a time (real
streaming), but its STFT/iSTFT are computed over the WHOLE clip up front -- it cannot
produce a single output sample until the entire input is known. This module replaces
that with a true incremental sliding-window STFT and overlap-add, so output samples
are available `hop_length` (16 ms) after enough new input has arrived, continuously,
with no whole-clip buffering and no knowledge of how long the stream will run.

INPUT FORMAT CONTRACT (enforced, not assumed)
`push_samples()` requires 1-D float32 PCM in [-1, 1] at the model's sample rate
(16,000 Hz), already downmixed to mono. It does NOT resample or downmix for you --
that is the transport adapter's explicit, logged responsibility (see
`downmix_and_resample()` below, used by adapters that receive something else, e.g. a
48 kHz stereo mic capture). A wrong-shaped or wrong-dtype array raises
`AudioFormatError` immediately rather than being silently coerced into the model.

DEEP-FILTER RING BUFFER
The shipped model (`checkpoints/model.json`) has `df_lookahead: 0` and `df_order: 5`:
each output frame is a weighted sum of the CURRENT + 4 PAST raw noisy spectrum frames
(never future ones), so a live stream can emit every output frame immediately with no
added algorithmic latency. The K=5-frame ring buffer holds raw noisy spectra, not
model outputs -- this mirrors `src/infer_stream.py::stream_infer`'s proven
`ring_r`/`ring_i` pattern exactly.

STATE RESET
Per the documented finding in `docs/PROGRESS.md` (unbounded GRU state over long
sequences costs ~1.8 dB SI-SNR vs. periodic reset), hidden state resets every
`clip_seconds` (the checkpoint's training clip length, default 3.0 s) of processed
audio. The deep-filter ring buffer is never reset, matching `infer_stream.py`.

KNOWN, DOCUMENTED LIMITATION vs. the offline path
Offline STFT uses `center=True` reflect-padding at BOTH ends (the first frame reflects
`n_fft/2` samples of FUTURE audio backwards in time; the last frame does the same at
the tail). A live stream never has that future audio at either end, so streaming mode
zero-pads instead, at both the very start (see `_StreamState.pad_to_skip`, which also
corrects a one-hop *delay* the zero-padding would otherwise introduce -- get this
wrong and every output sample lands a full hop away from its true position, which
reads as near-total decorrelation, not a subtle error) and the tail (`flush()`).
Because the GRU is recurrent, the small head-boundary deviation does not decay after
a few frames -- it propagates through hidden state for the rest of the stream at a
low, roughly constant level. Measured on real audio
(`scripts/verify_streaming_parity.py`): ~35-40 dB SNR vs. the offline reference for
the whole stream, essentially uniform rather than concentrated at the boundaries. That
is far above the model's own enhancement SNR (~14 dB, see `docs/PROGRESS.md`), so it
is not the dominant source of error in the pipeline -- but it is real, it is not
bit-identical, and this docstring says so rather than claiming otherwise.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - scipy is a hard requirement.txt dep, but fail loud not silent
    resample_poly = None


class AudioFormatError(ValueError):
    """Raised when audio handed to the streaming core violates its input contract."""


def ensure_pcm_contract(x: np.ndarray) -> np.ndarray:
    """Validate (not coerce-and-hope) that `x` is 1-D float32 PCM roughly in [-1, 1].

    Raises AudioFormatError with a specific reason rather than letting a wrong dtype
    or an unscaled int16-range array enter the model silently."""
    if not isinstance(x, np.ndarray):
        raise AudioFormatError(f"expected a numpy array, got {type(x)}")
    if x.ndim != 1:
        raise AudioFormatError(
            f"expected mono 1-D samples, got shape {x.shape} -- downmix before calling "
            f"push_samples() (see downmix_and_resample())")
    if x.dtype != np.float32:
        raise AudioFormatError(
            f"expected float32 PCM in [-1, 1], got dtype {x.dtype} -- if this came from "
            f"int16 PCM, divide by 32768.0 and cast to float32 before calling")
    if x.size and np.abs(x).max() > 4.0:
        # a genuine float32 [-1,1] signal never gets near this; an un-normalised int16
        # array cast to float32 without scaling will blow past it immediately.
        raise AudioFormatError(
            f"sample magnitude up to {np.abs(x).max():.1f} is far outside [-1, 1] -- "
            f"looks like unscaled integer PCM was cast to float32 without dividing by "
            f"its full-scale value first")
    return np.ascontiguousarray(x, dtype=np.float32)


def downmix_and_resample(x: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Explicit, logged format conversion for transport adapters that receive audio in
    a foreign format, e.g. a raw browser capture at 48 kHz stereo:

        48kHz stereo -> downmix -> resample -> 16kHz mono -> push_samples()

    `x`: (n_samples,) mono or (n_samples, n_channels). Never called implicitly by
    `push_samples()` -- the core only ever accepts the target contract."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == target_sr:
        return x
    if resample_poly is None:
        raise RuntimeError("scipy is required for resample_poly (adapter-side resampling)")
    from math import gcd
    g = gcd(int(orig_sr), int(target_sr))
    up, down = target_sr // g, orig_sr // g
    return resample_poly(x, up, down).astype(np.float32)


def sqrt_hann(win_length: int) -> np.ndarray:
    n = np.arange(win_length)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)
    return np.sqrt(np.clip(hann, 1e-12, None)).astype(np.float32)


@dataclass
class _StreamTelemetry:
    """Latency/queue telemetry -- surfaced to both server logs and the demo-mode UI so
    "it works" can be replaced with actual numbers (queue depth, drops, RTF), per the
    project's demo-reliability requirements, not just a subjective listen."""
    hops_processed: int = 0
    gru_resets: int = 0
    total_model_ms: float = 0.0
    last_model_ms: float = 0.0
    dropped_chunks: int = 0        # transport adapters increment this (see server_local.py)
    started_at: float = field(default_factory=time.time)
    _recent_ms: deque = field(default_factory=lambda: deque(maxlen=500))

    def record_hop(self, model_seconds: float) -> None:
        self.hops_processed += 1
        self.last_model_ms = model_seconds * 1000.0
        self.total_model_ms += self.last_model_ms
        self._recent_ms.append(self.last_model_ms)

    def as_dict(self, hop_seconds: float) -> dict:
        avg_ms = self.total_model_ms / self.hops_processed if self.hops_processed else 0.0
        recent = np.array(self._recent_ms) if self._recent_ms else np.zeros(1)
        return {
            "hops_processed": self.hops_processed,
            "audio_seconds_processed": round(self.hops_processed * hop_seconds, 3),
            "gru_resets": self.gru_resets,
            "dropped_chunks": self.dropped_chunks,
            "avg_model_ms": round(avg_ms, 4),
            "last_model_ms": round(self.last_model_ms, 4),
            "p50_model_ms": round(float(np.percentile(recent, 50)), 4),
            "p99_model_ms": round(float(np.percentile(recent, 99)), 4),
            "frame_duration_ms": round(hop_seconds * 1000, 3),
            "real_time_factor": round(avg_ms / (hop_seconds * 1000), 4) if hop_seconds else None,
            "wall_seconds_since_start": round(time.time() - self.started_at, 2),
        }


class _StreamState:
    """Opaque per-connection state. Created only via StreamingDenoiser.new_stream()."""

    def __init__(self, n_fft: int, hop: int, K: int, n_bins: int, gru_layers: int, gru_hidden: int):
        self.pending = np.zeros(0, dtype=np.float32)
        self.carry = np.zeros(n_fft - hop, dtype=np.float32)     # causal zero-pad start (see module docstring)
        self.hidden = np.zeros((gru_layers, 1, gru_hidden), dtype=np.float32)
        self.ring_r = np.zeros((K, n_bins), dtype=np.float32)    # newest-first raw noisy spectra
        self.ring_i = np.zeros((K, n_bins), dtype=np.float32)
        self.ola_acc = np.zeros(n_fft, dtype=np.float64)
        self.ola_wsum = np.zeros(n_fft, dtype=np.float64)
        self.frame_idx = 0
        # frame 0's window covers conceptual positions [-(n_fft-hop), hop) -- the first
        # (n_fft-hop) samples of OLA output correspond entirely to the zero-padded
        # prefix (see module docstring), not real audio, and must be discarded rather
        # than returned, or every real output sample would be off by (n_fft-hop)
        # samples relative to its true position (a full-signal decorrelation, not a
        # subtle error -- caught by scripts/verify_streaming_parity.py).
        self.pad_to_skip = n_fft - hop
        self.telemetry = _StreamTelemetry()
        self.closed = False


class StreamingDenoiser:
    """Loads the ONNX session + sidecar ONCE; safe to share across many concurrent
    streams (onnxruntime InferenceSession.run() is thread-safe for concurrent calls).
    Per-connection mutable state lives entirely in the _StreamState this hands out."""

    def __init__(self, onnx_path: str, sidecar_path: str | None = None,
                 clip_seconds: float | None = 3.0, providers=None):
        meta_path = sidecar_path or (os.path.splitext(onnx_path)[0] + ".json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.n_fft = self.meta["n_fft"]
        self.hop = self.meta["hop_length"]
        self.win_length = self.meta["win_length"]
        self.sample_rate = self.meta["sample_rate"]
        self.n_bins = self.n_fft // 2 + 1
        self.head = self.meta["head"]
        self.K = self.meta["df_order"] if self.head == "deepfilter" else 1
        self.lookahead = self.meta.get("df_lookahead", 0)
        if self.lookahead:
            raise NotImplementedError(
                f"df_lookahead={self.lookahead} would require holding back output "
                f"frames in a live stream; the shipped model uses 0, this class only "
                f"implements the zero-lookahead (fully causal) case")
        self.gru_layers = self.meta["gru_layers"]
        self.gru_hidden = self.meta["gru_hidden"]
        self.in0, self.in1 = self.meta["input_names"]
        self.window = sqrt_hann(self.win_length)
        self.clip_seconds = clip_seconds
        self.reset_every = int(clip_seconds * self.sample_rate / self.hop) if clip_seconds else 0

        so = ort.SessionOptions()
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=providers or ["CPUExecutionProvider"])

    def new_stream(self) -> _StreamState:
        return _StreamState(self.n_fft, self.hop, self.K, self.n_bins,
                             self.gru_layers, self.gru_hidden)

    def _run_one_hop(self, state: _StreamState, hop_samples: np.ndarray) -> np.ndarray:
        frame_in = np.concatenate([state.carry, hop_samples])          # (n_fft,)
        state.carry = frame_in[-(self.n_fft - self.hop):] if self.n_fft > self.hop else state.carry

        if self.reset_every and state.frame_idx > 0 and state.frame_idx % self.reset_every == 0:
            state.hidden = np.zeros_like(state.hidden)
            state.telemetry.gru_resets += 1

        spec = np.fft.rfft(frame_in * self.window)                     # (n_bins,) complex64-ish
        state.ring_r = np.roll(state.ring_r, 1, axis=0)
        state.ring_i = np.roll(state.ring_i, 1, axis=0)
        state.ring_r[0] = spec.real
        state.ring_i[0] = spec.imag

        x_t = np.stack([spec.real, spec.imag], axis=0).astype(np.float32)[None, :, :, None]  # (1,2,F,1)
        t0 = time.perf_counter()
        o_a, o_b, state.hidden = self.session.run(
            None, {self.in0: x_t, self.in1: state.hidden})
        state.telemetry.record_hop(time.perf_counter() - t0)
        state.frame_idx += 1

        if self.head == "deepfilter":
            coef = o_a[0, :, :, 0] + 1j * o_b[0, :, :, 0]               # (K, n_bins)
            hist = state.ring_r + 1j * state.ring_i                     # (K, n_bins), newest-first
            enh_spec = (coef * hist).sum(axis=0)                        # (n_bins,)
        else:
            mask = o_a[0, :, 0] + 1j * o_b[0, :, 0]
            enh_spec = spec * mask

        frame_out = np.fft.irfft(enh_spec, n=self.n_fft).astype(np.float64) * self.window

        state.ola_acc[:] += frame_out
        state.ola_wsum[:] += (self.window.astype(np.float64) ** 2)
        finalized_acc = state.ola_acc[:self.hop].copy()
        finalized_wsum = state.ola_wsum[:self.hop].copy()
        nz = finalized_wsum > 1e-10
        out = np.zeros(self.hop, dtype=np.float32)
        out[nz] = (finalized_acc[nz] / finalized_wsum[nz]).astype(np.float32)

        state.ola_acc = np.concatenate([state.ola_acc[self.hop:], np.zeros(self.hop)])
        state.ola_wsum = np.concatenate([state.ola_wsum[self.hop:], np.zeros(self.hop)])
        return out

    def push_samples(self, state: _StreamState, new_samples: np.ndarray) -> np.ndarray:
        """Feed newly-arrived audio samples (any length, any chunking); returns however
        many enhanced output samples are now finalized (may be empty -- e.g. if fewer
        than `hop_length` new samples have accumulated so far)."""
        if state.closed:
            raise RuntimeError("push_samples() called on a closed/flushed stream")
        new_samples = ensure_pcm_contract(new_samples)
        state.pending = np.concatenate([state.pending, new_samples])
        return self._drain(state)

    def _drain(self, state: _StreamState, pad_final: bool = False) -> np.ndarray:
        if pad_final and 0 < len(state.pending) < self.hop:
            state.pending = np.concatenate(
                [state.pending, np.zeros(self.hop - len(state.pending), dtype=np.float32)])
        outs = []
        while len(state.pending) >= self.hop:
            hop_samples = state.pending[:self.hop]
            state.pending = state.pending[self.hop:]
            out = self._run_one_hop(state, hop_samples)
            if state.pad_to_skip > 0:
                skip = min(state.pad_to_skip, len(out))
                state.pad_to_skip -= skip
                out = out[skip:]
            if len(out):
                outs.append(out)
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs)

    def flush(self, state: _StreamState) -> np.ndarray:
        """Call once when a stream ends (mic stopped / file fully fed). Zero-pads any
        leftover partial hop and drains the remaining overlap-add tail. The state is
        unusable after this (create a new one via new_stream() to process more)."""
        if state.closed:
            return np.zeros(0, dtype=np.float32)
        tail_frames = self._drain(state, pad_final=True)
        tail_len = self.n_fft - self.hop
        wsum = state.ola_wsum[:tail_len]
        acc = state.ola_acc[:tail_len]
        nz = wsum > 1e-10
        tail = np.zeros(tail_len, dtype=np.float32)
        tail[nz] = (acc[nz] / wsum[nz]).astype(np.float32)
        state.closed = True
        return np.concatenate([tail_frames, tail])

    def get_stats(self, state: _StreamState) -> dict:
        hop_seconds = self.hop / self.sample_rate
        return state.telemetry.as_dict(hop_seconds)
