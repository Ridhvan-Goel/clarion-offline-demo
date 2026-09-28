"""Standalone streaming denoiser for edge deployment -- numpy + onnxruntime only.

No torch, no checkpoint, no yaml. Needs just:
    - model.onnx           (from src/export_onnx.py)
    - model.json           (sidecar written next to it by the same script)
    - numpy, onnxruntime, soundfile

Runs the exported CRN one STFT frame at a time, carrying the GRU hidden state across
frames exactly as a live embedded deployment would, and reports the real-time factor
(mean model time per frame / frame duration; < 1.0 means it keeps up with live audio).

The STFT is computed over the whole clip up front (matching src/infer_stream.py) so the
result is bit-comparable with the torch reference; only the model call is streamed,
which is the dominant real-time cost. A production ring-buffer STFT is a drop-in change
to stft_stream()/istft_ola() and does not touch the model path.

Usage:
    python scripts/onnx_denoise.py --onnx checkpoints/model.onnx \
        --input_wav demo_out/gunshot_noisy.wav --output_wav out/gunshot_enh.wav
"""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_line(path, key):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def collect_provenance():
    """Everything identifying the machine this process is running on. On the Kria this
    reports the board; on a laptop it reports the laptop -- which is the point."""
    dt_model = None
    try:
        with open("/proc/device-tree/model", "rb") as f:
            dt_model = f.read().rstrip(b"\x00").decode(errors="replace")
    except OSError:
        pass
    mid = None
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            mid = open(p).read().strip()
            break
        except OSError:
            continue
    _arm_parts = {"0xd03": "ARM Cortex-A53", "0xd07": "ARM Cortex-A57",
                  "0xd08": "ARM Cortex-A72", "0xd09": "ARM Cortex-A73"}
    cpu_part = _first_line("/proc/cpuinfo", "CPU part")
    cpu_model = (_first_line("/proc/cpuinfo", "model name")
                 or _arm_parts.get(cpu_part or "")
                 or platform.processor() or "unknown")
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_part": cpu_part,
        "cpu_count": os.cpu_count(),
        "kernel": platform.release(),
        "device_tree_model": dt_model,     # e.g. "ZynqMP KV260 revB" -- a laptop cannot report this
        "machine_id": mid,
        "python": platform.python_version(),
    }


def sqrt_hann(win_length):
    """Periodic sqrt-Hann -- matches src/utils/audio.sqrt_hann_window / torch.hann_window(periodic=True)."""
    n = np.arange(win_length)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)
    return np.sqrt(np.clip(hann, 1e-12, None)).astype(np.float32)


def stft_stream(x, n_fft, hop, window):
    """center=True STFT matching torch.stft: reflect-pad n_fft//2 each side, then frame."""
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[idx] * window[None, :]                       # (T, n_fft)
    spec = np.fft.rfft(frames, n=n_fft, axis=1).T            # (F, T)
    return spec


def istft_ola(spec, n_fft, hop, window, length):
    """center=True iSTFT matching torch.istft: OLA with window^2 (NOLA) normalisation."""
    F_, T = spec.shape
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1)           # (T, n_fft)
    frames *= window[None, :]
    sig_len = n_fft + hop * (T - 1)
    out = np.zeros(sig_len, dtype=np.float64)
    wsum = np.zeros(sig_len, dtype=np.float64)
    w2 = (window ** 2).astype(np.float64)
    for i in range(T):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += w2
    nz = wsum > 1e-10
    out[nz] /= wsum[nz]
    pad = n_fft // 2
    return out[pad:pad + length].astype(np.float32)


def apply_deep_filter(X, coef, lookahead=0):
    """enh(f,t) = sum_i coef[i,f,t] * X(f, t - i + lookahead), out-of-range -> 0. Complex."""
    K = coef.shape[0]
    enh = np.zeros_like(X)
    for i in range(K):
        shift = i - lookahead
        if shift == 0:
            enh += coef[i] * X
        elif shift > 0:                                      # past frame
            enh[:, shift:] += coef[i][:, shift:] * X[:, :-shift]
        else:                                               # future frame (lookahead)
            s = -shift
            enh[:, :-s] += coef[i][:, :-s] * X[:, s:]
    return enh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="checkpoints/model.onnx")
    p.add_argument("--sidecar", default=None, help="defaults to <onnx>.json")
    p.add_argument("--input_wav", required=True)
    p.add_argument("--output_wav", default="out/enhanced.wav")
    p.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (0 = default)")
    p.add_argument("--no_provenance", action="store_true",
                   help="skip writing <output>.provenance.json")
    args = p.parse_args()

    wall_start = time.time()

    meta_path = args.sidecar or (os.path.splitext(args.onnx)[0] + ".json")
    with open(meta_path) as f:
        meta = json.load(f)
    n_fft, hop = meta["n_fft"], meta["hop_length"]
    sr_model = meta["sample_rate"]
    head = meta["head"]
    K = meta["df_order"]
    lookahead = meta["df_lookahead"]
    window = sqrt_hann(meta["win_length"])

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])

    noisy, sr = sf.read(args.input_wav, dtype="float32")
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=1)
    assert sr == sr_model, f"input {sr} Hz != model {sr_model} Hz -- resample first"

    X = stft_stream(noisy, n_fft, hop, window)               # (F, T) complex
    real = np.ascontiguousarray(X.real, dtype=np.float32)
    imag = np.ascontiguousarray(X.imag, dtype=np.float32)
    Fb, T = X.shape

    h = np.zeros((meta["gru_layers"], 1, meta["gru_hidden"]), dtype=np.float32)
    in0, in1 = meta["input_names"]

    if head == "deepfilter":
        cr = np.empty((K, Fb, T), dtype=np.float32)
        ci = np.empty((K, Fb, T), dtype=np.float32)
    else:
        mr = np.empty((Fb, T), dtype=np.float32)
        mi = np.empty((Fb, T), dtype=np.float32)

    lat = np.empty(T, dtype=np.float64)
    for t in range(T):
        x_t = np.stack([real[:, t:t + 1], imag[:, t:t + 1]], axis=0)[None]   # (1,2,F,1)
        t0 = time.perf_counter()
        o_a, o_b, h = sess.run(None, {in0: x_t, in1: h})
        lat[t] = time.perf_counter() - t0
        if head == "deepfilter":
            cr[:, :, t] = o_a[0, :, :, 0]
            ci[:, :, t] = o_b[0, :, :, 0]
        else:
            mr[:, t] = o_a[0, :, 0]
            mi[:, t] = o_b[0, :, 0]

    if head == "deepfilter":
        coef = cr + 1j * ci
        enh_spec = apply_deep_filter(X, coef, lookahead)
    else:
        mask = mr + 1j * mi
        enh_spec = X * mask

    enh = istft_ola(enh_spec, n_fft, hop, window, length=len(noisy))

    os.makedirs(os.path.dirname(args.output_wav) or ".", exist_ok=True)
    sf.write(args.output_wav, enh, sr)

    frame_ms = hop / sr_model * 1000.0
    stats = {
        "head": head,
        "n_frames": int(T),
        "frame_duration_ms": round(frame_ms, 3),
        "avg_latency_ms": round(float(lat.mean() * 1000), 4),
        "p50_latency_ms": round(float(np.percentile(lat, 50) * 1000), 4),
        "p99_latency_ms": round(float(np.percentile(lat, 99) * 1000), 4),
        "real_time_factor": round(float(lat.mean() * 1000 / frame_ms), 4),
        "onnxruntime": ort.__version__,
    }
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.output_wav}")

    if not args.no_provenance:
        prov = {
            "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall_start)),
            "wall_seconds": round(time.time() - wall_start, 3),
            "processed_on": collect_provenance(),
            "runtime": {
                "onnxruntime_version": ort.__version__,
                "onnxruntime_get_device": ort.get_device(),
                "session_providers": sess.get_providers(),
                "available_providers": ort.get_available_providers(),
                "intra_op_threads": args.threads or "default",
            },
            "model": {
                "onnx": os.path.abspath(args.onnx),
                "onnx_sha256": _sha256(args.onnx),
                "sidecar_sha256": _sha256(meta_path),
                "head": head, "df_order": K, "df_lookahead": lookahead,
            },
            "input": {
                "path": os.path.abspath(args.input_wav),
                "sha256": _sha256(args.input_wav),
                "sample_rate": int(sr), "samples": int(len(noisy)),
                "seconds": round(len(noisy) / sr, 3),
            },
            "output": {
                "path": os.path.abspath(args.output_wav),
                "sha256": _sha256(args.output_wav),
                "samples": int(len(enh)),
            },
            "perf": stats,
            "note": ("Every model-inference op ran via onnxruntime "
                     f"{sess.get_providers()} on the machine described in 'processed_on'. "
                     "Recompute output.sha256 on the received file to confirm these bytes "
                     "were produced there."),
        }
        prov_path = args.output_wav + ".provenance.json"
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
        p_on = prov["processed_on"]
        print(f"wrote {prov_path}")
        print(f"  processed on: {p_on['device_tree_model'] or p_on['os']} "
              f"| {p_on['hostname']} | {p_on['arch']} | {p_on['cpu_model']} x{p_on['cpu_count']}")
        print(f"  providers   : {sess.get_providers()}")
        print(f"  output sha256: {prov['output']['sha256']}")


if __name__ == "__main__":
    main()
