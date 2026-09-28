"""ANC web demo -- COMPUTER (CPU only) variant.

Same page as the Kria demo, but every clip is denoised on THIS machine's CPU via
onnxruntime (CPUExecutionProvider). No Kria, no SSH, no GPU. Use it to show the
identical Run-D model on a normal desktop CPU next to the edge board.

    python server_local.py            # -> http://0.0.0.0:8010
    PORT=9000 python server_local.py

The "Where it ran" card on the page is filled from a provenance record this process
builds directly (via src.streaming_denoiser's in-process model) -- on this machine it
reports this host / CPU / arch, and session_providers == ["CPUExecutionProvider"].

STREAMING / LIVE ENDPOINTS (see plan: real-time web ANC)
  POST /api/upload            -- decode-only, returns {uid}; pairs with the WS below
  WS   /ws/stream_result/{uid} -- progressive fragment delivery for an uploaded/recorded
                                   clip, driven by the SAME src.streaming_denoiser core
                                   as /api/denoise (just chunked for delivery, not
                                   reprocessed differently). ?pace=realtime paces
                                   fragments at real playback speed (used by the
                                   fallback-demo mode so it still visibly streams).
  WS   /ws/live                -- continuous live-mic streaming. Expects 16 kHz mono
                                   float32 PCM binary frames; asserts that contract on
                                   the first frame per connection rather than silently
                                   accepting something else (see src.streaming_denoiser
                                   AudioFormatError). Sends back enhanced PCM binary
                                   frames plus periodic JSON telemetry.
Both WS paths reuse one shared, already-loaded StreamingDenoiser -- each connection
gets its own lightweight per-stream state (GRU hidden state, ring buffers), not a
second model load.
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import socket
import sys
import time
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

if getattr(sys, "frozen", False):
    # PyInstaller onefile build: everything was bundled flat under sys._MEIPASS
    # (see build_exe.ps1), not under the source tree's webdemo/../ layout.
    HERE = Path(sys._MEIPASS) / "webdemo"  # type: ignore[attr-defined]
    ANC = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    HERE = Path(__file__).resolve().parent
    ANC = HERE.parent
WORK = HERE / "work"
STATIC = HERE / "static"
SAMPLES = STATIC / "samples"
WORK.mkdir(exist_ok=True)

sys.path.insert(0, str(ANC))
sys.path.insert(0, str(ANC / "scripts"))
from src.streaming_denoiser import AudioFormatError, StreamingDenoiser, ensure_pcm_contract  # noqa: E402
from onnx_denoise import _sha256, collect_provenance  # noqa: E402 - reuse, don't duplicate

MODEL = Path(os.environ.get("ANC_ONNX", ANC / "checkpoints" / "model.onnx"))
PORT = int(os.environ.get("PORT", "8010"))
# 0 = no duration cap. Upload processing is genuinely chunked/streamed end to end
# (decode -> inference -> delivery all pipelined per-fragment in /ws/stream_result,
# see its docstring) rather than buffering the whole clip, so there's no memory/latency
# reason to cap clip length here anymore. Set MAX_SECONDS to a positive value to
# reintroduce a cap (e.g. to bound how long a single demo job can hold the job lock).
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "0"))
THREADS = os.environ.get("ANC_THREADS", "")  # "" = onnxruntime default
LIVE_MAX_SECONDS = float(os.environ.get("LIVE_MAX_SECONDS", "600"))  # per-connection cap
FRAGMENT_SECONDS = float(os.environ.get("FRAGMENT_SECONDS", "0.5"))

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
SAMPLE_LABELS = {
    "military_radio": "Military radio",
    "parade": "Republic Day parade (1:00-1:20)",
    "parade2": "Republic Day parade (1:30-1:50)",
}

app = FastAPI(title="ANC demo -- computer CPU only")
_job_lock = asyncio.Lock()   # still serializes the legacy one-shot /api/denoise path
_waiting = 0

DENOISER: StreamingDenoiser | None = None
if MODEL.exists():
    so_threads = int(THREADS) if THREADS else 0
    DENOISER = StreamingDenoiser(str(MODEL), clip_seconds=3.0,
                                  providers=["CPUExecutionProvider"])
    if so_threads:
        print(f"note: ANC_THREADS={so_threads} is ignored for the in-process denoiser "
              f"(onnxruntime session threading is fixed at load time); restart with "
              f"the env var set before first use if you need to change it.")
    # Warm up onnxruntime's first-call JIT/thread-pool costs HERE, at process start,
    # not on the first real user's request -- measured ~1.4s on a cold process vs.
    # ~0.1s once warm, entirely a one-time cost unrelated to clip length or the
    # streaming pipeline itself.
    _warmup_state = DENOISER.new_stream()
    DENOISER.push_samples(_warmup_state, np.zeros(DENOISER.hop * 4, dtype=np.float32))
    DENOISER.flush(_warmup_state)
    del _warmup_state


async def run(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    p = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        p.kill()
        raise HTTPException(504, f"step timed out after {timeout:.0f}s")
    return p.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def local_ips() -> list[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254")):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def cpu_label() -> str:
    if platform.system() == "Windows":
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or "CPU")
    return platform.processor() or platform.machine()


def spectrogram_magnitude_db(wav_path: Path) -> tuple[np.ndarray, int, float]:
    """Shared STFT-magnitude-in-dB computation, split out of spectrogram_png() so the
    caller can compute a common vmax across a before/after PAIR before rendering
    either image -- otherwise each PNG independently normalizes to its own 99.5th
    percentile and the two color scales aren't comparable (a quieter enhanced clip
    would look artificially "boosted" back up to the same brightness as the noisy
    one)."""
    x, sr = sf.read(str(wav_path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    n_fft, hop = 512, 128
    win = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    nfr = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
    S = np.fft.rfft(x[idx] * win, axis=1).T
    mag = 20 * np.log10(np.abs(S) + 1e-6)
    return mag, sr, len(x)


def spectrogram_png(wav_path: Path, out_png: Path, title: str, vmax: float | None = None) -> None:
    """vmax: shared color-scale ceiling (dB) so a before/after pair is rendered on the
    SAME scale and visually comparable (background-energy reduction actually reads as
    darker, not just renormalized back to the same brightness). None = self-normalize
    (backward-compatible standalone use)."""
    mag, sr, n_samples = spectrogram_magnitude_db(wav_path)
    if vmax is None:
        vmax = float(np.percentile(mag, 99.5))
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=130)
    ax.imshow(mag, origin="lower", aspect="auto", cmap="magma",
              extent=[0, n_samples / sr, 0, sr / 2000], vmin=vmax - 75, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("kHz", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(out_png), facecolor="#12141a")
    plt.close(fig)


def sweep_work(keep_minutes: float = 45) -> None:
    cutoff = time.time() - keep_minutes * 60
    for f in WORK.iterdir():
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _legacy_stats(raw: dict) -> dict:
    """Map src.streaming_denoiser's telemetry dict onto the key names the existing
    frontend (static/app.js render()) already reads, so nothing else has to change."""
    return {
        "head": DENOISER.head,
        "n_frames": raw["hops_processed"],
        "frame_duration_ms": raw["frame_duration_ms"],
        "avg_latency_ms": raw["avg_model_ms"],
        "p50_latency_ms": raw["p50_model_ms"],
        "p99_latency_ms": raw["p99_model_ms"],
        "real_time_factor": raw["real_time_factor"],
        "onnxruntime": ort.__version__,
        "gru_resets": raw["gru_resets"],
        "dropped_chunks": raw["dropped_chunks"],
    }


def _build_provenance(onnx_path: Path, in_wav: Path, out_wav: Path,
                       wall_start: float, stats: dict) -> dict:
    """Same shape onnx_denoise.py used to write as a sidecar file, built directly
    in-process now instead of shelled out to and read back."""
    return {
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall_start)),
        "wall_seconds": round(time.time() - wall_start, 3),
        "processed_on": collect_provenance(),
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "onnxruntime_get_device": ort.get_device(),
            "session_providers": DENOISER.session.get_providers(),
            "available_providers": ort.get_available_providers(),
        },
        "model": {
            "onnx": str(onnx_path.resolve()),
            "onnx_sha256": _sha256(str(onnx_path)),
            "head": DENOISER.head, "df_order": DENOISER.K, "df_lookahead": DENOISER.lookahead,
        },
        "input": {
            "path": str(in_wav.resolve()),
            "sha256": _sha256(str(in_wav)),
        },
        "output": {
            "path": str(out_wav.resolve()),
            "sha256": _sha256(str(out_wav)) if out_wav.exists() else None,
        },
        "perf": stats,
        "note": (f"Every model-inference op ran via onnxruntime "
                 f"{DENOISER.session.get_providers()} on the machine described in "
                 f"'processed_on'. Recompute output.sha256 on the received file to "
                 f"confirm these bytes were produced there."),
    }


async def decode_upload(src: Path, uid: str) -> Path:
    """ffmpeg-decode an arbitrary upload/sample to 16 kHz mono s16 wav, blocking until
    the WHOLE file is decoded. Only used by the legacy one-shot /api/denoise path --
    the streaming path below (_ffmpeg_pcm_pipe) decodes incrementally instead, so it
    doesn't make the user wait for a full-file decode before anything happens."""
    in_wav = WORK / f"{uid}_in.wav"
    if FFPROBE and MAX_SECONDS:
        rc, out, _ = await run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(src)], timeout=20)
        try:
            if float(out.strip()) > MAX_SECONDS + 0.5:
                raise HTTPException(413, f"clip longer than {MAX_SECONDS:.0f}s")
        except ValueError:
            pass
    if not FFMPEG:
        raise HTTPException(500, "ffmpeg not found on the server")
    rc, _, err = await run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(in_wav)],
                           timeout=max(60.0, MAX_SECONDS) if MAX_SECONDS else 3600.0)
    if rc != 0 or not in_wav.exists():
        raise HTTPException(400, f"could not decode audio: {err[:300]}")
    return in_wav


async def probe_duration(src: Path) -> float | None:
    """Fast, metadata-only duration read (container header, not a decode pass) -- used
    for the MAX_SECONDS guard and progress-percentage estimates before the actual
    incremental decode starts. None if ffprobe is unavailable or can't tell."""
    if not FFPROBE:
        return None
    rc, out, _ = await run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(src)], timeout=10)
    try:
        return float(out.strip())
    except ValueError:
        return None


async def ffmpeg_pcm_pipe(src: Path, sr: int, read_bytes: int = 8192):
    """Decode `src` to mono float32 PCM at `sr` Hz and yield it AS FFMPEG PRODUCES IT --
    no intermediate wav file, no waiting for the whole clip to decode first. This is
    what makes "instant after upload" literally true rather than just fast: the first
    yielded chunk arrives after ffmpeg's startup latency (tens of ms) plus one
    read_bytes worth of decode, not after the entire file has been processed.
    read_bytes=8192 = 2048 float32 samples = 128ms @16kHz, a reasonable balance between
    first-chunk latency and per-read overhead."""
    if not FFMPEG:
        raise HTTPException(500, "ffmpeg not found on the server")
    proc = await asyncio.create_subprocess_exec(
        FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    leftover = b""
    try:
        while True:
            chunk = await proc.stdout.read(read_bytes)
            if not chunk:
                break
            data = leftover + chunk
            usable_len = len(data) - (len(data) % 4)  # keep 4-byte float32 alignment
            if usable_len <= 0:
                leftover = data
                continue
            yield data[:usable_len]
            leftover = data[usable_len:]
        rc = await proc.wait()
        if rc != 0:
            err = (await proc.stderr.read()).decode(errors="replace")
            raise HTTPException(400, f"could not decode audio: {err[:300]}")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


def _run_whole_file_sync(in_wav: Path):
    """Blocking; run via run_in_threadpool. Whole-clip push+flush through the SAME
    streaming core the live/fragment paths use -- one implementation, not two."""
    noisy, sr = sf.read(str(in_wav), dtype="float32")
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=1)
    assert sr == DENOISER.sample_rate, f"{in_wav} is {sr} Hz, model expects {DENOISER.sample_rate} Hz"
    state = DENOISER.new_stream()
    out = DENOISER.push_samples(state, ensure_pcm_contract(noisy))
    tail = DENOISER.flush(state)
    enh = np.concatenate([out, tail])
    return enh, sr, DENOISER.get_stats(state)


def _finalize_result(uid: str, in_wav: Path, enh: np.ndarray, sr: int, stats_raw: dict,
                      wall_start: float, queue_waited: float) -> dict:
    """Write out_wav + spectrograms + provenance, build the same response shape
    /api/denoise has always returned. Shared by the sync and streaming paths so the
    "final state" looks identical either way."""
    out_wav = WORK / f"{uid}_out.wav"
    sf.write(str(out_wav), enh, sr)

    stats = _legacy_stats(stats_raw)
    prov = _build_provenance(MODEL, in_wav, out_wav, wall_start, stats)

    in_png = WORK / f"{uid}_in.png"
    out_png = WORK / f"{uid}_out.png"
    try:
        # Shared color scale so the before/after pair is actually comparable -- the
        # noisy input sets the ceiling (it's the louder/broader-band signal), and the
        # enhanced side is rendered on that SAME scale, so suppressed background
        # energy reads as visibly darker instead of being renormalized back to the
        # same average brightness.
        in_mag, _, _ = spectrogram_magnitude_db(in_wav)
        shared_vmax = float(np.percentile(in_mag, 99.5))
        spectrogram_png(in_wav, in_png, "Noisy", vmax=shared_vmax)
        spectrogram_png(out_wav, out_png, "Enhanced", vmax=shared_vmax)
    except Exception as e:  # noqa: BLE001
        print("spectrogram failed:", e)

    sweep_work()
    return {
        "id": uid,
        "on_kria": False,
        "mode": "pc-cpu",
        "queue_waited_s": queue_waited,
        "input_audio": f"/files/{in_wav.name}",
        "output_audio": f"/files/{out_wav.name}",
        "input_spec": f"/files/{in_png.name}" if in_png.exists() else None,
        "output_spec": f"/files/{out_png.name}" if out_png.exists() else None,
        "stats": stats,
        "provenance": prov,
    }


async def denoise(src: Path, uid: str) -> dict:
    global _waiting
    if DENOISER is None:
        raise HTTPException(500, f"model not found: {MODEL}")
    wall_start = time.time()
    in_wav = await decode_upload(src, uid)

    waited0 = time.time()
    _waiting += 1
    try:
        async with _job_lock:
            queue_waited = round(time.time() - waited0, 2)
            enh, sr, stats_raw = await run_in_threadpool(_run_whole_file_sync, in_wav)
    finally:
        _waiting -= 1

    return _finalize_result(uid, in_wav, enh, sr, stats_raw, wall_start, queue_waited)


_CACHE_BUST_ASSETS = ("app.js", "styles.css", "streaming_player.js", "audio-worklet-capture.js")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Appends ?v=<mtime> to each static asset reference so an edited app.js/styles.css
    is never served stale from the browser's cache -- the URL itself changes the moment
    the file's mtime does, no manual hard-refresh (or cache-control tuning) required."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in _CACHE_BUST_ASSETS:
        fpath = STATIC / name
        if fpath.exists():
            v = int(fpath.stat().st_mtime)
            html = html.replace(f'/static/{name}"', f'/static/{name}?v={v}"')
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/health")
async def health():
    samples = [{"key": k, "label": v} for k, v in SAMPLE_LABELS.items()
               if (SAMPLES / f"{k}.wav").exists()]
    return {
        "mode": "pc-cpu",
        "kria_online": False,
        "local_fallback": True,
        "host": platform.node(),
        "cpu": cpu_label(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "model_present": MODEL.exists(),
        "max_seconds": MAX_SECONDS,
        "waiting": _waiting,
        "samples": samples,
        "streaming": {
            "live_available": DENOISER is not None,
            "live_max_seconds": LIVE_MAX_SECONDS,
            "fragment_seconds": FRAGMENT_SECONDS,
            "sample_rate": DENOISER.sample_rate if DENOISER else None,
        },
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "execution_provider": (DENOISER.session.get_providers()[0] if DENOISER else None),
        },
    }


@app.post("/api/denoise")
async def api_denoise(file: UploadFile | None = File(default=None),
                      sample: str | None = Query(default=None)):
    uid = uuid.uuid4().hex[:12]
    if sample:
        sp = SAMPLES / f"{sample}.wav"
        if not sp.exists():
            raise HTTPException(404, "unknown sample")
        src = WORK / f"{uid}_src.wav"
        src.write_bytes(sp.read_bytes())
    elif file is not None:
        data = await file.read()
        ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
        src = WORK / f"{uid}_src{ext}"
        src.write_bytes(data)
    else:
        raise HTTPException(400, "no file and no sample")
    try:
        return JSONResponse(await denoise(src, uid))
    finally:
        try:
            src.unlink()
        except OSError:
            pass


@app.post("/api/upload")
async def api_upload(file: UploadFile | None = File(default=None),
                     sample: str | None = Query(default=None)):
    """Save the raw upload ONLY -- no decode here. Decoding now happens incrementally
    inside /ws/stream_result (ffmpeg_pcm_pipe), so this returns as soon as the bytes
    are on disk instead of after a full-file ffmpeg pass. That's what makes "instant
    after upload" true: nothing here blocks on the clip's length."""
    if DENOISER is None:
        raise HTTPException(500, f"model not found: {MODEL}")
    uid = uuid.uuid4().hex[:12]
    if sample:
        sp = SAMPLES / f"{sample}.wav"
        if not sp.exists():
            raise HTTPException(404, "unknown sample")
        src = WORK / f"{uid}_src.wav"
        src.write_bytes(sp.read_bytes())
    elif file is not None:
        data = await file.read()
        ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
        src = WORK / f"{uid}_src{ext}"
        src.write_bytes(data)
    else:
        raise HTTPException(400, "no file and no sample")
    return {"uid": uid}


@app.websocket("/ws/stream_result/{uid}")
async def ws_stream_result(websocket: WebSocket, uid: str, pace: str = "fast"):
    """Progressive fragment delivery for a previously-uploaded (/api/upload) clip --
    decode, inference and delivery are ALL pipelined here: each chunk ffmpeg decodes is
    pushed through the model and sent to the client immediately, so the first audio
    arrives after ffmpeg's startup latency (tens of ms) + one chunk's decode + one hop's
    inference, not after the entire file has been read. ?pace=realtime paces delivery
    to real playback speed instead of "as fast as computed", for the fallback-demo mode
    where the point is to visibly resemble a live mic session."""
    await websocket.accept()
    if DENOISER is None:
        await websocket.close(code=1011, reason="model not loaded")
        return
    matches = list(WORK.glob(f"{uid}_src.*"))
    if not matches:
        await websocket.close(code=1008, reason="unknown uid -- call /api/upload first")
        return
    src = matches[0]

    wall_start = time.time()
    sr = DENOISER.sample_rate
    try:
        duration = await probe_duration(src)
        if MAX_SECONDS and duration is not None and duration > MAX_SECONDS + 0.5:
            await websocket.send_json({"error": f"clip longer than {MAX_SECONDS:.0f}s"})
            return
        total_samples_est = int(duration * sr) if duration else None

        state = DENOISER.new_stream()
        seq = 0
        samples_in = 0
        input_chunks = []    # raw decoded input, accumulated to write {uid}_in.wav at the end
        produced = []        # every fragment's enhanced samples -- the SAME bytes already
                              # streamed to the client, concatenated for out_wav/spectrograms
        async for raw_chunk in ffmpeg_pcm_pipe(src, sr):
            piece = np.frombuffer(raw_chunk, dtype=np.float32)
            samples_in += len(piece)
            if MAX_SECONDS and samples_in > MAX_SECONDS * sr:
                await websocket.send_json({"error": f"clip longer than {MAX_SECONDS:.0f}s"})
                return
            input_chunks.append(piece)

            t0 = time.perf_counter()
            out = await run_in_threadpool(DENOISER.push_samples, state, ensure_pcm_contract(piece))
            proc_s = time.perf_counter() - t0
            seq += 1
            pct = (round(min(100.0, samples_in / total_samples_est * 100), 1)
                   if total_samples_est else None)
            await websocket.send_json({
                "seq": seq, "pct": pct, "n_samples": int(len(out)),
                "processing_ms": round(proc_s * 1000, 2),
            })
            if len(out):
                produced.append(out)
                await websocket.send_bytes(out.tobytes())
            if pace == "realtime":
                frag_s = len(piece) / sr
                sleep_for = frag_s - proc_s
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        tail = await run_in_threadpool(DENOISER.flush, state)
        if len(tail):
            produced.append(tail)
        enh = np.concatenate(produced) if produced else np.zeros(0, dtype=np.float32)
        noisy = np.concatenate(input_chunks) if input_chunks else np.zeros(0, dtype=np.float32)
        in_wav = WORK / f"{uid}_in.wav"
        await run_in_threadpool(sf.write, str(in_wav), noisy, sr)

        stats_raw = DENOISER.get_stats(state)
        result = await run_in_threadpool(
            _finalize_result, uid, in_wav, enh, sr, stats_raw, wall_start, 0.0)
        await websocket.send_json({"done": True, "result": result})
    except WebSocketDisconnect:
        pass
    except AudioFormatError as e:
        await websocket.send_json({"error": str(e)})
    finally:
        try:
            src.unlink()
        except OSError:
            pass
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Continuous live-mic streaming. First binary frame's size must be a multiple of
    4 bytes (float32); format is asserted, not assumed (AudioFormatError closes the
    connection with a clear reason rather than feeding garbage into the model)."""
    await websocket.accept()
    if DENOISER is None:
        await websocket.close(code=1011, reason="model not loaded")
        return

    state = DENOISER.new_stream()
    start = time.time()
    last_telemetry = start
    try:
        while True:
            if time.time() - start > LIVE_MAX_SECONDS:
                await websocket.send_json({"error": f"live session capped at {LIVE_MAX_SECONDS:.0f}s"})
                break
            msg = await websocket.receive_bytes()
            samples = np.frombuffer(msg, dtype=np.float32)
            try:
                out = await run_in_threadpool(DENOISER.push_samples, state, samples)
            except AudioFormatError as e:
                await websocket.send_json({"error": str(e)})
                break
            if len(out):
                await websocket.send_bytes(out.tobytes())
            now = time.time()
            if now - last_telemetry >= 1.0:
                await websocket.send_json({"telemetry": DENOISER.get_stats(state)})
                last_telemetry = now
    except WebSocketDisconnect:
        pass
    finally:
        try:
            DENOISER.flush(state)
        except Exception:  # noqa: BLE001
            pass


@app.get("/files/{name}")
async def files(name: str):
    f = (WORK / name).resolve()
    if f.parent != WORK or not f.exists():
        raise HTTPException(404, "not found")
    mime = {"wav": "audio/wav", "png": "image/png", "mp3": "audio/mpeg"}.get(
        name.rsplit(".", 1)[-1], "application/octet-stream")
    return FileResponse(str(f), media_type=mime)


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    urls = [f"http://{ip}:{PORT}" for ip in local_ips()] or [f"http://127.0.0.1:{PORT}"]
    print("\n" + "=" * 60)
    print(" ANC demo -- computer CPU only")
    print(f" model : {MODEL}")
    print(f" CPU   : {cpu_label()}  ({os.cpu_count()} logical)  arch={platform.machine()}")
    for u in urls:
        print("   ", u)
    print("=" * 60 + "\n")
    uvicorn.run("server_local:app", host="0.0.0.0", port=PORT, log_level="info")
