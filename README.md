# Clarion — Offline Noise Cancellation Demo

A standalone, fully offline demo of **Clarion**, an AI adaptive noise-cancellation
model for defence comms (SIH problem statement 26052). Enhances noisy speech
(gunfire, engines, crowd/babble, wind, radio static, etc.) in real time on an
ordinary laptop CPU — no GPU, no internet connection, no cloud API.

This repo intentionally ships only what's needed to run the demo: the exported
ONNX model and the local demo app. The full training pipeline (data prep, model
experiments, Kria/FPGA deployment) lives in a separate, private repository.

## Run it (no Python required)

Download the latest `clarion-offline-demo.exe` from the
**[Releases](../../releases)** page and double-click it. A native window opens;
drop in a WAV/MP3/M4A file or try one of the bundled samples. Everything runs on
`127.0.0.1` — you can disconnect from Wi-Fi/Ethernet first to confirm it still
works.

First launch takes a few seconds longer (the exe unpacks itself to a temp
folder); subsequent runs are fast.

## Run it from source

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python webdemo/desktop_app.py
```

Or run it as an ordinary web page instead of a native window:

```bash
python webdemo/server_local.py     # http://localhost:8010
```

## What's under the hood

- **Model:** causal CRN backbone + order-5 deep-filter head, 16 kHz, 32 ms
  window / 16 ms hop, 0 ms look-ahead (strictly causal — no future audio used).
- **Inference:** ONNX Runtime, `CPUExecutionProvider` only.
- **Everything shown in the UI is measured live** on your machine — CPU model,
  execution provider, per-clip latency/RTF, and a SHA-256 provenance record for
  the output file. Nothing is faked or pre-baked.

## Building the .exe yourself

```powershell
pip install -r requirements.txt pyinstaller
.\build_exe.ps1
```

See `build_exe.ps1` for why specific dependencies are excluded/included.
