"""ANC demo as an OFFLINE desktop app.

Same page and same Run-D ONNX model as ``server_local.py``, but wrapped in a
native OS window (pywebview -> Edge WebView2 on Windows) instead of a browser
tab. The HTTP server binds to ``127.0.0.1`` on a random free port, so it is
reachable only from this machine -- no LAN, no internet, no ngrok, no browser.

    ..\\.venv\\Scripts\\python desktop_app.py     # or: .\\run_desktop.ps1

Everything the app needs (model, scripts, ffmpeg on PATH) is local; unplug the
network and it still works.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller onefile build: see build_exe.ps1 for how these get laid out
    # under sys._MEIPASS at run time.
    HERE = Path(sys._MEIPASS) / "webdemo"  # type: ignore[attr-defined]
else:
    HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# server_local reads these at import time -- set before uvicorn loads the app.
os.environ.setdefault("MAX_SECONDS", "120")

import uvicorn  # noqa: E402
import webview  # noqa: E402

HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


PORT = int(os.environ.get("ANC_DESKTOP_PORT") or 0) or _free_port()
_server: uvicorn.Server | None = None


def _run_server() -> None:
    global _server
    cfg = uvicorn.Config("server_local:app", host=HOST, port=PORT,
                         log_level="warning", access_log=False)
    _server = uvicorn.Server(cfg)
    _server.run()


def _wait_until_up(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    threading.Thread(target=_run_server, daemon=True).start()

    base = f"http://{HOST}:{PORT}"
    if not _wait_until_up(base + "/api/health"):
        print("local server did not start", file=sys.stderr)
        return 1

    window = webview.create_window(
        "AI Adaptive Noise Cancellation",
        url=base,
        width=1180,
        height=900,
        min_size=(820, 640),
    )

    def _shutdown() -> None:
        if _server is not None:
            _server.should_exit = True

    window.events.closed += _shutdown
    webview.start()  # blocks on the main thread until the window is closed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
