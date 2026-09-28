# Builds the standalone Windows .exe (onefile, windowed, no console) shipped on the
# Releases page. Run from the repo root with the venv from `pip install -r
# requirements.txt pyinstaller` active.
#
#   .\build_exe.ps1
#
# Notes:
# - torch/scipy/sympy/tensorflow are explicitly excluded: nothing in this app imports
#   them, but a bare `--collect-all onnxruntime` pulls torch in transitively through
#   onnxruntime's optional (unused here) transformers/quantization submodules, which
#   balloons the exe from ~76 MB to ~240 MB for code that never runs.
# - server_local.py and the app's own src/scripts modules are added via --add-data
#   (loose files under sys._MEIPASS at run time), not left to PyInstaller's static
#   import graph -- desktop_app.py's dynamic `sys.path.insert` + string-based
#   `uvicorn.Config("server_local:app", ...)` load them the same way whether frozen
#   or not, so this needs no hidden-import bookkeeping to stay in sync with the code.
$ErrorActionPreference = "Stop"

python -m PyInstaller `
  --onefile `
  --windowed `
  --name clarion-offline-demo `
  --add-data "webdemo/static;webdemo/static" `
  --add-data "webdemo/server_local.py;webdemo" `
  --add-data "src;src" `
  --add-data "scripts;scripts" `
  --add-data "checkpoints;checkpoints" `
  --collect-all onnxruntime `
  --collect-all uvicorn `
  --collect-all fastapi `
  --collect-all starlette `
  --collect-all webview `
  --collect-all pythonnet `
  --collect-all clr_loader `
  --collect-all soundfile `
  --hidden-import multipart `
  --exclude-module torch `
  --exclude-module tensorflow `
  --exclude-module sympy `
  --exclude-module scipy `
  --exclude-module tensorboard `
  --exclude-module IPython `
  --exclude-module notebook `
  webdemo/desktop_app.py

Write-Host "Built: dist/clarion-offline-demo.exe"
