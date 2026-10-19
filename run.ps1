# One command to run the SafeUAV console on Windows:  .\run.ps1   (optional: -Port 8001  -NoBrowser)
param([int]$Port = 0, [switch]$NoBrowser)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating the virtual environment (.venv) ..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 or newer is required. Install it from python.org and try again." }
}
& $py -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 or newer is required." }

& $py -c "import pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the pinned packages ..."
    & $py -m pip install --quiet -r requirements.txt
}

# Gemini is optional: set GEMINI_API_KEY in your environment. Without it (and without Ollama) the console
# still works with the basic parser.
$args2 = @("-m", "app.geo.server")
if ($Port -gt 0) { $args2 += $Port }
if (-not $NoBrowser) {
    $url = "http://127.0.0.1:" + ($(if ($Port -gt 0) { $Port } else { 8000 }))
    Start-Job { param($u) Start-Sleep 2; Start-Process $u } -ArgumentList $url | Out-Null
}
Set-Location backend
& (Join-Path $PSScriptRoot $py) @args2
