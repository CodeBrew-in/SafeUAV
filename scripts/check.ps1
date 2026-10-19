# Quality gate: JavaScript syntax check, then the whole test suite (about 3 minutes).
# Usage:  .\scripts\check.ps1          (add -Fast to skip the slow grid-cache and live-model tests)
param([switch]$Fast)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path ".venv\Scripts\python.exe").Path

if (Get-Command node -ErrorAction SilentlyContinue) {
    node --check backend/app/geo/static/app.js
    if ($LASTEXITCODE -ne 0) { throw "app.js has a syntax error" }
    Write-Host "app.js: syntax ok"
} else { Write-Host "node not found: skipping the JavaScript syntax check" }

Set-Location backend
$extra = @()
if ($Fast) { $extra = @("-k", "not grid_cache_is_bounded and not live_model") }
& $py -m pytest -q @extra
if ($LASTEXITCODE -ne 0) { throw "tests failed" }
Write-Host "All checks passed."
