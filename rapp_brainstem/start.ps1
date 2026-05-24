# start.ps1 — Windows launcher for RAPP Brainstem
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Refresh PATH so newly-installed tools (gh, python) are found
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

# Ensure UTF-8 output from Python
$env:PYTHONUTF8 = "1"

# Check Python is available
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "ERROR: Python not found on PATH. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Install deps if needed
try {
    python -c "import flask, requests, dotenv" 2>$null
} catch {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt -q
}
# Double-check after install
$depCheck = python -c "import flask, requests, dotenv" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt -q
}

# Check gh CLI (optional — the web login flow works without it)
$gh = Get-Command gh -ErrorAction SilentlyContinue
if ($gh) {
    Write-Host "gh CLI found — token will be auto-detected if you're logged in." -ForegroundColor Green
} else {
    Write-Host "gh CLI not found — you can authenticate via the web UI at http://localhost:7071" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting RAPP Brainstem..." -ForegroundColor Cyan
# Launch via the boot wrapper so organs, senses, and the /web mount
# are wired in additively. The wrapper runs the canonical kernel
# verbatim (Constitution Article XXXIII §4 — kernel stays untouched).
# Falls back to legacy boot.py at root, then to the kernel directly,
# for older organism layouts.
if (Test-Path "utils/boot.py") {
    python utils/boot.py
} elseif (Test-Path "boot.py") {
    python boot.py
} else {
    python brainstem.py
}
