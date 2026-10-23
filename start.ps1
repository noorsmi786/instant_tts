# Start the Vapi Custom TTS Server (Windows PowerShell)
# Usage: .\start.ps1
# Optional: .\start.ps1 --host 127.0.0.1 --port 8000

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$VenvDir = Join-Path $ProjectDir "venv311"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
$EnvFile = Join-Path $ProjectDir ".env"
$EnvExample = Join-Path $ProjectDir ".env.example"

Write-Host "`n🎙️  Vapi Custom TTS Server — Chatterbox-Nano" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

# ── 1. Check .env ─────────────────────────────────────────────────────────────
if (-not (Test-Path $EnvFile)) {
    Write-Host "`n[SETUP] .env not found. Creating from .env.example..." -ForegroundColor Yellow
    Copy-Item $EnvExample $EnvFile
    Write-Host "[SETUP] ⚠️  Edit .env and set VAPI_SECRET before continuing." -ForegroundColor Red
    Write-Host "        Open: $EnvFile" -ForegroundColor Yellow
    exit 1
}

# Check that VAPI_SECRET is not the placeholder
$envContent = Get-Content $EnvFile -Raw
if ($envContent -match "REPLACE_WITH_YOUR_STRONG_SECRET") {
    Write-Host "`n❌ ERROR: VAPI_SECRET is still the placeholder value." -ForegroundColor Red
    Write-Host "   Edit .env and set a real secret, then restart." -ForegroundColor Yellow
    exit 1
}

# ── 2. Create venv if missing ─────────────────────────────────────────────────
if (-not (Test-Path $VenvPython)) {
    Write-Host "`n[SETUP] Creating Python 3.11 virtual environment..." -ForegroundColor Yellow
    & py -3.11 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create venv. Is Python 3.11 installed?" -ForegroundColor Red
        Write-Host "   Check: py -0" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[SETUP] ✅ venv311 created." -ForegroundColor Green
}

# ── 3. Install / verify dependencies ─────────────────────────────────────────
$markerFile = Join-Path $VenvDir "deps_installed.marker"
if (-not (Test-Path $markerFile)) {
    Write-Host "`n[SETUP] Installing dependencies (this may take several minutes)..." -ForegroundColor Yellow
    Write-Host "        Chatterbox-Nano model weights will also download on first run." -ForegroundColor DarkGray

    # Upgrade pip first
    & $VenvPython -m pip install --quiet --upgrade pip

    # Install CPU-only PyTorch first to avoid pulling in CUDA binaries
    Write-Host "[SETUP] Installing PyTorch (CPU)..." -ForegroundColor Yellow
    & $VenvPip install --quiet torch==2.3.1 torchaudio==2.3.1 `
        --index-url https://download.pytorch.org/whl/cpu

    # Install remaining deps
    Write-Host "[SETUP] Installing server dependencies..." -ForegroundColor Yellow
    & $VenvPip install --quiet -r (Join-Path $ProjectDir "requirements.txt") `
        --extra-index-url https://download.pytorch.org/whl/cpu

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ pip install failed. Check error output above." -ForegroundColor Red
        exit 1
    }

    # Write marker so we skip reinstall next time
    Set-Content -Path $markerFile -Value (Get-Date).ToString()
    Write-Host "[SETUP] ✅ Dependencies installed." -ForegroundColor Green
}

# ── 4. Parse optional args ────────────────────────────────────────────────────
$host = "127.0.0.1"
$port = "8000"
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--host") { $host = $args[$i+1]; $i++ }
    if ($args[$i] -eq "--port") { $port = $args[$i+1]; $i++ }
}

# ── 5. Start server ───────────────────────────────────────────────────────────
Write-Host "`n🚀 Starting server on http://${host}:${port}" -ForegroundColor Green
Write-Host "   Preview UI:   http://127.0.0.1:${port}/preview/ui" -ForegroundColor Cyan
Write-Host "   Health check: http://127.0.0.1:${port}/health" -ForegroundColor Cyan
Write-Host "   API docs:     http://127.0.0.1:${port}/docs" -ForegroundColor Cyan
Write-Host "   Vapi webhook: POST http://127.0.0.1:${port}/vapi/tts" -ForegroundColor Cyan
Write-Host "`n   (Model warmup takes 30-90s on first load — check logs below)" -ForegroundColor DarkGray
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor DarkGray

Set-Location $ProjectDir
& $VenvPython -m uvicorn app.main:app --host $host --port $port --log-level info
