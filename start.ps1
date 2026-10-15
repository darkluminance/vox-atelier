param([string]$ModelRoot = 'D:\models')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot 'data') | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'dist\index.html'))) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        throw 'The frontend has not been built. Install Node.js, then run npm ci and npm run build in this folder.'
    }
    Write-Host 'Building the frontend for this checkout…'
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'node_modules'))) {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
}
$env:TTS_MODEL_ROOT = $ModelRoot
$qwenPython = Join-Path $PSScriptRoot '.venv-qwen\Scripts\python.exe'
$chatterPython = Join-Path $PSScriptRoot '.venv-chatterbox\Scripts\python.exe'
if (Test-Path -LiteralPath $qwenPython) { $env:TTS_PYTHON_QWEN = $qwenPython }
if (Test-Path -LiteralPath $chatterPython) { $env:TTS_PYTHON_CHATTERBOX = $chatterPython }
if (Test-Path -LiteralPath $chatterPython) { $env:TTS_PYTHON_CSM = $chatterPython }
$serverPython = 'python'
$serverOut = Join-Path $PSScriptRoot 'data\server.log'
$serverErr = Join-Path $PSScriptRoot 'data\server-errors.log'
Write-Host 'Vox Atelier is starting…'
$server = Start-Process -FilePath $serverPython -ArgumentList @('-m','uvicorn','backend.server:app','--host','127.0.0.1','--port','8765') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($server.HasExited) { break }
        try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/models' -TimeoutSec 1; $ready = $true; break } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw "The server did not start. See $serverErr" }
    Write-Host 'Ready at http://127.0.0.1:8765'
    Start-Process 'http://127.0.0.1:8765'
    Read-Host 'Press Enter here to stop the local server'
}
finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id }
}
