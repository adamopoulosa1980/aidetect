<#
.SYNOPSIS
    Build the aidetect standalone executable with PyInstaller.

.DESCRIPTION
    Full (default): bundles torch + transformers, so Binoculars works zero-shot
                    with no training data and no model file. One folder, several
                    GB -> dist\aidetect\aidetect.exe

    Lite (-Lite):   CPU only, one file, ~67 MB -> dist\aidetect.exe
                    Ships no trained classifier, so it cannot score anything
                    until given a --model produced by FeatureDetector.save().

    The build picks up whichever torch is installed. Against a CPU-only torch
    the packaged app runs Binoculars on CPU: roughly 28GB of RAM for the Falcon
    pair, and impractically slow. Install a CUDA torch first if the target
    machine has a GPU.

.EXAMPLE
    .\packaging\build.ps1
    .\packaging\build.ps1 -Lite
#>
[CmdletBinding()]
param(
    [switch]$Lite,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    if (-not $SkipInstall) {
        Write-Host '==> Installing build dependencies' -ForegroundColor Cyan
        if ($Lite) {
            python -m pip install -e '.[build]'
        } else {
            python -m pip install -e '.[gpu,build]'
        }
        if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
    }

    if ($Lite) {
        Write-Host '==> Building LITE executable (CPU only, needs a trained --model)' -ForegroundColor Cyan
        $env:AIDETECT_LITE = '1'
    } else {
        Write-Host '==> Building FULL executable (torch + transformers, several GB)' -ForegroundColor Cyan
        $env:AIDETECT_LITE = '0'
        python -c "import torch; print('    torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
    }

    python -m PyInstaller --clean --noconfirm aidetect.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

    $exe = if ($Lite) { 'dist\aidetect.exe' } else { 'dist\aidetect\aidetect.exe' }
    if (-not (Test-Path $exe)) { throw "Expected $exe but it was not produced" }

    $root = if ($Lite) { $exe } else { 'dist\aidetect' }
    $sizeMb = [math]::Round(((Get-ChildItem $root -Recurse -File | Measure-Object -Property Length -Sum).Sum) / 1MB, 1)
    Write-Host "==> Built $exe ($sizeMb MB total)" -ForegroundColor Green

    Write-Host '==> Smoke test: --help' -ForegroundColor Cyan
    & $exe --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed (exit $LASTEXITCODE)" }
    Write-Host '==> OK' -ForegroundColor Green
}
finally {
    Remove-Item Env:\AIDETECT_LITE -ErrorAction SilentlyContinue
    Pop-Location
}
