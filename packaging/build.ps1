<#
.SYNOPSIS
    Build the aidetect standalone executable(s) with PyInstaller.

.DESCRIPTION
    Full (default): bundles torch + transformers, so Binoculars works zero-shot
                    with no training data and no model file. One folder, several
                    GB -> dist\aidetect-full\aidetect-full.exe

    Lite (-Lite):   CPU only, one file, ~67 MB -> dist\aidetect-lite.exe
                    Ships no trained classifier, so it cannot score anything
                    until given a --model produced by FeatureDetector.save().

    Both (-Both):   Builds lite first, then full. The two flavours have
                    different names, so they sit in dist\ side by side.

    The build picks up whichever torch is installed. Against a CPU-only torch
    the packaged app runs Binoculars on CPU: roughly 28GB of RAM for the Falcon
    pair, and impractically slow. Install a CUDA torch first if the target
    machine has a GPU.

    -Python selects the interpreter to build with, so the build can target a
    specific conda env rather than whatever `python` resolves to on PATH.

.EXAMPLE
    .\packaging\build.ps1
    .\packaging\build.ps1 -Lite
    .\packaging\build.ps1 -Both
    .\packaging\build.ps1 -Both -Python "$env:USERPROFILE\.conda\envs\aidetect\python.exe"
#>
[CmdletBinding()]
param(
    [switch]$Lite,
    [switch]$Both,
    [switch]$SkipInstall,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo

function Invoke-Build {
    param([bool]$IsLite)

    if ($IsLite) {
        Write-Host '==> Building LITE executable (CPU only, needs a trained --model)' -ForegroundColor Cyan
        $env:AIDETECT_LITE = '1'
    } else {
        Write-Host '==> Building FULL executable (torch + transformers, several GB)' -ForegroundColor Cyan
        $env:AIDETECT_LITE = '0'
        & $Python -c "import torch; print('    torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
    }

    & $Python -m PyInstaller --clean --noconfirm aidetect.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

    $exe = if ($IsLite) { 'dist\aidetect-lite.exe' } else { 'dist\aidetect-full\aidetect-full.exe' }
    if (-not (Test-Path $exe)) { throw "Expected $exe but it was not produced" }

    $root = if ($IsLite) { $exe } else { 'dist\aidetect-full' }
    $sizeMb = [math]::Round(((Get-ChildItem $root -Recurse -File | Measure-Object -Property Length -Sum).Sum) / 1MB, 1)
    Write-Host "==> Built $exe ($sizeMb MB total)" -ForegroundColor Green

    Write-Host '==> Smoke test: --help' -ForegroundColor Cyan
    & $exe --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed (exit $LASTEXITCODE)" }
    Write-Host '==> OK' -ForegroundColor Green
}

try {
    # -Both needs the GPU extras too, since one of the two builds is the full one.
    $needsGpu = -not $Lite -or $Both

    if (-not $SkipInstall) {
        Write-Host '==> Installing build dependencies' -ForegroundColor Cyan
        if ($needsGpu) {
            & $Python -m pip install -e '.[gpu,build]'
        } else {
            & $Python -m pip install -e '.[build]'
        }
        if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
    }

    if ($Both) {
        # Lite first: it is the quick one, so a broken spec surfaces in a minute
        # rather than after the multi-GB full build.
        Invoke-Build -IsLite $true
        Invoke-Build -IsLite $false
    } else {
        Invoke-Build -IsLite ([bool]$Lite)
    }
}
finally {
    Remove-Item Env:\AIDETECT_LITE -ErrorAction SilentlyContinue
    Pop-Location
}
