$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$VenvDir = 'psdenv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'

function Invoke-Checked {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $($args -join ' ')" }
}

# 1) Create the virtual environment (PyTorch needs Python 3.9 - 3.12).
if (-not (Test-Path $VenvPy)) {
    Write-Host 'Creating virtual environment...'
    $base = $null
    foreach ($v in '3.11', '3.12', '3.10', '3.9') {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py "-$v" -c 'import sys' 2>$null
            if ($LASTEXITCODE -eq 0) { $base = @('py', "-$v"); break }
        }
    }
    if (-not $base -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $base = @('python') }
    }
    if (-not $base) { throw 'Python 3.9 - 3.12 is required. Install it from https://www.python.org/downloads/' }
    Invoke-Checked @base -m venv $VenvDir
}

# 2) Install requirements when requirements.txt changed since the last install.
$stamp = Join-Path $VenvDir '.requirements'
$needInstall = -not (Test-Path $stamp) -or
    (Get-FileHash requirements.txt).Hash -ne (Get-FileHash $stamp).Hash
if ($needInstall) {
    Write-Host 'Installing requirements...'
    Invoke-Checked $VenvPy -m pip install --upgrade pip
    Invoke-Checked $VenvPy -m pip install -r requirements.txt
    Copy-Item requirements.txt $stamp -Force
}

# 3) Download any missing models.
Invoke-Checked $VenvPy main.py --download-models

# 4) Run. With no arguments this opens the web UI.
& $VenvPy main.py @args
exit $LASTEXITCODE
