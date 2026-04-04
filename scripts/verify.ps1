$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Invoke-RepoCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host ("==> {0}" -f $Name) -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw ("{0} failed with exit code {1}." -f $Name, $LASTEXITCODE)
    }
}

Push-Location $repoRoot
try {
    Invoke-RepoCheck -Name "Pyright" -Command { uv run pyright }
    Invoke-RepoCheck -Name "Ty" -Command { uv run ty check . }
    Invoke-RepoCheck -Name "Ruff" -Command { uv run ruff check . }
    Invoke-RepoCheck -Name "Pytest" -Command { uv run pytest }
    Invoke-RepoCheck -Name "CompileAll" -Command { uv run python -m compileall app main.py }

    Write-Host ""
    Write-Host "All verification checks passed." -ForegroundColor Green
}
finally {
    Pop-Location
}
