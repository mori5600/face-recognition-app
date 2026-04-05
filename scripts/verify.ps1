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

function Assert-ThirdPartyLicensesCurrent {
    $temporaryPath = Join-Path $env:TEMP ("third-party-licenses-verify-{0}.md" -f ([guid]::NewGuid().ToString("N")))
    $expectedPath = Join-Path $repoRoot "THIRD_PARTY_LICENSES.md"

    try {
        & (Join-Path $repoRoot "scripts\update-third-party-licenses.ps1") -OutputPath $temporaryPath
        $expectedContent = Get-Content $expectedPath -Raw
        $actualContent = Get-Content $temporaryPath -Raw

        if ($expectedContent -ne $actualContent) {
            throw "THIRD_PARTY_LICENSES.md is out of date. Run .\scripts\update-third-party-licenses.ps1 and commit the updated file."
        }
    }
    finally {
        if (Test-Path $temporaryPath) {
            Remove-Item $temporaryPath -Force
        }
    }
}

Push-Location $repoRoot
try {
    Invoke-RepoCheck -Name "Pyright" -Command { uv run pyright }
    Invoke-RepoCheck -Name "Ty" -Command { uv run ty check . }
    Invoke-RepoCheck -Name "Ruff" -Command { uv run ruff check . }
    Invoke-RepoCheck -Name "Pytest" -Command { uv run pytest }
    Invoke-RepoCheck -Name "CompileAll" -Command { uv run python -m compileall app main.py }
    Invoke-RepoCheck -Name "ThirdPartyLicenses" -Command { Assert-ThirdPartyLicensesCurrent }

    Write-Host ""
    Write-Host "All verification checks passed." -ForegroundColor Green
}
finally {
    Pop-Location
}
