param(
    [string]$OutputPath = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "THIRD_PARTY_LICENSES.md")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOutputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
$tempFile = Join-Path $env:TEMP ("third-party-licenses-{0}.md" -f ([guid]::NewGuid().ToString("N")))

try {
    Push-Location $repoRoot

    uv run pip-licenses `
        --format=markdown `
        --with-urls `
        --output-file $tempFile `
        --ignore-packages face-recognition-app
    if ($LASTEXITCODE -ne 0) {
        throw ("pip-licenses failed with exit code {0}." -f $LASTEXITCODE)
    }

    $generatedBody = (Get-Content $tempFile -Raw).Trim()
    $content = @"
# Third-Party Python Licenses

This repository publishes source code only. It does not vendor Python package files, downloaded model files, or SQLite databases.
This file lists the Python packages currently installed in the repository environment and the licenses declared for those packages.

Regenerate this file with `powershell -ExecutionPolicy Bypass -File .\scripts\update-third-party-licenses.ps1`.

$generatedBody
"@

    $normalizedContent = $content.Replace("`r`n", "`n")
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($resolvedOutputPath, $normalizedContent, $utf8NoBom)
    Write-Host ("Wrote {0}" -f $resolvedOutputPath) -ForegroundColor Green
}
finally {
    if (Test-Path $tempFile) {
        Remove-Item $tempFile -Force
    }
    Pop-Location
}
