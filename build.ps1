param(
    [string]$Config = "Release",
    [string]$Out = "experiment.exe"
)

$ErrorActionPreference = "Stop"

$vsRoot = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

$mpiInc = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
$mpiLib = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
if (-not (Test-Path $mpiInc)) { throw "MS-MPI include dir not found: $mpiInc" }
if (-not (Test-Path $mpiLib)) { throw "MS-MPI lib dir not found: $mpiLib" }

$opt = if ($Config -eq "Debug") { "/Od /Zi" } else { "/O2" }

$srcs = "experiment.cpp pagerank_common.cpp"
$cmd = @"
call "$vcvars" >nul && cl.exe /nologo /EHsc /std:c++17 /openmp $opt /MD ^
    /I "$mpiInc" $srcs ^
    /Fe:$Out ^
    /link /LIBPATH:"$mpiLib" msmpi.lib
"@

$tmpBat = [System.IO.Path]::GetTempFileName() + ".bat"
Set-Content -Path $tmpBat -Value $cmd -Encoding ASCII
try {
    & cmd.exe /c $tmpBat
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }
    Write-Host "Built $Out"
} finally {
    Remove-Item $tmpBat -ErrorAction SilentlyContinue
}
