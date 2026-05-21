param(
    [string]$Exe = ".\experiment.exe",
    [string]$Results = "results.csv",
    [int]$Iters = 5,
    [int[]]$NodeSizes = @(10000, 100000, 500000),
    [int]$EdgeFactor = 5,
    [int[]]$Threads = @(1, 2, 4, 8),
    [int[]]$Procs = @(1, 2, 4, 8)
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Exe)) {
    & make build
}

if (Test-Path $Results) { Remove-Item $Results }

foreach ($N in $NodeSizes) {
    $M = $N * $EdgeFactor
    Write-Host "=== N=$N M=$M sequential baseline ==="
    $env:OMP_NUM_THREADS = "1"
    & mpiexec -n 1 $Exe --nodes $N --edges $M --iters $Iters `
        --no-omp --no-mpi --results $Results

    foreach ($T in $Threads) {
        Write-Host "=== N=$N M=$M OMP threads=$T ==="
        $env:OMP_NUM_THREADS = "$T"
        & mpiexec -n 1 $Exe --nodes $N --edges $M --iters $Iters `
            --no-seq --no-mpi --results $Results
    }

    foreach ($P in $Procs) {
        Write-Host "=== N=$N M=$M MPI procs=$P ==="
        $env:OMP_NUM_THREADS = "1"
        & mpiexec -n $P $Exe --nodes $N --edges $M --iters $Iters `
            --no-seq --no-omp --results $Results
    }
}

Write-Host "Done. Results in $Results"
