#!/usr/bin/env bash
set -euo pipefail

EXE=./experiment.exe
RESULTS=results.csv
ITERS=${ITERS:-5}

NODE_SIZES=(${NODE_SIZES:-10000 100000 500000})
EDGE_FACTOR=${EDGE_FACTOR:-5}
THREADS=(${THREADS:-1 2 4 8})
PROCS=(${PROCS:-1 2 4 8})

if [ ! -f "$EXE" ]; then
  make build
fi

rm -f "$RESULTS"

for N in "${NODE_SIZES[@]}"; do
  M=$((N * EDGE_FACTOR))
  echo "=== N=$N M=$M sequential baseline ==="
  OMP_NUM_THREADS=1 mpiexec -n 1 "$EXE" \
    --nodes "$N" --edges "$M" --iters "$ITERS" \
    --no-omp --no-mpi --results "$RESULTS"

  for T in "${THREADS[@]}"; do
    echo "=== N=$N M=$M OMP threads=$T ==="
    OMP_NUM_THREADS=$T mpiexec -n 1 "$EXE" \
      --nodes "$N" --edges "$M" --iters "$ITERS" \
      --no-seq --no-mpi --results "$RESULTS"
  done

  for P in "${PROCS[@]}"; do
    echo "=== N=$N M=$M MPI procs=$P ==="
    OMP_NUM_THREADS=1 mpiexec -n "$P" "$EXE" \
      --nodes "$N" --edges "$M" --iters "$ITERS" \
      --no-seq --no-omp --results "$RESULTS"
  done
done

echo "Done. Results in $RESULTS"
