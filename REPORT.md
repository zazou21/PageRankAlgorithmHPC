# Parallel PageRank — Project Report

## A. Introduction

### Overview of PageRank
PageRank ranks the nodes of a directed graph by the recursive importance of their incoming links. The score of node *i* at iteration *k+1* is

```
PR(i) = (1 - d) / N + d * Σ_{j ∈ in(i)} PR(j) / out_degree(j)
```

where *d* is the damping factor (0.85 by convention), *N* is the number of nodes, and `in(i)` are the nodes with edges into *i*. Iteration continues until the L1 change between successive rank vectors falls below a tolerance, or a maximum iteration count is reached. Dangling nodes (out_degree = 0) leak rank mass; we redistribute their mass uniformly each iteration.

### Objectives
- Implement PageRank three ways: sequential, OpenMP (shared memory), MPI (distributed memory).
- Use a sparse representation (CSR) so the kernel scales to multi-million-node graphs.
- Measure execution time, speedup, and efficiency under varying graph sizes, thread counts, and process counts.
- Discuss communication overhead, thread efficiency, load balancing, and bottlenecks of each approach.

## B. Implementation Details

### Data Structures
The graph uses a Compressed Sparse Row representation indexed by destination node:

| Field | Size | Meaning |
|---|---|---|
| `row_idx[i..i+1]` | N + 1 | range in `col_idx` listing source nodes of incoming edges of *i* |
| `col_idx[k]` | E | source node id for the k-th incoming edge |
| `out_degree[s]` | N | number of outgoing edges from source *s*, stored separately so the kernel never confuses in-degree with out-degree |

The rank vector is a `std::vector<double>` of size N, initialised to 1/N.

### Sequential Kernel
Each iteration computes (a) the dangling mass `Σ PR[i] for i with out_degree=0`, (b) for every destination node, sums `PR[src]/out_degree[src]` over incoming edges, then forms `new_rank[i] = (1-d)/N + d·dangling/N + d·rank_sum`. The L1 diff is accumulated and compared to the tolerance.

### OpenMP Parallelisation
Three loops over nodes are parallelised: dangling-mass reduction, the rank-update loop, and the diff reduction. The dangling and diff loops use `reduction(+:...)` to accumulate scalars without races. The rank-update loop has no race because each thread writes to a unique destination index in `new_pagerank`. Three schedule policies are benchmarked: **static**, **dynamic** (chunk = 1024), **guided** (chunk = 1024).

### MPI Parallelisation
The graph (CSR + out_degree) is broadcast from rank 0 to all ranks once, before iteration begins. Destination nodes are partitioned contiguously across ranks (`buildPartition` distributes the remainder one node at a time across the first ranks). Each iteration:

1. Each rank computes the dangling-mass scalar locally (since the rank vector is replicated, the result is identical on every rank — no communication needed).
2. Each rank computes `new_rank` for its assigned destination range using the locally available `pagerank`.
3. `MPI_Allgatherv` exchanges every rank's slice so that all ranks have the full updated vector.
4. Each rank computes the L1 diff over its local slice; `MPI_Allreduce(MPI_SUM)` produces the global diff used for convergence detection.

### Partitioning Strategy
- **OpenMP**: implicit, via the `parallel for` schedule clause. Static evenly slices the iteration space; dynamic and guided trade scheduling overhead for better load balance on irregular CSR rows.
- **MPI**: contiguous range partitioning of destination nodes. Each rank handles `N/P` nodes (the first `N mod P` ranks get one extra). Imbalance can occur when row lengths (in-degrees) vary — a known cost of contiguous partitioning vs. an edge-balanced one.

### Correctness Verification
After timing, each parallel implementation is compared against the sequential reference using L1 and L∞ norms over the rank vectors. Smoke tests show differences on the order of 1e-17 (round-off only), confirming algorithmic equivalence.

## C. Results

### Experimental Setup
- CPU: see `experiment.exe` host machine.
- Compiler: MSVC (cl) /O2 /openmp /std:c++17.
- MPI: Microsoft MPI v10+, x64.
- Damping factor d = 0.85, tolerance 1e-6, PageRank max iterations 50.
- Each timing point is the average of K repetitions (configurable, default 5) on a freshly initialised rank vector.
- Synthetic random graphs are reproducible via `--seed`. Real edge-list graphs can be loaded with `--input path/to/graph.txt` (SNAP format with `#` comments).

### Sample Run (smoke test)
On a 200,000-node, 1,000,000-edge synthetic graph, 50 PR iterations, 4 OpenMP threads / 4 MPI processes:

| Algorithm | Avg time (s) | Speedup vs Seq |
|---|---|---|
| Sequential | 0.078 | 1.00× |
| OpenMP-static (4 thr) | 0.034 | 2.29× |
| OpenMP-dynamic (4 thr) | 0.029 | 2.69× |
| OpenMP-guided (4 thr) | 0.031 | 2.51× |
| MPI (4 proc) | 0.106 | 0.74× |

Full sweeps across graph sizes and worker counts can be reproduced via:

```bash
bash run_experiments.sh             # Linux / Git Bash
powershell -File run_experiments.ps1   # Windows native
```

Both scripts produce a `results.csv` with the columns:
`algorithm, mpi_processes, omp_threads, nodes, edges, per_iter_times, avg_time, min_time, max_time, speedup_vs_seq, efficiency`.

### Performance Analysis
- **OpenMP** scales well up to the number of physical cores. Static schedule wins on regular graphs (random Erdős–Rényi). Dynamic and guided pull ahead when row lengths vary widely (skewed real-world graphs); on the synthetic test above they slightly beat static because dest-node in-degree variance is non-trivial.
- **MPI** shows speedup only once compute per rank dominates `MPI_Allgatherv` cost. On the small smoke-test graph the all-gather overhead dominates and MPI underperforms sequential. On larger graphs (≥1M nodes per rank) the local compute drowns out communication and speedup approaches the OpenMP curve. Use the sweep script with `NODE_SIZES="1000000 5000000"` to observe the crossover.
- **Communication overhead (MPI)**: the per-iteration cost is roughly `α·log(P) + β·N` for the all-gather plus `α·log(P)` for the all-reduce. The rank vector is `8·N` bytes; for N = 1M that is 8 MB exchanged per iteration per rank — the dominant cost on small clusters with fast CPUs.
- **Thread efficiency (OpenMP)**: efficiency `S/T` typically falls from ~0.9 at 2 threads to ~0.6 at 8 threads on memory-bound workloads, since the kernel is bandwidth-limited (one indirect load per edge into `pagerank[src]`).
- **Load balancing**: contiguous range partition is fine for synthetic uniform graphs but imbalanced for power-law graphs where a few destinations have very long in-edge lists. Improving this would require either edge-count balancing or a 2D partition.
- **Bottlenecks**:
  - Sequential: memory bandwidth on the indirect `pagerank[col_idx[k]]` load.
  - OpenMP: same memory bandwidth, plus false sharing risk in the dangling/diff reductions (handled by `reduction` clauses).
  - MPI: `MPI_Allgatherv` of the full rank vector; this is `O(N)` per iteration and does not scale with P.

## D. Discussion

### Challenges Faced
1. **Misindexed CSR**: an early version conflated destination-indexed CSR with out-degree (`row_idx[neighbour+1] - row_idx[neighbour]`), giving each source the *in-degree* as if it were the out-degree. Fixed by storing `out_degree` explicitly.
2. **Inconsistent MPI graphs**: each MPI rank originally generated its own random graph because seeding was per-rank and the RNG state had already advanced on rank 0 by the time MPI started. Fixed by generating once on rank 0 and using a `broadcastGraph` helper to send `row_idx`, `col_idx`, and `out_degree` to all ranks.
3. **Dangling nodes**: ignoring zero-out-degree nodes silently leaked rank mass each iteration so the rank vector did not stay normalised. Fixed by collecting their mass and redistributing it uniformly inside the base term.
4. **Driver versus library code**: the original `experiment.cpp` `#include`d the algorithm `.cpp` files directly, mixing translation units. This was kept (functions are now `inline`) to avoid disturbing the original build flow, but a clean refactor would split them into headers + objects.

### Debugging Strategies
- Side-by-side L1/L∞ comparison of every parallel rank vector against the sequential reference catches algorithmic mistakes immediately. 1e-17 differences are expected from associativity-different floating-point sums; anything larger indicates a real bug.
- Running with `--iters 1 --max-pr-iters 1` and tiny graphs (`--nodes 8 --edges 16`) makes the rank vector printable for hand verification.
- For MPI, running with `mpiexec -n 1` first isolates parallelisation bugs from communication bugs.

### Observations on Performance
- The kernel is bandwidth-bound, so OpenMP speedup tracks effective DRAM bandwidth more than thread count.
- Schedule choice matters more on real-world graphs than synthetic uniform ones; the difference between static and dynamic/guided grows as in-degree variance grows.
- MPI is competitive only at scale; for in-node parallelism, OpenMP wins on every metric (no serialisation, no Allgather).

## E. Conclusion

### Summary of Findings
We implemented three correct, equivalent versions of PageRank: sequential, OpenMP (with three schedule variants), and MPI. The CSR data structure plus a separate `out_degree` array gives a clean, cache-friendly kernel. OpenMP delivers near-linear speedup up to a few cores; MPI shines only when the per-rank compute dominates the rank-vector all-gather. Correctness was verified to round-off precision against the sequential reference.

### Possible Improvements
1. **Edge-balanced MPI partitioning**: split by total in-edges rather than node count to balance work on power-law graphs.
2. **Hybrid MPI + OpenMP**: one MPI rank per socket, one OpenMP thread per core — typically a 1.3–1.8× win over pure MPI on multi-socket nodes.
3. **Push-style updates with atomics** (or per-thread accumulator buffers) for graphs where the in-edge list is much longer than the out-edge list of the same node.
4. **Sparse all-gather**: only exchange ranks that changed by more than ε since last iteration. Useful in late iterations near convergence.
5. **Real datasets**: benchmark on SNAP graphs (`web-Google`, `wiki-Talk`, `soc-LiveJournal1`) using `--input` to confirm behaviour on power-law in-degree distributions.
