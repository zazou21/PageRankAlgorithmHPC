#include "pagerank_common.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

using namespace std;

inline void buildPartition(int total_nodes, int world_size,
                           vector<int>& counts, vector<int>& offsets) {
    counts.assign(world_size, total_nodes / world_size);
    const int remainder = total_nodes % world_size;
    for (int i = 0; i < remainder; ++i) counts[i]++;

    offsets.assign(world_size, 0);
    for (int i = 1; i < world_size; ++i) {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }
}

inline void updatePageRankMPI(const pagerank::CsrGraph& graph,
                              vector<double>& pagerank,
                              double damping_factor,
                              int max_iterations,
                              double tolerance,
                              MPI_Comm comm = MPI_COMM_WORLD) {
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    const int N = graph.num_nodes;
    if (N <= 0) return;

    if (static_cast<int>(pagerank.size()) != N) {
        pagerank.assign(N, 1.0 / N);
    }

    vector<int> counts, offsets;
    buildPartition(N, world_size, counts, offsets);

    const int local_count = counts[world_rank];
    const int start_node = offsets[world_rank];
    const int end_node = start_node + local_count;

    vector<double> local_new(local_count, 0.0);
    vector<double> gathered_new(N, 0.0);

    for (int iter = 0; iter < max_iterations; ++iter) {
        double dangling_sum = 0.0;
        for (int node = 0; node < N; ++node) {
            if (graph.out_degree[node] == 0) dangling_sum += pagerank[node];
        }
        const double base = (1.0 - damping_factor) / N
                          + damping_factor * dangling_sum / N;

        for (int node = start_node; node < end_node; ++node) {
            double rank_sum = 0.0;
            for (uint64_t edge_idx = graph.row_idx[node];
                 edge_idx < graph.row_idx[node + 1]; ++edge_idx) {
                const uint64_t src = graph.col_idx[edge_idx];
                const uint64_t od = graph.out_degree[src];
                if (od > 0) rank_sum += pagerank[src] / static_cast<double>(od);
            }
            local_new[node - start_node] = base + damping_factor * rank_sum;
        }

        MPI_Allgatherv(local_new.data(), local_count, MPI_DOUBLE,
                       gathered_new.data(), counts.data(), offsets.data(),
                       MPI_DOUBLE, comm);

        double local_diff = 0.0;
        for (int node = start_node; node < end_node; ++node) {
            local_diff += std::abs(gathered_new[node] - pagerank[node]);
        }

        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, comm);

        pagerank.swap(gathered_new);

        if (global_diff < tolerance) break;
    }
}

inline void broadcastGraph(pagerank::CsrGraph& graph, int root, MPI_Comm comm) {
    int world_rank = 0;
    MPI_Comm_rank(comm, &world_rank);

    int header[2] = {graph.num_nodes, graph.num_edges};
    MPI_Bcast(header, 2, MPI_INT, root, comm);
    if (world_rank != root) {
        graph.num_nodes = header[0];
        graph.num_edges = header[1];
        graph.row_idx.assign(graph.num_nodes + 1, 0);
        graph.col_idx.assign(graph.num_edges, 0);
        graph.out_degree.assign(graph.num_nodes, 0);
    }
    if (graph.num_nodes <= 0) return;

    MPI_Bcast(graph.row_idx.data(),
              static_cast<int>(graph.num_nodes + 1),
              MPI_UINT64_T, root, comm);
    if (graph.num_edges > 0) {
        MPI_Bcast(graph.col_idx.data(),
                  graph.num_edges, MPI_UINT64_T, root, comm);
    }
    MPI_Bcast(graph.out_degree.data(),
              graph.num_nodes, MPI_UINT64_T, root, comm);
}
