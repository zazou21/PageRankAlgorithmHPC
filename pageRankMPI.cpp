// Initialize:
//  rank[i] = 1 / N
// For iter = 1 → max_iterations:
//  For each node i:
//  new_rank[i] = (1 - d) / N
//  For each node j:
//  For each neighbor i of j:
//  contribution = rank[j] / out_degree[j]
//  new_rank[i] += d * contribution
//  diff = 0
//  For each node i:
//  diff += |new_rank[i] - rank[i]|
//  rank[i] = new_rank[i]
//  If diff < tolerance:
//  Distribute nodes across processes
// Exchange rank using MPI_Allgather
// Compute local updates
// Use MPI_Allreduce for diff

#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "pagerank_common.hpp"

using namespace std;

void buildPartition(int total_nodes, int world_size, vector<int>& counts, vector<int>& offsets) {
	counts.assign(world_size, total_nodes / world_size);
	const int remainder = total_nodes % world_size;
	for (int i = 0; i < remainder; ++i) {
		counts[i]++;
	}

	offsets.assign(world_size, 0);
	for (int i = 1; i < world_size; ++i) {
		offsets[i] = offsets[i - 1] + counts[i - 1];
	}
}



void updatePageRankMPI(const pagerank::CsrGraph& graph,
					   vector<double>& pagerank,
					   double damping_factor,
					   int max_iterations,
					   double tolerance,
					   MPI_Comm comm = MPI_COMM_WORLD               
                    ) {
	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(comm, &world_rank);
	MPI_Comm_size(comm, &world_size);

	const int num_nodes = graph.num_nodes;
	if (num_nodes <= 0) {
		return;
	}

	if ((pagerank.size()) != num_nodes) {
		pagerank.assign(num_nodes, 1.0 / num_nodes);
	}

	vector<int> counts;
	vector<int> offsets;
	buildPartition(num_nodes, world_size, counts, offsets);

	const int local_count = counts[world_rank];
	const int start_node = offsets[world_rank];
	const int end_node = start_node + local_count;

	vector<double> local_new(local_count, 0.0);
	vector<double> gathered_new(num_nodes, 0.0);

	for (int iter = 0; iter < max_iterations; ++iter) {
		for (int node = start_node; node < end_node; ++node) {
			double rank_sum = 0.0;
			for (uint64_t edge_idx = graph.row_idx[node]; edge_idx < graph.row_idx[node + 1]; ++edge_idx) {
				const int neighbor = (graph.col_idx[edge_idx]);
				const uint64_t out_degree = graph.row_idx[neighbor + 1] - graph.row_idx[neighbor];
				if (out_degree > 0) {
					rank_sum += pagerank[neighbor] / (out_degree);
				}
			}

			local_new[node - start_node] = (1.0 - damping_factor) / num_nodes + damping_factor * rank_sum;
		}

		MPI_Allgatherv(local_new.data(),
					   local_count,
					   MPI_DOUBLE,
					   gathered_new.data(),
					   counts.data(),
					   offsets.data(),
					   MPI_DOUBLE,
					   comm);

		double local_diff = 0.0;
		for (int node = start_node; node < end_node; ++node) {
			local_diff += std::abs(gathered_new[node] - pagerank[node]);
		}

		double global_diff = 0.0;
		MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, comm);

		pagerank.swap(gathered_new);

		if (global_diff < tolerance) {
			break;
		}
	}
}


