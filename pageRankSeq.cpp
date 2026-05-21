#include "pagerank_common.hpp"

#include <cmath>
#include <vector>

using namespace std;

inline void updatePageRank(const pagerank::CsrGraph& graph,
                           vector<double>& pagerank,
                           double damping_factor,
                           int max_iterations,
                           double tolerance) {
    const int N = graph.num_nodes;
    if (N <= 0) return;

    vector<double> new_pagerank(N, 0.0);
    for (int iter = 0; iter < max_iterations; ++iter) {
        double dangling_sum = 0.0;
        for (int node = 0; node < N; ++node) {
            if (graph.out_degree[node] == 0) {
                dangling_sum += pagerank[node];
            }
        }
        const double base = (1.0 - damping_factor) / N
                          + damping_factor * dangling_sum / N;

        for (int node = 0; node < N; ++node) {
            double rank_sum = 0.0;
            for (uint64_t edge_idx = graph.row_idx[node];
                 edge_idx < graph.row_idx[node + 1]; ++edge_idx) {
                const uint64_t src = graph.col_idx[edge_idx];
                const uint64_t od = graph.out_degree[src];
                if (od > 0) {
                    rank_sum += pagerank[src] / static_cast<double>(od);
                }
            }
            new_pagerank[node] = base + damping_factor * rank_sum;
        }

        double diff = 0.0;
        for (int j = 0; j < N; ++j) {
            diff += std::abs(new_pagerank[j] - pagerank[j]);
            pagerank[j] = new_pagerank[j];
        }
        if (diff < tolerance) break;
    }
}
