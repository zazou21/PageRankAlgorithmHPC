// Initialize data
// For iter = 1 → max_iterations:
//  Compute intermediate values
//  Update state
//  Synchronize (if needed)
//  Check convergence (optional)
// End




#include "pagerank_common.hpp"

#include <cmath>

using namespace std;

void updatePageRank(const pagerank::CsrGraph& graph, vector<double>& pagerank, double damping_factor, int max_iterations,double tolerance)
{
    vector<double> new_pagerank(graph.num_nodes, 0.0);
    for (int i = 0; i < max_iterations; i++)
    {
        for (int node = 0; node < graph.num_nodes; node++)
        {
            double rank_sum = 0.0;
            for (uint64_t edge_idx = graph.row_idx[node]; edge_idx < graph.row_idx[node + 1]; edge_idx++)
            {
                int neighbor = graph.col_idx[edge_idx];
                rank_sum += pagerank[neighbor] / (graph.row_idx[neighbor + 1] - graph.row_idx[neighbor]);
            }
            new_pagerank[node] = (1.0 - damping_factor) / graph.num_nodes + damping_factor * rank_sum;
        }
        double diff =0.0;
        for (int j = 0; j < graph.num_nodes; j++)
        {
            diff += std::abs(new_pagerank[j] - pagerank[j]);
            pagerank[j] = new_pagerank[j];
        }
        if (diff < tolerance)
        {
            break;
        }
}
}