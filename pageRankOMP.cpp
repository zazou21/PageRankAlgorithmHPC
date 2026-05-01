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
//  Break
// Parallelize loops over nodes
// Use reduction for diff

#include "pagerank_common.hpp"

using namespace std;

void updatePageRankOMP(const pagerank::CsrGraph& graph, vector<double>& pagerank, double damping_factor, int max_iterations,double tolerance)
{
    vector<double> new_pagerank(graph.num_nodes, 0.0);
    for (int i = 0; i < max_iterations; i++)
    {
        #pragma omp parallel for
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
        #pragma omp parallel for reduction(+:diff)
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

void updatePageRankOMPDynamic(const pagerank::CsrGraph& graph, vector<double>& pagerank, double damping_factor, int max_iterations,double tolerance,int chunk_size)
{
    vector<double> new_pagerank(graph.num_nodes, 0.0);
    for (int i = 0; i < max_iterations; i++)
    {
        #pragma omp parallel for schedule(dynamic, chunk_size)
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
        #pragma omp parallel for reduction(+:diff) schedule(dynamic, chunk_size)
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

void updatePageRankOMPGuided(const pagerank::CsrGraph& graph, vector<double>& pagerank, double damping_factor, int max_iterations,double tolerance,int chunk_size)
{
    vector<double> new_pagerank(graph.num_nodes, 0.0);
    for (int i = 0; i < max_iterations; i++)
    {
        #pragma omp parallel for schedule(guided, chunk_size)
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
        #pragma omp parallel for reduction(+:diff) schedule(guided, chunk_size)
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