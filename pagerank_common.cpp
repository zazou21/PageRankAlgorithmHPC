#include "pagerank_common.hpp"

#include <cstdlib>

namespace pagerank {

const double kDampingFactor = 0.85;
const int kMaxIterations = 100;
const double kTolerance = 1e-6;
const double kValueMax = static_cast<double>(RAND_MAX);

std::vector<double> initPageRanks(int num_nodes) {
    return std::vector<double>(num_nodes, 1.0 / num_nodes);
}

CsrGraph initNodeGraph(int num_nodes, int num_edges) {
    CsrGraph graph;
    graph.num_nodes = num_nodes;
    graph.num_edges = num_edges;
    graph.col_idx.resize(num_edges);
    graph.row_idx.resize(num_nodes + 1);
    graph.values.resize(num_edges, 1.0);

    for (int i = 0; i < num_edges; i++) {
        graph.values[i] = rand() / kValueMax;
    }

    return graph;
}

}  // namespace pagerank
