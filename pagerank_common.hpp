#pragma once

#include <cstdint>
#include <vector>

namespace pagerank {

extern const double kDampingFactor;
extern const int kMaxIterations;
extern const double kTolerance;
extern const double kValueMax;

struct CsrGraph {
    int num_edges;
    int num_nodes;
    std::vector<uint64_t> col_idx;
    std::vector<uint64_t> row_idx;
    std::vector<double> values;
};

std::vector<double> initPageRanks(int num_nodes);
CsrGraph initNodeGraph(int num_nodes, int num_edges);

}
