#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pagerank {

extern const double kDampingFactor;
extern const int kMaxIterations;
extern const double kTolerance;
extern const double kValueMax;

struct CsrGraph {
    int num_edges = 0;
    int num_nodes = 0;
    std::vector<uint64_t> col_idx;
    std::vector<uint64_t> row_idx;
    std::vector<uint64_t> out_degree;
};

std::vector<double> initPageRanks(int num_nodes);
CsrGraph initNodeGraph(int num_nodes, int num_edges, unsigned int seed = 12345u);
CsrGraph loadEdgeList(const std::string& path);

}
