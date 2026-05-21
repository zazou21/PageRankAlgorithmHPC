#include "pagerank_common.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pagerank {

const double kDampingFactor = 0.85;
const int kMaxIterations = 100;
const double kTolerance = 1e-6;
const double kValueMax = static_cast<double>(RAND_MAX);

std::vector<double> initPageRanks(int num_nodes) {
    return std::vector<double>(num_nodes, 1.0 / num_nodes);
}

namespace {

CsrGraph buildCsrFromEdges(int num_nodes,
                           const std::vector<std::pair<uint64_t, uint64_t>>& edges) {
    CsrGraph graph;
    graph.num_nodes = num_nodes;
    graph.num_edges = static_cast<int>(edges.size());
    graph.row_idx.assign(num_nodes + 1, 0);
    graph.col_idx.assign(edges.size(), 0);
    graph.out_degree.assign(num_nodes, 0);

    for (const auto& e : edges) {
        const uint64_t src = e.first;
        const uint64_t dst = e.second;
        graph.row_idx[dst + 1]++;
        graph.out_degree[src]++;
    }
    for (int i = 1; i <= num_nodes; ++i) {
        graph.row_idx[i] += graph.row_idx[i - 1];
    }

    std::vector<uint64_t> cursor(graph.row_idx.begin(), graph.row_idx.end() - 1);
    for (const auto& e : edges) {
        const uint64_t src = e.first;
        const uint64_t dst = e.second;
        graph.col_idx[cursor[dst]++] = src;
    }

    return graph;
}

}

CsrGraph initNodeGraph(int num_nodes, int num_edges, unsigned int seed) {
    if (num_nodes <= 0 || num_edges < 0) {
        return CsrGraph{};
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, num_nodes - 1);

    std::vector<std::pair<uint64_t, uint64_t>> edges;
    edges.reserve(num_edges);
    for (int i = 0; i < num_edges; ++i) {
        int src = dist(rng);
        int dst = dist(rng);
        if (src == dst) {
            dst = (dst + 1) % num_nodes;
        }
        edges.emplace_back(static_cast<uint64_t>(src), static_cast<uint64_t>(dst));
    }

    return buildCsrFromEdges(num_nodes, edges);
}

CsrGraph loadEdgeList(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open edge-list file: " + path);
    }

    std::vector<std::pair<uint64_t, uint64_t>> edges;
    uint64_t max_node = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::istringstream iss(line);
        uint64_t src = 0, dst = 0;
        if (!(iss >> src >> dst)) continue;
        edges.emplace_back(src, dst);
        max_node = std::max(max_node, std::max(src, dst));
    }

    const int num_nodes = static_cast<int>(max_node + 1);
    return buildCsrFromEdges(num_nodes, edges);
}

}
