#include "pagerank_common.hpp"
#include "pageRankSeq.cpp"
#include "pageRankOMP.cpp"
#include "pageRankMPI.cpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cstdlib>


//csv format: algorithm,number of processes for mpi,number of threads for omp,number of nodes,number of edges,time taken for each iteration,average time for all iterations,min time,max time

#define NUM_ITERATIONS 10
#define MIN_NODES 1000000
#define MAX_NODES 10000000
#define MIN_EDGES 5000000
#define MAX_EDGES 50000000

using namespace std;

struct TimingStats {
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = 0.0;
    int count = 0;

    void add(double value) {
        sum += value;
        min = std::min(min, value);
        max = std::max(max, value);
        count++;
    }

    double average() const {
        return count > 0 ? (sum / count) : 0.0;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(static_cast<unsigned int>(time(0)));
    double start = 0.0;
    double end = 0.0;
    TimingStats seq_stats;
    TimingStats omp_stats;
    TimingStats mpi_stats;
    long long total_nodes = 0;
    long long total_edges = 0;
    vector<int> iteration_nodes;
    vector<int> iteration_edges;
    vector<double> seq_times;
    vector<double> omp_times;
    vector<double> mpi_times;

    const int omp_threads = omp_get_max_threads();
    const string omp_schedule = "static";

    if (world_rank == 0) {
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int num_nodes = MIN_NODES + rand() % (MAX_NODES - MIN_NODES + 1);
            int num_edges = MIN_EDGES + rand() % (MAX_EDGES - MIN_EDGES + 1);
            total_nodes += num_nodes;
            total_edges += num_edges;
            iteration_nodes.push_back(num_nodes);
            iteration_edges.push_back(num_edges);
        }

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int num_nodes = iteration_nodes[i];
            int num_edges = iteration_edges[i];

            pagerank::CsrGraph graph = pagerank::initNodeGraph(num_nodes, num_edges);
            vector<double> pagerank_seq = pagerank::initPageRanks(num_nodes);

            start = omp_get_wtime();
            updatePageRank(graph, pagerank_seq, pagerank::kDampingFactor, pagerank::kMaxIterations, pagerank::kTolerance);
            end = omp_get_wtime();
            double seq_time = end - start;
            seq_stats.add(seq_time);
            seq_times.push_back(seq_time);
            cout << "Sequential Time: " << seq_time << " seconds" << endl;
        }

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int num_nodes = iteration_nodes[i];
            int num_edges = iteration_edges[i];

            pagerank::CsrGraph graph = pagerank::initNodeGraph(num_nodes, num_edges);
            vector<double> pagerank_omp = pagerank::initPageRanks(num_nodes);

            start = omp_get_wtime();
            updatePageRankOMP(graph, pagerank_omp, pagerank::kDampingFactor, pagerank::kMaxIterations, pagerank::kTolerance);
            end = omp_get_wtime();
            double omp_time = end - start;
            omp_stats.add(omp_time);
            omp_times.push_back(omp_time);
            cout << "OpenMP Time: " << omp_time << " seconds" << endl;
        }
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        int num_nodes = 0;
        int num_edges = 0;

        if (world_rank == 0) {
            num_nodes = iteration_nodes[i];
            num_edges = iteration_edges[i];
        }

        MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        pagerank::CsrGraph graph = pagerank::initNodeGraph(num_nodes, num_edges);
        vector<double> pagerank_mpi = pagerank::initPageRanks(num_nodes);

        MPI_Barrier(MPI_COMM_WORLD);
        double mpi_start = MPI_Wtime();
        updatePageRankMPI(graph, pagerank_mpi, pagerank::kDampingFactor, pagerank::kMaxIterations, pagerank::kTolerance);
        MPI_Barrier(MPI_COMM_WORLD);
        double mpi_end = MPI_Wtime();
        double local_mpi_time = mpi_end - mpi_start;
        double max_mpi_time = 0.0;

        MPI_Reduce(&local_mpi_time, &max_mpi_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            mpi_stats.add(max_mpi_time);
            mpi_times.push_back(max_mpi_time);
            cout << "MPI Time: " << max_mpi_time << " seconds" << endl;
        }
    }

    if (world_rank == 0) {
        const long long avg_nodes = (NUM_ITERATIONS > 0)
            ? static_cast<long long>((total_nodes + (NUM_ITERATIONS / 2)) / NUM_ITERATIONS)
            : 0;
        const long long avg_edges = (NUM_ITERATIONS > 0)
            ? static_cast<long long>((total_edges + (NUM_ITERATIONS / 2)) / NUM_ITERATIONS)
            : 0;

        const std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
        const std::filesystem::path results_path = exe_path.parent_path() / "results.csv";

        ifstream existing(results_path);
        bool write_header = true;
        if (existing.good()) {
            write_header = (existing.peek() == ifstream::traits_type::eof());
        }

        ofstream out(results_path, ios::app);
        if (write_header) {
            out << "algorithm,number of processes for mpi,number of threads for omp,number of nodes,number of edges,time taken for each iteration,average time for all iterations,min time,max time\n";
        }

        auto times_to_string = [](const vector<double>& times) {
            string result;
            for (size_t i = 0; i < times.size(); ++i) {
                if (i > 0) {
                    result += "|";
                }
                result += std::to_string(times[i]);
            }
            return result;
        };

        auto write_row = [&](const string& algorithm,
                             const string& mpi_processes,
                             const string& omp_threads_value,
                             const string& nodes_value,
                             const string& edges_value,
                             const string& iteration_times_value,
                             const TimingStats& stats) {
            out << algorithm << ","
                << mpi_processes << ","
                << omp_threads_value << ","
                << nodes_value << ","
                << edges_value << ","
                << iteration_times_value << ","
                << stats.average() << ","
                << stats.min << ","
                << stats.max << "\n";
        };

        const string avg_nodes_str = std::to_string(avg_nodes);
        const string avg_edges_str = std::to_string(avg_edges);
        write_row("Sequential",
                  "NA",
                  "NA",
                  avg_nodes_str,
                  avg_edges_str,
                  times_to_string(seq_times),
                  seq_stats);
        write_row("OpenMP",
                  "NA",
                  std::to_string(omp_threads),
                  avg_nodes_str,
                  avg_edges_str,
                  times_to_string(omp_times),
                  omp_stats);
        write_row("MPI",
                  std::to_string(world_size),
                  "NA",
                  avg_nodes_str,
                  avg_edges_str,
                  times_to_string(mpi_times),
                  mpi_stats);
    }

    MPI_Finalize();
    return 0;
}