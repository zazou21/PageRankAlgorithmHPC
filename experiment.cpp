#include "pagerank_common.hpp"
#include "pageRankSeq.cpp"
#include "pageRankOMP.cpp"
#include "pageRankMPI.cpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>

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

struct Config {
    int num_nodes = 100000;
    int num_edges = 500000;
    int iterations = 10;
    int max_pr_iters = pagerank::kMaxIterations;
    double damping = pagerank::kDampingFactor;
    double tolerance = pagerank::kTolerance;
    unsigned int seed = 12345u;
    string input_path;
    string results_path;
    int omp_chunk = 1024;
    bool run_seq = true;
    bool run_omp = true;
    bool run_omp_dynamic = true;
    bool run_omp_guided = true;
    bool run_mpi = true;
    bool verify = true;
    double verify_tol = 1e-6;
};

static void printUsage(const char* exe) {
    cout << "Usage: " << exe << " [options]\n"
         << "  --nodes N            number of nodes (synthetic graph)\n"
         << "  --edges M            number of edges (synthetic graph)\n"
         << "  --iters K            number of timed repetitions\n"
         << "  --max-pr-iters K     PageRank max iterations\n"
         << "  --damping D          damping factor (default 0.85)\n"
         << "  --tol T              convergence tolerance\n"
         << "  --seed S             RNG seed for synthetic graph\n"
         << "  --input PATH         load edge-list file instead of synthetic\n"
         << "  --results PATH       CSV results file\n"
         << "  --omp-chunk N        chunk size for dynamic/guided schedules\n"
         << "  --no-seq / --no-omp / --no-mpi\n"
         << "  --no-omp-dynamic / --no-omp-guided\n"
         << "  --no-verify          skip correctness check\n";
}

static bool parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto next = [&](const string& name, string& out) {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << "\n";
                return false;
            }
            out = argv[++i];
            return true;
        };
        string v;
        if (a == "--nodes" && next(a, v)) cfg.num_nodes = stoi(v);
        else if (a == "--edges" && next(a, v)) cfg.num_edges = stoi(v);
        else if (a == "--iters" && next(a, v)) cfg.iterations = stoi(v);
        else if (a == "--max-pr-iters" && next(a, v)) cfg.max_pr_iters = stoi(v);
        else if (a == "--damping" && next(a, v)) cfg.damping = stod(v);
        else if (a == "--tol" && next(a, v)) cfg.tolerance = stod(v);
        else if (a == "--seed" && next(a, v)) cfg.seed = static_cast<unsigned int>(stoul(v));
        else if (a == "--input" && next(a, v)) cfg.input_path = v;
        else if (a == "--results" && next(a, v)) cfg.results_path = v;
        else if (a == "--omp-chunk" && next(a, v)) cfg.omp_chunk = stoi(v);
        else if (a == "--no-seq") cfg.run_seq = false;
        else if (a == "--no-omp") cfg.run_omp = false;
        else if (a == "--no-mpi") cfg.run_mpi = false;
        else if (a == "--no-omp-dynamic") cfg.run_omp_dynamic = false;
        else if (a == "--no-omp-guided") cfg.run_omp_guided = false;
        else if (a == "--no-verify") cfg.verify = false;
        else if (a == "-h" || a == "--help") { printUsage(argv[0]); return false; }
        else { cerr << "Unknown arg: " << a << "\n"; printUsage(argv[0]); return false; }
    }
    return true;
}

static double l1Diff(const vector<double>& a, const vector<double>& b) {
    double d = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) d += std::abs(a[i] - b[i]);
    return d;
}

static double linfDiff(const vector<double>& a, const vector<double>& b) {
    double d = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

static string timesToString(const vector<double>& times) {
    string s;
    for (size_t i = 0; i < times.size(); ++i) {
        if (i > 0) s += "|";
        s += std::to_string(times[i]);
    }
    return s;
}

static double loadBaselineAvg(const std::filesystem::path& results_path,
                              int nodes, int edges) {
    ifstream in(results_path);
    if (!in.good()) return 0.0;

    string line;
    double found = 0.0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> cols;
        while (std::getline(ss, field, ',')) {
            cols.push_back(field);
        }
        if (cols.size() < 7) continue;
        if (cols[0] != "Sequential") continue;
        try {
            if (std::stoi(cols[3]) != nodes) continue;
            if (std::stoi(cols[4]) != edges) continue;
            found = std::stod(cols[6]);
        } catch (...) {
            continue;
        }
    }
    return found;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Config cfg;
    if (!parseArgs(argc, argv, cfg)) {
        MPI_Finalize();
        return world_rank == 0 ? 0 : 0;
    }

    pagerank::CsrGraph graph;
    if (world_rank == 0) {
        if (!cfg.input_path.empty()) {
            graph = pagerank::loadEdgeList(cfg.input_path);
            cfg.num_nodes = graph.num_nodes;
            cfg.num_edges = graph.num_edges;
        } else {
            graph = pagerank::initNodeGraph(cfg.num_nodes, cfg.num_edges, cfg.seed);
        }
        cout << "Graph: " << graph.num_nodes << " nodes, "
             << graph.num_edges << " edges\n";
    }

    if (cfg.run_mpi) {
        broadcastGraph(graph, 0, MPI_COMM_WORLD);
    }

    const int omp_threads = omp_get_max_threads();

    TimingStats seq_stats, omp_stats, omp_dyn_stats, omp_gui_stats, mpi_stats;
    vector<double> seq_times, omp_times, omp_dyn_times, omp_gui_times, mpi_times;
    vector<double> rank_seq, rank_omp, rank_omp_dyn, rank_omp_gui, rank_mpi;

    if (world_rank == 0 && cfg.run_seq) {
        for (int i = 0; i < cfg.iterations; ++i) {
            vector<double> r = pagerank::initPageRanks(graph.num_nodes);
            const double t0 = omp_get_wtime();
            updatePageRank(graph, r, cfg.damping, cfg.max_pr_iters, cfg.tolerance);
            const double t1 = omp_get_wtime();
            const double dt = t1 - t0;
            seq_stats.add(dt);
            seq_times.push_back(dt);
            if (i == cfg.iterations - 1) rank_seq = std::move(r);
            cout << "[Seq]  iter " << i << " time=" << dt << "s\n";
        }
    }

    if (world_rank == 0 && cfg.run_omp) {
        for (int i = 0; i < cfg.iterations; ++i) {
            vector<double> r = pagerank::initPageRanks(graph.num_nodes);
            const double t0 = omp_get_wtime();
            updatePageRankOMP(graph, r, cfg.damping, cfg.max_pr_iters, cfg.tolerance);
            const double t1 = omp_get_wtime();
            const double dt = t1 - t0;
            omp_stats.add(dt);
            omp_times.push_back(dt);
            if (i == cfg.iterations - 1) rank_omp = std::move(r);
            cout << "[OMP-static]  iter " << i << " time=" << dt << "s\n";
        }
    }

    if (world_rank == 0 && cfg.run_omp && cfg.run_omp_dynamic) {
        for (int i = 0; i < cfg.iterations; ++i) {
            vector<double> r = pagerank::initPageRanks(graph.num_nodes);
            const double t0 = omp_get_wtime();
            updatePageRankOMPDynamic(graph, r, cfg.damping, cfg.max_pr_iters,
                                     cfg.tolerance, cfg.omp_chunk);
            const double t1 = omp_get_wtime();
            const double dt = t1 - t0;
            omp_dyn_stats.add(dt);
            omp_dyn_times.push_back(dt);
            if (i == cfg.iterations - 1) rank_omp_dyn = std::move(r);
            cout << "[OMP-dynamic] iter " << i << " time=" << dt << "s\n";
        }
    }

    if (world_rank == 0 && cfg.run_omp && cfg.run_omp_guided) {
        for (int i = 0; i < cfg.iterations; ++i) {
            vector<double> r = pagerank::initPageRanks(graph.num_nodes);
            const double t0 = omp_get_wtime();
            updatePageRankOMPGuided(graph, r, cfg.damping, cfg.max_pr_iters,
                                    cfg.tolerance, cfg.omp_chunk);
            const double t1 = omp_get_wtime();
            const double dt = t1 - t0;
            omp_gui_stats.add(dt);
            omp_gui_times.push_back(dt);
            if (i == cfg.iterations - 1) rank_omp_gui = std::move(r);
            cout << "[OMP-guided]  iter " << i << " time=" << dt << "s\n";
        }
    }

    if (cfg.run_mpi) {
        for (int i = 0; i < cfg.iterations; ++i) {
            vector<double> r = pagerank::initPageRanks(graph.num_nodes);
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            updatePageRankMPI(graph, r, cfg.damping, cfg.max_pr_iters, cfg.tolerance);
            MPI_Barrier(MPI_COMM_WORLD);
            const double t1 = MPI_Wtime();
            const double local_dt = t1 - t0;
            double max_dt = 0.0;
            MPI_Reduce(&local_dt, &max_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (world_rank == 0) {
                mpi_stats.add(max_dt);
                mpi_times.push_back(max_dt);
                if (i == cfg.iterations - 1) rank_mpi = std::move(r);
                cout << "[MPI]  iter " << i << " time=" << max_dt << "s\n";
            }
        }
    }

    if (world_rank == 0 && cfg.verify && !rank_seq.empty()) {
        auto report = [&](const string& name, const vector<double>& r) {
            if (r.empty()) return;
            cout << "[verify] " << name
                 << " L1=" << l1Diff(rank_seq, r)
                 << " Linf=" << linfDiff(rank_seq, r) << "\n";
        };
        report("OMP-static", rank_omp);
        report("OMP-dynamic", rank_omp_dyn);
        report("OMP-guided", rank_omp_gui);
        report("MPI", rank_mpi);
    }

    if (world_rank == 0) {
        std::filesystem::path results_path;
        if (!cfg.results_path.empty()) {
            results_path = cfg.results_path;
        } else {
            const std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
            results_path = exe_path.parent_path() / "results.csv";
        }

        ifstream existing(results_path);
        bool write_header = true;
        if (existing.good()) {
            write_header = (existing.peek() == ifstream::traits_type::eof());
        }

        ofstream out(results_path, ios::app);
        if (write_header) {
            out << "algorithm,mpi_processes,omp_threads,nodes,edges,"
                   "per_iter_times,avg_time,min_time,max_time,"
                   "speedup_vs_seq,efficiency\n";
        }

        double seq_avg = seq_stats.average();
        if (seq_avg <= 0.0 && !cfg.run_seq) {
            seq_avg = loadBaselineAvg(results_path, graph.num_nodes, graph.num_edges);
        }

        auto write_row = [&](const string& algorithm,
                             const string& mpi_p,
                             const string& omp_t,
                             const vector<double>& times,
                             const TimingStats& stats,
                             int parallel_units) {
            const double avg = stats.average();
            const double speedup = (seq_avg > 0.0 && avg > 0.0) ? (seq_avg / avg) : 0.0;
            const double efficiency = (parallel_units > 0 && speedup > 0.0)
                                    ? (speedup / parallel_units) : 0.0;
            out << algorithm << ","
                << mpi_p << ","
                << omp_t << ","
                << graph.num_nodes << ","
                << graph.num_edges << ","
                << timesToString(times) << ","
                << avg << ","
                << stats.min << ","
                << stats.max << ","
                << speedup << ","
                << efficiency << "\n";
        };

        if (cfg.run_seq) {
            write_row("Sequential", "NA", "NA", seq_times, seq_stats, 1);
        }
        if (cfg.run_omp) {
            write_row("OpenMP-static", "NA", std::to_string(omp_threads),
                      omp_times, omp_stats, omp_threads);
            if (cfg.run_omp_dynamic) {
                write_row("OpenMP-dynamic", "NA", std::to_string(omp_threads),
                          omp_dyn_times, omp_dyn_stats, omp_threads);
            }
            if (cfg.run_omp_guided) {
                write_row("OpenMP-guided", "NA", std::to_string(omp_threads),
                          omp_gui_times, omp_gui_stats, omp_threads);
            }
        }
        if (cfg.run_mpi) {
            write_row("MPI", std::to_string(world_size), "NA",
                      mpi_times, mpi_stats, world_size);
        }
    }

    MPI_Finalize();
    return 0;
}
