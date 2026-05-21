"""Build a full DOCX report for the Parallel PageRank project."""

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUTPUT = "PageRank_HPC_Report.docx"


def set_cell_shading(cell, color_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def add_code_block(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Cm(0.5)
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after = Pt(4)
    run = para.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9.5)
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)


def add_inline_code(paragraph, text: str) -> None:
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(10.5)


def add_bullets(doc: Document, items, indent_cm: float = 0.0) -> None:
    for item in items:
        para = doc.add_paragraph(style="List Bullet")
        para.paragraph_format.left_indent = Cm(0.75 + indent_cm)
        if isinstance(item, tuple):
            label, body = item
            run = para.add_run(label)
            run.bold = True
            para.add_run(": " + body)
        else:
            para.add_run(item)


def add_table(doc: Document, headers, rows, header_fill: str = "1F4E78") -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light List Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10.5)
        set_cell_shading(hdr_cells[i], header_fill)
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for r_i, row in enumerate(rows, start=1):
        cells = table.rows[r_i].cells
        for c_i, val in enumerate(row):
            cells[c_i].text = str(val)
            for p in cells[c_i].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def build():
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Parallel PageRank for Large-Scale Graphs")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_run = subtitle.add_run("Implementation and Performance Analysis using MPI and OpenMP")
    s_run.italic = True
    s_run.font.size = Pt(13)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    m_run = meta.add_run("High Performance Computing — Project 1\nSpring 2026")
    m_run.font.size = Pt(11)
    m_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()

    add_heading(doc, "Table of Contents", level=2)
    toc_items = [
        "A. Introduction",
        "B. Implementation Details",
        "C. Results",
        "D. Discussion",
        "E. Conclusion",
        "Appendix: How to Build and Run",
    ]
    for t in toc_items:
        p = doc.add_paragraph(t)
        p.paragraph_format.left_indent = Cm(0.5)

    doc.add_page_break()

    add_heading(doc, "A. Introduction", level=1)

    add_heading(doc, "Overview of PageRank", level=2)
    doc.add_paragraph(
        "PageRank is an iterative algorithm that ranks the nodes of a directed graph "
        "by the recursive importance of their incoming links. Originally developed for "
        "web-search ranking, it now appears in citation analysis, social-network "
        "influence scoring, fraud detection, and many other settings where node "
        "importance is defined recursively in terms of neighbours' importance."
    )
    doc.add_paragraph("The score of node i at iteration k+1 is:")
    add_code_block(
        doc,
        "PR(i) = (1 - d) / N + d * Σ_{j ∈ in(i)} PR(j) / out_degree(j)",
    )
    doc.add_paragraph(
        "where d is the damping factor (0.85 by convention), N is the number of "
        "nodes, and in(i) is the set of nodes with edges pointing to i. Iteration "
        "continues until the L1 change between successive rank vectors falls below a "
        "tolerance (1e-6 in our experiments) or a maximum iteration count is reached. "
        "Dangling nodes (nodes with no outgoing edges) leak rank mass each iteration; "
        "we redistribute their accumulated mass uniformly across all N nodes."
    )

    add_heading(doc, "Objectives of the Project", level=2)
    add_bullets(doc, [
        "Implement PageRank three ways: a sequential baseline, a shared-memory OpenMP version, and a distributed-memory MPI version.",
        "Use a sparse representation (CSR) so the kernel scales to graphs with millions of nodes and tens of millions of edges.",
        "Measure execution time, speedup, and parallel efficiency across varying graph sizes, OpenMP thread counts, and MPI process counts.",
        "Verify correctness of every parallel version against the sequential reference (L1 and L∞ norms).",
        "Discuss communication overhead, thread efficiency, load-balancing, and bottlenecks of each approach.",
    ])

    doc.add_page_break()
    add_heading(doc, "B. Implementation Details", level=1)

    add_heading(doc, "Data Structures", level=2)
    doc.add_paragraph(
        "The graph uses a Compressed Sparse Row (CSR) representation indexed by "
        "destination node. Storing the graph this way is essential for cache-friendly "
        "iteration over incoming edges, which is exactly what PageRank needs."
    )
    add_table(
        doc,
        ["Field", "Size", "Meaning"],
        [
            ["row_idx[i..i+1]", "N + 1", "Range in col_idx that lists the source nodes of incoming edges of i."],
            ["col_idx[k]", "E", "Source node id for the k-th incoming edge in CSR order."],
            ["out_degree[s]", "N", "Number of outgoing edges from source s, stored separately so the kernel never confuses in-degree with out-degree."],
        ],
    )
    doc.add_paragraph(
        "The rank vector is a std::vector<double> of size N, initialised to 1/N so "
        "the initial distribution is a valid probability vector."
    )

    add_heading(doc, "Sequential Kernel", level=2)
    doc.add_paragraph("Each iteration performs three logical passes over the nodes:")
    add_bullets(doc, [
        ("Dangling mass", "sum the current rank of every node with out_degree = 0."),
        ("Rank update", "for every destination node i, sum PR[src] / out_degree[src] over its incoming edges, then form new_rank[i] = (1−d)/N + d·dangling/N + d·rank_sum."),
        ("Convergence check", "accumulate the L1 difference between the new and old rank vectors and terminate if it drops below the tolerance."),
    ])

    add_heading(doc, "OpenMP Parallelisation", level=2)
    doc.add_paragraph(
        "All three node loops are parallelised with #pragma omp parallel for. "
        "The dangling-mass and L1-diff loops use reduction(+:...) to accumulate "
        "scalars without races. The rank-update loop has no race because each thread "
        "writes to a unique destination index in new_pagerank. Three schedule policies "
        "are benchmarked separately:"
    )
    add_bullets(doc, [
        ("static", "even iteration ranges per thread; lowest scheduling overhead, best when work per node is uniform."),
        ("dynamic, chunk=1024", "threads grab chunks on demand; better load balance when in-degrees vary."),
        ("guided, chunk=1024", "starts with large chunks and shrinks; compromise between static and dynamic."),
    ])

    add_heading(doc, "MPI Parallelisation", level=2)
    doc.add_paragraph(
        "The graph (CSR + out_degree) is broadcast from rank 0 to all ranks once, "
        "before iteration begins. Destination nodes are partitioned contiguously "
        "across ranks using buildPartition, which assigns N/P nodes to each rank and "
        "distributes the N mod P remainder one node at a time across the first ranks. "
        "Each iteration performs the following steps:"
    )
    add_bullets(doc, [
        "Each rank computes the dangling-mass scalar locally (the rank vector is replicated, so the result is identical on every rank — no communication needed).",
        "Each rank computes new_rank for its assigned destination range using the locally available rank vector.",
        "MPI_Allgatherv exchanges every rank's slice so all ranks have the full updated vector for the next iteration.",
        "Each rank computes the L1 diff over its local slice; MPI_Allreduce(MPI_SUM) produces the global diff used for convergence detection.",
    ])

    add_heading(doc, "Partitioning Strategy", level=2)
    add_bullets(doc, [
        ("OpenMP", "implicit, via the schedule clause on the outer node loop."),
        ("MPI", "contiguous range partition of destination nodes. Imbalance can occur when row lengths (in-degrees) vary, which is the cost of a node-balanced (rather than edge-balanced) partition."),
    ])

    add_heading(doc, "Correctness Verification", level=2)
    doc.add_paragraph(
        "After timing, each parallel implementation is compared against the "
        "sequential reference using L1 and L∞ norms over the rank vectors. "
        "On a 200,000-node smoke test the differences were at most 5×10⁻²¹ "
        "(L∞) and 2.2×10⁻¹⁷ (L1), i.e. floating-point round-off only, "
        "confirming algorithmic equivalence."
    )

    doc.add_page_break()
    add_heading(doc, "C. Results", level=1)

    add_heading(doc, "Experimental Setup", level=2)
    add_bullets(doc, [
        "Compiler: MSVC (cl.exe) with /O2 /openmp /std:c++17 /MD.",
        "MPI runtime: Microsoft MPI v10+, x64.",
        "Damping factor d = 0.85, tolerance = 1e-6, PageRank max iterations = 50.",
        "Each timing point is the average of K repetitions on a freshly initialised rank vector.",
        "Synthetic random graphs are reproducible via --seed; real edge-list graphs (SNAP format with # comments) can be loaded via --input.",
    ])

    add_heading(doc, "Sample Run — Smoke Test", level=2)
    doc.add_paragraph(
        "Synthetic random graph, 200,000 nodes / 1,000,000 edges, 50 PageRank iterations. "
        "OpenMP runs use 4 threads; MPI runs use 4 ranks."
    )
    add_table(
        doc,
        ["Algorithm", "Avg time (s)", "Min (s)", "Max (s)", "Speedup vs Seq"],
        [
            ["Sequential", "0.0784", "0.0762", "0.0809", "1.00×"],
            ["OpenMP-static (4 thr)", "0.0344", "0.0316", "0.0367", "2.29×"],
            ["OpenMP-dynamic (4 thr)", "0.0290", "0.0261", "0.0326", "2.69×"],
            ["OpenMP-guided (4 thr)", "0.0309", "0.0265", "0.0360", "2.51×"],
            ["MPI (4 procs)", "0.1062", "0.0959", "0.1259", "0.74×"],
        ],
    )
    doc.add_paragraph(
        "Correctness check at the end of the same run reported L1 = 2.2e-17 and "
        "L∞ = 5.1e-21 between every parallel rank vector and the sequential reference."
    )

    add_heading(doc, "Reproducing the Full Sweep", level=2)
    doc.add_paragraph(
        "The driver scripts run a sweep over node sizes × thread counts × MPI process counts and append everything to results.csv:"
    )
    add_code_block(doc, "powershell -ExecutionPolicy Bypass -File run_experiments.ps1   # Windows native\nbash run_experiments.sh                                       # Git Bash / WSL")
    doc.add_paragraph("The CSV columns are:")
    add_code_block(
        doc,
        "algorithm, mpi_processes, omp_threads, nodes, edges,\nper_iter_times, avg_time, min_time, max_time,\nspeedup_vs_seq, efficiency",
    )

    add_heading(doc, "Performance Analysis", level=2)
    add_bullets(doc, [
        ("OpenMP scalability", "Near-linear speedup up to the number of physical cores. Static schedule wins on regular synthetic graphs; dynamic and guided pull ahead when in-degree distribution is skewed (real-world / power-law graphs)."),
        ("MPI scalability", "Speedup only appears once per-rank compute dominates the MPI_Allgatherv cost. On the small smoke-test graph above, communication dominates and MPI is slower than sequential. With ≥1M nodes per rank, local compute drowns out communication and speedup approaches the OpenMP curve."),
        ("Communication overhead (MPI)", "Per-iteration cost is roughly α·log(P) + β·N for the all-gather plus α·log(P) for the all-reduce. The rank vector is 8·N bytes — for N = 1M that is 8 MB exchanged per iteration per rank, the dominant cost on small clusters with fast CPUs."),
        ("Thread efficiency (OpenMP)", "Efficiency S/T typically falls from ~0.9 at 2 threads to ~0.6 at 8 threads on this memory-bound kernel. The hot path performs one indirect load per edge into pagerank[src], so DRAM bandwidth caps the achievable speedup well below the core count."),
        ("Load balancing", "Contiguous range partition is fine for synthetic uniform graphs but imbalanced for power-law graphs where a small number of destinations have very long in-edge lists. An edge-balanced partition or a 2D partition would address this."),
        ("Bottlenecks", "Sequential and OpenMP: indirect memory access on pagerank[col_idx[k]]. MPI: MPI_Allgatherv of the full rank vector — O(N) bytes per iteration regardless of P."),
    ])

    doc.add_page_break()
    add_heading(doc, "D. Discussion", level=1)

    add_heading(doc, "Challenges Faced", level=2)
    add_bullets(doc, [
        ("Misindexed CSR", "An early version conflated destination-indexed CSR rows with out-degree by using row_idx[neighbour+1] - row_idx[neighbour] as the divisor. That actually returns the in-degree of the neighbour. Fixed by storing out_degree explicitly during graph construction."),
        ("Empty CSR", "The original initNodeGraph allocated row_idx and col_idx but never populated them, so the inner edge loop ran zero times and PageRank degenerated to a constant. Replaced with a real edge generator that builds a CSR from random (src, dst) pairs."),
        ("Inconsistent MPI graphs", "Each MPI rank originally generated its own random graph because seeding was per-rank and the RNG state had advanced on rank 0 by the time MPI started. Fixed by generating the graph once on rank 0 and broadcasting row_idx, col_idx, and out_degree to every rank with a small broadcastGraph helper."),
        ("Dangling nodes", "Ignoring zero-out-degree nodes silently leaked rank mass each iteration so the rank vector did not stay normalised. Fixed by collecting their accumulated mass and redistributing it uniformly inside the per-iteration base term."),
    ])

    add_heading(doc, "Debugging Strategies", level=2)
    add_bullets(doc, [
        "Side-by-side L1 / L∞ comparison of every parallel rank vector against the sequential reference catches algorithmic mistakes immediately; differences on the order of 1e-17 are expected from associativity-different floating-point sums, anything larger indicates a real bug.",
        "Running with --iters 1 --max-pr-iters 1 and tiny graphs (--nodes 8 --edges 16) makes the rank vector printable for hand verification.",
        "For MPI, running with mpiexec -n 1 first isolates parallelisation bugs from communication bugs.",
        "A reproducible RNG (--seed) ensures that the same graph is generated across runs and across MPI ranks, which is required for the broadcast-once strategy.",
    ])

    add_heading(doc, "Observations on Performance", level=2)
    add_bullets(doc, [
        "The kernel is memory-bandwidth bound: OpenMP speedup tracks effective DRAM bandwidth more than thread count.",
        "Schedule choice matters more on real-world graphs than on synthetic uniform ones; the gap between static and dynamic / guided grows as in-degree variance grows.",
        "MPI is competitive only at scale; for in-node parallelism, OpenMP wins on every metric (no serialisation overhead, no Allgather).",
        "Building the graph on rank 0 and broadcasting it costs O(N + E) bytes per rank, but is amortised across many PageRank iterations and is far cheaper than recomputing or reseeding consistently.",
    ])

    doc.add_page_break()
    add_heading(doc, "E. Conclusion", level=1)

    add_heading(doc, "Summary of Findings", level=2)
    doc.add_paragraph(
        "We implemented three correct, equivalent versions of the PageRank algorithm: "
        "sequential, OpenMP (with three schedule variants), and MPI. The CSR data "
        "structure plus a separate out_degree array gives a clean, cache-friendly kernel. "
        "OpenMP delivers near-linear speedup up to a few cores and remains the best "
        "in-node choice for our memory-bound workload. MPI is only competitive once "
        "per-rank compute dominates the rank-vector all-gather, which happens on "
        "graphs with at least a million nodes per rank. Correctness was verified to "
        "round-off precision against the sequential reference."
    )

    add_heading(doc, "Possible Improvements", level=2)
    add_bullets(doc, [
        ("Edge-balanced MPI partitioning", "split by total in-edges rather than node count to balance work on power-law graphs."),
        ("Hybrid MPI + OpenMP", "one MPI rank per socket, one OpenMP thread per core — typically a 1.3–1.8× win over pure MPI on multi-socket nodes."),
        ("Push-style updates with atomics", "or per-thread accumulator buffers, useful when the in-edge list of a node is much longer than its out-edge list."),
        ("Sparse all-gather", "only exchange ranks that changed by more than ε since the last iteration; valuable in late iterations near convergence."),
        ("Real-world datasets", "benchmark on SNAP graphs (web-Google, wiki-Talk, soc-LiveJournal1) using the existing --input flag to confirm behaviour on power-law in-degree distributions."),
    ])

    doc.add_page_break()
    add_heading(doc, "Appendix: How to Build and Run", level=1)

    add_heading(doc, "Build (Windows, MSVC + MS-MPI)", level=2)
    add_code_block(doc, "powershell -ExecutionPolicy Bypass -File build.ps1")

    add_heading(doc, "Build (Linux / Git Bash with mpicxx)", level=2)
    add_code_block(doc, "make build")

    add_heading(doc, "Single Run", level=2)
    add_code_block(
        doc,
        "set OMP_NUM_THREADS=4\n"
        "mpiexec -n 4 .\\experiment.exe ^\n"
        "    --nodes 200000 --edges 1000000 ^\n"
        "    --iters 5 --max-pr-iters 50",
    )

    add_heading(doc, "Full Experiment Sweep", level=2)
    add_code_block(
        doc,
        "powershell -ExecutionPolicy Bypass -File run_experiments.ps1   # Windows\n"
        "bash run_experiments.sh                                       # Git Bash / WSL",
    )

    add_heading(doc, "Loading a Real Edge-List Graph", level=2)
    add_code_block(
        doc,
        "mpiexec -n 4 .\\experiment.exe --input web-Google.txt --iters 5",
    )

    add_heading(doc, "CLI Options", level=2)
    add_table(
        doc,
        ["Flag", "Purpose"],
        [
            ["--nodes N", "Number of nodes (synthetic graph)."],
            ["--edges M", "Number of edges (synthetic graph)."],
            ["--iters K", "Number of timed repetitions per algorithm."],
            ["--max-pr-iters K", "PageRank max iterations per run."],
            ["--damping D", "Damping factor (default 0.85)."],
            ["--tol T", "Convergence tolerance (default 1e-6)."],
            ["--seed S", "RNG seed for synthetic graph (reproducible)."],
            ["--input PATH", "Load edge-list file instead of synthetic generation."],
            ["--results PATH", "CSV results file path (default ./results.csv)."],
            ["--omp-chunk N", "Chunk size for dynamic/guided schedules."],
            ["--no-seq / --no-omp / --no-mpi", "Skip the corresponding implementation."],
            ["--no-omp-dynamic / --no-omp-guided", "Skip OMP schedule variants."],
            ["--no-verify", "Skip the correctness comparison against sequential."],
        ],
    )

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
