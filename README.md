# CrossJIT Evaluation

This repository reproduces RQ1–RQ4 of the current CrossJIT paper. It contains
original timing samples, diagnostics, logs, experiment protocols, frozen
admission inputs, and reproduction scripts. Derived statistics, tables and
figures remain local and are excluded from Git.

| Paper question | Reproduction entry points | Original records |
| --- | --- | --- |
| RQ1: whole-workload performance across platforms | `scripts/rq1/run.py`, `scripts/run_rq1_machine.py` | `data/rq1/`: Intel i5-14600K, Apple M4/MacBook, AMD Ryzen 7 9700X |
| RQ2: optimization coverage and frequency | `scripts/rq2/run.py`, `scripts/rq2/adi.py`, `scripts/rq2/allocation.py` | `data/rq2/`: FULL diagnostics, ADI fragment and independent allocation logs |
| RQ3: additional benefits of multi-operation optimization | `scripts/rq3/run.py`, `scripts/rq3/storage.py` | `data/rq3/`: OFF/SINGLE/FULL scope ablation and storage-permission replay |
| RQ4: guard MISS and inactive-path overhead | `scripts/run_rq4_guard.py`, `scripts/rq4/callbacks.py`, `scripts/rq4/inactive.py` | `data/rq4/`: callback and inactive-path experiments; matched FULL counts and OFF/FULL times come from RQ3 |

## Reproduce the measurements

Publish the final implementation on `main` in each source repository and make
the repositories publicly readable before building. Docker downloads all source
repositories through HTTPS, without SSH forwarding or personal credentials.
The candidate repositories are resolved from `main` once per build; RQ1–RQ4 use
that same candidate binary and frozen v32 strategy. RQ1 additionally uses the
independent baseline builds defined in `config/sources.json`. RQ2–RQ4 use the
integrated binary with CrossJIT OFF as the same-binary reference.

```bash
python3 scripts/run_docker_pipeline.py --results results/reproduction --jobs 8
```

Default protocol: all 53 NPBench workloads, preset S, six BLAS threads,
`PYTHON_JIT_STRESS=1`, 300 warmup calls, 100 timed calls, and ten independent
process trials. Input restoration and validation occur outside timing;
diagnostics run in separate processes. Use native `linux/amd64` or
`linux/arm64` Docker execution, not emulation.

To run only RQ1 on another machine:

```bash
python3 scripts/run_docker_pipeline.py --results results/rq1-machine --rqs rq1 --jobs 8
```

For RQ2–RQ4 without RQ1:

```bash
python3 scripts/run_redesigned_docker_pipeline.py --results results/rq2-rq4 --jobs 8
```

The ADI fragment belongs to RQ2/RQ3. The generated-planner storage replay belongs
to RQ3. Guard-MISS, callback and inactive-path costs belong to RQ4. PyRoaring and
historical unrelated case studies are outside this evaluation package.

## Reanalyze the paper's original records

```bash
python3 scripts/paper/verify_data.py
python3 scripts/paper/derive_results.py --output results/paper-derived
```

This path runs no performance experiments. It checks archive SHA256 values,
unpacks original records into `.cache/`, and writes derived results under
`results/`. Existing author-side derivatives are retained in ignored
`data/derived/`. Neither directory is uploaded.

All timing ratios use the ratio of process medians,
`median(baseline_process_times) / median(candidate_process_times)` (S_A).
Confidence intervals use 10,000 paired-process bootstrap draws. Geometric-mean
intervals share resampled trial indices across workloads and recompute both
the ratios and the aggregate. See [the paper mapping](docs/paper-alignment.md)
and [the data description](data/README.md).

`config/optimizations/v32/` holds build-time admission inputs and their raw
measurement evidence. These fixed inputs are required to reproduce the paper's
strategy; the runner does not retune admission during an ablation or on another
platform. They are separate from the derived evaluation reports.

## Checks

```bash
python3 -m unittest discover -s tests
```

Docker execution was not exercised during the publication reorganization.
The source downloader and command planning have local tests; an actual build
requires the source repositories to be public.
