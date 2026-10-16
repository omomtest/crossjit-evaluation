# Mapping to the current paper

The paper's four questions are whole-workload performance, realized optimization
coverage/frequency, additional multi-operation benefits, and guard-MISS/inactive
overhead, respectively. The public entry points and `data/rq1` through `data/rq4`
follow that order.

RQ1 compares independent baseline and candidate builds on Intel i5-14600K,
Apple M4, and AMD Ryzen 7 9700X. RQ2–RQ4 use the same integrated candidate binary
for OFF, SINGLE and FULL modes. SINGLE sets
`PYTHON_JIT_CROSSJIT_MAX_REGION_NODES=1`; FULL leaves the limit unset. Admission
inputs remain fixed across modes and platforms.

RQ2 counts actual post-warmup successful plan invocations. Traversal fusion and
internal-array non-materialization follow executed plan structure weighted by
HIT counts; storage reuse counts adopted routes. These properties can overlap.
The ADI fragment performs 128 evaluations of
`numerator / (scale * left + offset)` with a 98-element float64 strided view and
Python floats 2, 3 and 4. Independent allocation logs are untimed diagnostics.

RQ3 compares OFF/SINGLE, SINGLE/FULL and OFF/FULL across all 53 workloads, grouped
by actual FULL execution. It also replays 30 captured native-stage descriptions
with reuse permissions enabled and disabled. The replay uses eight-double
synthetic ownership buffers; its storage bytes are not workload bytes saved.

RQ4 groups complete-call OFF/FULL ratios by FULL guard outcomes and reports
`(median(FULL) - median(OFF)) / misses_per_complete_call`. This is net amortized
cost, including HIT benefits and other path differences, not isolated guard
latency. Separate-guards/FULL callback comparisons retain the same admitted
inventory. The inactive experiment reports the 17 inactive workloads separately
from the declared controlled paths.

Every reported timing ratio and interval uses S_A, the ratio of process medians.
Ten paired process trials are resampled 10,000 times. Aggregate intervals share
trial-index draws across workloads and recompute the workload ratios and GM.
The previous median-of-pair-ratios implementation has been replaced accordingly.
Reanalysis of the packaged raw records matched 1,412 current-paper point
estimates and interval fields within numerical tolerance.

The candidate source repositories are resolved from their final `main` once per
Docker build. RQ1–RQ4 then share the same candidate build and fixed policies.
Published measurement archives omit source commit and branch records.
