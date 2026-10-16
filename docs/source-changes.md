# Source changes included in the final implementation

The Host change adds an evaluation scope control at region formation. Setting
`PYTHON_JIT_CROSSJIT_MAX_REGION_NODES=1` limits candidates to one node before
closure, physical proof, Provider admission and lowering. With the variable
unset, the production region limit is preserved.

The NumPy Provider adds an early-return check in native preparation and an
execution-description/plan exporter. Collection requires diagnostics to be
enabled and `NPY_CROSSJIT_EVALUATION_AUDIT` to be present. It constructs Python
records only in that diagnostic configuration. It does not run in each timed
plan invocation. Collection failure can reject that diagnostic preparation and
retain ordinary execution. The source change still changes the preparation
binary, so an unmeasured claim of identical performance would be unjustified.

The existing RQ2–RQ4 isolated runtime's recorded hashes match all five current
core source files: the Host, NumPy Provider, native stages, audit exporter and
generated-planner template. Those measurements already use the diagnostic
implementation. Existing RQ1 measurements remain evidence for their original
independent builds; new runs from final `main` are new measurements.

PyRoaring's pending changes constitute its CrossJIT Provider integration:
registration, CRoaring bindings, guarded bitmap compositions, ownership and
result bridging, build-time Generator support, and regression/benchmark
scripts. They affect the PyRoaring integration, not the NumPy RQ1–RQ4 cohorts.
Earlier two-operation PyRoaring measurements and the later XOR increment are
different implementation stages and must retain their original interpretation.

The Generator had no pending source changes at publication. Its existing final
implementation is published with compressed history. Privacy cleanup changes
historical path strings in ancillary reports, not computation or measured
numeric samples.
