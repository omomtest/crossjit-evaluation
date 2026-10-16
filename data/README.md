# Original evaluation records

Only original process timing samples, diagnostic snapshots, logs, task/input
metadata and protocol manifests are published here, ordered by RQ1–RQ4.
All original numeric timing and diagnostic observations are retained, including
the Intel, Apple M4/MacBook and AMD Ryzen 7 9700X RQ1 cohorts used by the paper.

| Directory | Archives |
| --- | --- |
| `rq1/` | `intel.tar.gz`, `apple-m4.tar.gz`, `amd-9700x.tar.gz` |
| `rq2/` | `full-diagnostics.tar.gz`, `adi-fragment.tar.gz`, `adi-allocation.tar.gz` |
| `rq3/` | `scope-ablation.tar.gz`, `storage-replay.tar.gz` |
| `rq4/` | `callbacks.tar.gz`, `inactive.tar.gz` |

RQ4's complete-call guard-MISS analysis also uses the matched FULL diagnostics
and OFF/FULL timing records in `rq3/scope-ablation.tar.gz`. The ADI raw timing
and observation records support both RQ2 and RQ3; the independent GDB allocation
logs are diagnostics and are never included in timing samples.

The published archives omit generated input-array payloads, computed reference
arrays, summary reports, CSV tables, plot inputs, figures, unrelated case studies
and source checkout/version records. Input metadata, seeds and validation
outcomes remain available. The complete historical exports and their reference
arrays are preserved in the author's private local backup.

Personal filesystem paths, host/account identifiers and tar ownership fields
are redacted. Each adjacent member manifest records the original content hash
and the hash of the published record; numerical observations are unchanged.
`manifest.json` verifies archive and member-manifest sizes and hashes.

```bash
python3 scripts/paper/verify_data.py
python3 scripts/paper/derive_results.py --output results/paper-derived
```

Existing derived author artifacts are retained locally under ignored
`data/derived/`; freshly generated derivatives go to ignored `results/`.
