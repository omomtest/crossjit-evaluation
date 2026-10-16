#!/usr/bin/env python3
"""Export a saved summary as a Matplotlib figure; no benchmark runs or dependency installs."""
import argparse
import os
from pathlib import Path
import tempfile

from common import canonical_hash, digest, read_json, save


def plot(summary_path, output):
    cache = tempfile.TemporaryDirectory(prefix="crossjit-plot-")
    os.environ.setdefault("MPLCONFIGDIR",cache.name)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = read_json(summary_path)
    manifest_path = summary_path.parent / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if manifest:
        cfg = manifest["configuration"]
        context = (f"W={cfg['warmup']}, stress={cfg.get('jit_stress', 0)}, R={cfg['repeat']}, trials={cfg['trials']}, BLAS={cfg['blas_threads']}, "
                   f"preset={cfg['preset']}; {manifest['system']['machine']}; identity={canonical_hash(manifest['identity'])[:12]}")
    else:
        context = "Configuration/identity manifest unavailable"
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(9, max(3, .24*len(rows)+1.5)), layout="constrained")
    for y, row in enumerate(rows):
        speedup = row.get("speedup")
        if speedup is None:
            ax.text(1, y, "missing", color="#a84832", va="center", fontsize=8)
            continue
        color = "#327b63" if speedup >= 1 else "#b8624c"
        low, high = row["speedup_ci95"]
        ax.plot(speedup, y, "o", color=color, markersize=4)
        if low is not None:
            ax.hlines(y, low, high, color=color, linewidth=1)
    ax.axvline(1, color="#777777", linewidth=.8, linestyle="--")
    ax.set_yticks(range(len(rows)), [row["workload"] for row in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Speedup ({data['comparison']}); 95% CI from independent process pairs")
    ax.set_title(f"{data['experiment']} — {data['measurement_status']}\n"
                 f"requested workloads: {len(rows)}; complete: {data['complete']}; missing CI: unconfirmed\n{context}", fontsize=9)
    ax.grid(axis="x", alpha=.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    cache.cleanup()
    save(output.with_suffix(output.suffix+".json"), dict(summary=str(summary_path.absolute()),
         summary_sha256=digest(summary_path), output_sha256=digest(output), matplotlib_version=matplotlib.__version__,
         manifest_sha256=digest(manifest_path) if manifest else None,
         comparison=data["comparison"], measurement_status=data["measurement_status"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--output", type=Path, required=True, help="Standalone .svg, .pdf or .png figure")
    args = p.parse_args()
    if args.output.suffix.lower() not in (".svg", ".pdf", ".png"):
        p.error("output must be .svg, .pdf or .png")
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
