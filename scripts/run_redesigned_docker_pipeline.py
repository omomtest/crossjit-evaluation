#!/usr/bin/env python3
"""Build locked HTTPS sources and collect the current paper RQ2–RQ4."""
import shlex
import subprocess

from common import ROOT
from run_docker_pipeline import argument_parser, plan


def redesigned_plan(args):
    # Reuse only snapshot/build planning, with no RQ worker scheduled by it.
    args.rqs = []
    commands, configuration = plan(args)
    configuration["rqs"] = ["rq2", "rq3", "rq4"]
    image = configuration["image"]
    command = ["python3", str(ROOT / "scripts/docker_run.py"), "--image", image,
               "--results", str(args.results.resolve()), "--scripts-dir", str(ROOT / "scripts")]
    if args.cpuset_cpus:
        command += ["--cpuset-cpus", args.cpuset_cpus]
    if args.memory:
        command += ["--memory", args.memory]
    command += ["scripts/run_redesigned_evaluation.py", "--config", "/opt/evaluation/config/experiment.json",
                "--output", "/results/evaluation"]
    for key in ("warmup", "repeat", "trials", "jit_stress", "seed", "timeout", "blas_threads", "memory_budget_mb", "preset"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.cpu_affinity:
        command += ["--cpu-affinity", args.cpu_affinity]
    if args.workloads:
        command += ["--workloads", *args.workloads]
    if args.skip_secondary:
        command += ["--skip-secondary"]
    commands.append(("evaluation-rq2-rq4", command))
    return commands, configuration


def main():
    p = argument_parser()
    # Hide the legacy selector entirely: this entry cannot accept/schedule RQ1.
    action = next(a for a in p._actions if a.dest == "rqs")
    p._remove_action(action)
    for option in action.option_strings:
        del p._option_string_actions[option]
    p.description = __doc__
    p.add_argument("--skip-secondary", action="store_true")
    args = p.parse_args()
    if args.calibrate or args.policy:
        p.error("this frozen-strategy runner uses one shared v32 registry/policy; per-mode calibration is not supported")
    commands, configuration = redesigned_plan(args)
    if args.dry_run:
        for name, command in commands:
            print(name + ": " + shlex.join(command))
        return
    for name, command in commands:
        print(name + ": " + shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
