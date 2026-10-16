#!/usr/bin/env python3
"""Derive current-paper RQ1–RQ4 results from original records without timing runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

from raw_data import ROOT, entries, unpack
sys.path.insert(0, str(ROOT / 'scripts'))
from common import read_json, save
from reporting import summarize_matrix, coverage_report
from redesign_reporting import composition_report, reach_report
from generator_evidence import captured_descriptions, fixtures
from run_rq4_guard import guard_report
from optimization_properties import attributes, METRICS


def materialize(entry, output, cache):
    source = unpack(entry, cache)
    if output.exists():
        raise ValueError('choose a fresh output directory: '+str(output))
    shutil.copytree(source, output)
    return output


def storage_report(path, reach):
    records = [*fixtures(), *captured_descriptions(reach)]
    rows = [json.loads(line) for line in (path / 'execution.jsonl').read_text().splitlines()]
    for row in rows:
        record = records[row['description']]
        row.update(identity=record['identity'], scope=record['scope'])
        if row['permissions'] and 'original_plan' in record and row['plan'] != record['original_plan']:
            raise ValueError('recorded replay differs from captured runtime plan')
    report = dict(complete=all(r['correct'] for r in rows), records=records, rows=rows,
                  captured_workload_descriptions=len(records)-len(fixtures()),
                  scope='Original generated-planner stdout; eight-double synthetic ownership bridge. Bytes are replay storage, not workload bytes saved.')
    save(path / 'synthesis.json', report)
    return report


def optimization_report(reach, storage, output):
    replay = {}
    for row in storage['rows']:
        replay.setdefault(row['identity'], {})[row['permissions']] = row
    rows = []
    totals = Counter()
    histogram = Counter()
    for window in reach['rows']:
        weights = Counter()
        for signature in window['actual_signatures']:
            attr = attributes(signature, replay)
            hits = signature['hits']
            for metric, count in attr['weights'].items():
                weights[metric] += hits * count
            histogram[signature['node_count']] += hits
        totals.update(weights)
        rows.append(dict(workload=window['workload'], trial=window['trial'], repeat=window['repeat'] if 'repeat' in window else 100,
                         counts=dict(weights)))
    report = dict(complete=reach['complete'], total_counts=dict(totals), hit_histogram=dict(histogram), rows=rows,
        limitations='Traversal fusion and internal-array non-materialization follow executed plan structure weighted by successful HITs. Reuse counts adopted routes; mechanisms overlap.')
    save(output / 'optimization-properties.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/paper-derived')
    p.add_argument('--cache', type=Path)
    args = p.parse_args()
    targets = {'rq1/intel.tar.gz':'rq1/intel', 'rq1/apple-m4.tar.gz':'rq1/apple-m4',
               'rq1/amd-9700x.tar.gz':'rq1/amd-9700x', 'rq2/full-diagnostics.tar.gz':'rq2/full',
               'rq2/adi-fragment.tar.gz':'rq2/adi', 'rq3/scope-ablation.tar.gz':'rq3/scope',
               'rq3/storage-replay.tar.gz':'rq3/storage', 'rq4/callbacks.tar.gz':'rq4/callbacks',
               'rq4/inactive.tar.gz':'rq4/inactive', 'rq2/adi-allocation.tar.gz':'rq2/allocation'}
    paths = {}
    for entry in entries():
        label = targets[entry['path']]
        paths[label] = materialize(entry, args.output / label, args.cache)
    for platform in ('intel','apple-m4','amd-9700x'):
        print('RQ1 '+platform, flush=True)
        summarize_matrix(paths['rq1/'+platform])
    reach = reach_report(paths['rq2/full'])
    coverage_report(paths['rq2/full'])
    for name in ('rq2/adi', 'rq3/scope'):
        print(name, flush=True)
        scope_reach = reach_report(paths[name], ('off','single','on'))
        composition_report(paths[name], scope_reach)
    storage = storage_report(paths['rq3/storage'], reach)
    optimization_report(reach, storage, args.output / 'rq2')
    summarize_matrix(paths['rq4/callbacks'], 'separate_guards', 'on')
    summarize_matrix(paths['rq4/inactive'], 'off', 'on')
    guard_report(paths['rq3/scope'], args.output / 'rq4/guard')
    save(args.output / 'protocol.json', dict(estimator='S_A: ratio of process medians', bootstrap_samples=10000,
         bootstrap_seed=20261002, shared_process_pair_indices=True, raw_data_changed=False))
    print('Derived results: '+str(args.output))


if __name__ == '__main__':
    main()
