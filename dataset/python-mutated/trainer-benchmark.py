import argparse
import datetime
import io
import itertools
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import fmean
import pandas as pd
import torch
from tqdm import tqdm
import transformers
nan = float('nan')

class Tee:
    """
    A helper class to tee print's output into a file.
    Usage:
    sys.stdout = Tee(filename)
    """

    def __init__(self, filename):
        if False:
            return 10
        self.stdout = sys.stdout
        self.file = open(filename, 'a')

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        return getattr(self.stdout, attr)

    def write(self, msg):
        if False:
            return 10
        self.stdout.write(msg)
        self.file.write(re.sub('^.*\\r', '', msg, 0, re.M))

def get_original_command(max_width=80, full_python_path=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the original command line string that can be replayed nicely and wrapped for 80 char width.\n\n    Args:\n        max_width (`int`, `optional`, defaults to 80):\n            The width to wrap for.\n        full_python_path (`bool`, `optional`, defaults to `False`):\n             Whether to replicate the full path or just the last segment (i.e. `python`).\n    '
    cmd = []
    env_keys = ['CUDA_VISIBLE_DEVICES']
    for key in env_keys:
        val = os.environ.get(key, None)
        if val is not None:
            cmd.append(f'{key}={val}')
    python = sys.executable if full_python_path else sys.executable.split('/')[-1]
    cmd.append(python)
    cmd += list(map(shlex.quote, sys.argv))
    lines = []
    current_line = ''
    while len(cmd) > 0:
        current_line += f'{cmd.pop(0)} '
        if len(cmd) == 0 or len(current_line) + len(cmd[0]) + 1 > max_width - 1:
            lines.append(current_line)
            current_line = ''
    return '\\\n'.join(lines)

def get_base_command(args, output_dir):
    if False:
        i = 10
        return i + 15
    args.base_cmd = re.sub('[\\\\\\n]+', ' ', args.base_cmd)
    args.base_cmd = re.sub('--output_dir\\s+[^\\s]+', '', args.base_cmd)
    args.base_cmd += f' --output_dir {output_dir}'
    args.base_cmd = re.sub('--overwrite_output_dir\\s+', '', args.base_cmd)
    args.base_cmd += ' --overwrite_output_dir'
    return [sys.executable] + shlex.split(args.base_cmd)

def process_run_single(id, cmd, variation, output_dir, target_metric_key, metric_keys, verbose):
    if False:
        return 10
    if 0:
        import random
        from time import sleep
        sleep(0)
        return dict({k: random.uniform(0, 100) for k in metric_keys}, **{target_metric_key: random.choice([nan, 10.31, 100.2, 55.6666, 222.22222222])})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print('STDOUT', result.stdout)
        print('STDERR', result.stderr)
    prefix = variation.replace(' ', '-')
    with open(Path(output_dir) / f'log.{prefix}.stdout.txt', 'w') as f:
        f.write(result.stdout)
    with open(Path(output_dir) / f'log.{prefix}.stderr.txt', 'w') as f:
        f.write(result.stderr)
    if result.returncode != 0:
        if verbose:
            print('failed')
        return {target_metric_key: nan}
    with io.open(f'{output_dir}/all_results.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return {k: v for (k, v) in metrics.items() if k in metric_keys}

def process_run(id, cmd, variation_key, variation, longest_variation_len, target_metric_key, report_metric_keys, repeat_times, output_dir, verbose):
    if False:
        for i in range(10):
            print('nop')
    results = []
    metrics = []
    preamble = f'{id}: {variation:<{longest_variation_len}}'
    outcome = f'{preamble}: '
    metric_keys = set(report_metric_keys + [target_metric_key])
    for i in tqdm(range(repeat_times), desc=preamble, leave=False):
        single_run_metrics = process_run_single(id, cmd, variation, output_dir, target_metric_key, metric_keys, verbose)
        result = single_run_metrics[target_metric_key]
        if not math.isnan(result):
            metrics.append(single_run_metrics)
            results.append(result)
            outcome += '✓'
        else:
            outcome += '✘'
    outcome = f'\x1b[2K\r{outcome}'
    if len(metrics) > 0:
        mean_metrics = {k: fmean([x[k] for x in metrics]) for k in metrics[0].keys()}
        mean_target = round(mean_metrics[target_metric_key], 2)
        results_str = f'{outcome} {mean_target}'
        if len(metrics) > 1:
            results_str += f' {tuple((round(x, 2) for x in results))}'
        print(results_str)
        mean_metrics[variation_key] = variation
        return mean_metrics
    else:
        print(outcome)
        return {variation_key: variation, target_metric_key: nan}

def get_versions():
    if False:
        for i in range(10):
            print('nop')
    properties = torch.cuda.get_device_properties(torch.device('cuda'))
    return f"\nDatetime    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nSoftware:\ntransformers: {transformers.__version__}\ntorch       : {torch.__version__}\ncuda        : {torch.version.cuda}\npython      : {platform.python_version()}\n\nHardware:\n{torch.cuda.device_count()} GPUs      : {properties.name}, {properties.total_memory / 2 ** 30:0.2f}GB\n"

def process_results(results, target_metric_key, report_metric_keys, base_variation, output_dir):
    if False:
        return 10
    df = pd.DataFrame(results)
    variation_key = 'variation'
    diff_key = 'diff_%'
    sentinel_value = nan
    if base_variation is not None and len(df[df[variation_key] == base_variation]):
        sentinel_value = df.loc[df[variation_key] == base_variation][target_metric_key].item()
    if math.isnan(sentinel_value):
        sentinel_value = df.loc[df[target_metric_key] != nan][target_metric_key].min()
    if not math.isnan(sentinel_value):
        df[diff_key] = df.apply(lambda r: round(100 * (r[target_metric_key] - sentinel_value) / sentinel_value) if not math.isnan(r[target_metric_key]) else 0, axis='columns')
    cols = [variation_key, target_metric_key, diff_key, *report_metric_keys]
    df = df.reindex(cols, axis='columns')
    df = df.rename(str.capitalize, axis='columns')
    df_github = df.rename(lambda c: c.replace('_', '<br>'), axis='columns')
    df_console = df.rename(lambda c: c.replace('_', '\n'), axis='columns')
    report = ['', 'Copy between the cut-here-lines and paste as is to github or a forum']
    report += ['----------8<-----------------8<--------']
    report += ['*** Results:', df_github.to_markdown(index=False, floatfmt='.2f')]
    report += ['```']
    report += ['*** Setup:', get_versions()]
    report += ['*** The benchmark command line was:', get_original_command()]
    report += ['```']
    report += ['----------8<-----------------8<--------']
    report += ['*** Results (console):', df_console.to_markdown(index=False, floatfmt='.2f')]
    print('\n\n'.join(report))

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-cmd', default=None, type=str, required=True, help='Base cmd')
    parser.add_argument('--variations', default=None, type=str, nargs='+', required=True, help="Multi-dimensional variations, example: '|--fp16|--bf16' '|--tf32'")
    parser.add_argument('--base-variation', default=None, type=str, help='Baseline variation to compare to. if None the minimal target value will be used to compare against')
    parser.add_argument('--target-metric-key', default=None, type=str, required=True, help='Target metric key in output_dir/all_results.json, e.g., train_samples_per_second')
    parser.add_argument('--report-metric-keys', default='', type=str, help="Report metric keys - other metric keys from output_dir/all_results.json to report, e.g., train_loss. Use a single argument e.g., 'train_loss train_samples")
    parser.add_argument('--repeat-times', default=1, type=int, help='How many times to re-run each variation - an average will be reported')
    parser.add_argument('--output_dir', default='output_benchmark', type=str, help='The output directory where all the benchmark reports will go to and additionally this directory will be used to override --output_dir in the script that is being benchmarked')
    parser.add_argument('--verbose', default=False, action='store_true', help='Whether to show the outputs of each run or just the benchmark progress')
    args = parser.parse_args()
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    base_cmd = get_base_command(args, output_dir)
    dims = [list(map(str.strip, re.split('\\|', x))) for x in args.variations]
    variations = list(map(str.strip, map(' '.join, itertools.product(*dims))))
    longest_variation_len = max((len(x) for x in variations))
    report_metric_keys = args.report_metric_keys.split()
    report_fn = f"benchmark-report-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
    print(f"\nNote: each run's output is also logged under {output_dir}/log.*.std*.txt")
    print(f"and this script's output is also piped into {report_fn}")
    sys.stdout = Tee(report_fn)
    print(f'\n*** Running {len(variations)} benchmarks:')
    print(f"Base command: {' '.join(base_cmd)}")
    variation_key = 'variation'
    results = []
    for (id, variation) in enumerate(tqdm(variations, desc='Total completion: ', leave=False)):
        cmd = base_cmd + variation.split()
        results.append(process_run(id + 1, cmd, variation_key, variation, longest_variation_len, args.target_metric_key, report_metric_keys, args.repeat_times, output_dir, args.verbose))
    process_results(results, args.target_metric_key, report_metric_keys, args.base_variation, output_dir)
if __name__ == '__main__':
    main()