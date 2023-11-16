"""
Render benchmark graphs from data in GCS.

To use this script, you must be authenticated with GCS,
see https://cloud.google.com/docs/authentication/client-libraries for more information.

Install dependencies:
    google-cloud-storage==2.9.0

Use the script:
    python3 scripts/ci/render_bench.py --help

    python3 scripts/ci/render_bench.py       crates       --num-days 30       --output ./benchmarks

    python3 scripts/ci/render_bench.py       sizes       --num-days $((30*6))       --output gs://rerun-builds/graphs
"""
from __future__ import annotations
import argparse
import json
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from subprocess import run
from typing import Callable, Dict, Generator, List
from google.cloud import storage
SCRIPT_PATH = os.path.dirname(os.path.relpath(__file__))
DATE_FORMAT = '%Y-%m-%d'
ESCAPED_DATE_FORMAT = DATE_FORMAT.replace('%', '%%')

def non_empty_lines(s: str) -> Generator[str, None]:
    if False:
        return 10
    for line in s.splitlines():
        if len(line.strip()) == 0:
            continue
        yield line

@dataclass
class CommitWithDate:
    date: datetime
    commit: str

def get_commits(after: datetime) -> list[CommitWithDate]:
    if False:
        while True:
            i = 10
    args = ['git', 'log']
    args += [f'--after="{after.year}-{after.month}-{after.day} 00:00:00"']
    args += ['--format=%cd;%H', '--date=iso-strict']
    log = run(args, check=True, capture_output=True, text=True).stdout.strip().splitlines()
    commits = (commit.split(';', 1) for commit in log)
    return [CommitWithDate(date=datetime.fromisoformat(date).astimezone(timezone.utc), commit=commit) for (date, commit) in commits]

@dataclass
class Measurement:
    name: str
    value: float
    unit: str

@dataclass
class BenchmarkEntry:
    name: str
    value: float
    unit: str
    date: datetime
    commit: str
    is_duplicate: bool = False

    def duplicate(self, date: datetime) -> BenchmarkEntry:
        if False:
            i = 10
            return i + 15
        return BenchmarkEntry(name=self.name, value=self.value, unit=self.unit, date=date, commit=self.commit, is_duplicate=True)
Benchmarks = Dict[str, List[BenchmarkEntry]]
FORMAT_BENCHER_RE = re.compile('test\\s+(\\S+).*bench:\\s+(\\d+)\\s+ns\\/iter')

def parse_bencher_line(data: str) -> Measurement:
    if False:
        i = 10
        return i + 15
    (name, ns_iter) = FORMAT_BENCHER_RE.match(data).groups()
    return Measurement(name, float(ns_iter), 'ns/iter')

def parse_bencher_text(data: str) -> list[Measurement]:
    if False:
        print('Hello World!')
    return [parse_bencher_line(line) for line in non_empty_lines(data)]

def parse_sizes_json(data: str) -> list[Measurement]:
    if False:
        for i in range(10):
            print('nop')
    return [Measurement(name=entry['name'], value=float(entry['value']), unit=entry['unit']) for entry in json.loads(data)]
Blobs = Dict[str, storage.Blob]

def fetch_blobs(gcs: storage.Client, bucket: str, path_prefix: str) -> Blobs:
    if False:
        i = 10
        return i + 15
    blobs = gcs.bucket(bucket).list_blobs(prefix=path_prefix)
    return {blob.name: blob for blob in blobs}

def collect_benchmark_data(commits: list[CommitWithDate], bucket: Blobs, short_sha_to_path: Callable[[str], str], parser: Callable[[str], list[Measurement]]) -> Benchmarks:
    if False:
        print('Hello World!')
    benchmarks: Benchmarks = {}

    def insert(entry: BenchmarkEntry) -> None:
        if False:
            for i in range(10):
                print('nop')
        if entry.name not in benchmarks:
            benchmarks[entry.name] = []
        benchmarks[entry.name].append(entry)
    previous_entry: BenchmarkEntry | None = None
    for v in reversed(commits):
        short_sha = v.commit[0:7]
        path = short_sha_to_path(short_sha)
        if path not in bucket:
            if previous_entry is not None:
                insert(previous_entry.duplicate(date=v.date))
            continue
        for measurement in parser(bucket[path].download_as_text()):
            entry = BenchmarkEntry(name=measurement.name, value=measurement.value, unit=measurement.unit, date=v.date, commit=v.commit)
            previous_entry = entry
            insert(entry)
    return benchmarks

def get_crates_benchmark_data(gcs: storage.Client, commits: list[CommitWithDate]) -> Benchmarks:
    if False:
        return 10
    print('Fetching benchmark data for "Rust Crates"…')
    return collect_benchmark_data(commits, bucket=fetch_blobs(gcs, 'rerun-builds', 'benches'), short_sha_to_path=lambda short_sha: f'benches/{short_sha}', parser=parse_bencher_text)

def get_size_benchmark_data(gcs: storage.Client, commits: list[CommitWithDate]) -> Benchmarks:
    if False:
        while True:
            i = 10
    print('Fetching benchmark data for "Sizes"…')
    return collect_benchmark_data(commits, bucket=fetch_blobs(gcs, 'rerun-builds', 'sizes/commit'), short_sha_to_path=lambda short_sha: f'sizes/commit/{short_sha}/data.json', parser=parse_sizes_json)
BYTE_UNITS = ['b', 'kb', 'kib', 'mb', 'mib', 'gb', 'gib', 'tb', 'tib']
VALID_CONVERSIONS = {'ns/iter': ['ns/iter']}
for unit in BYTE_UNITS:
    VALID_CONVERSIONS[unit] = BYTE_UNITS
UNITS = {'b': 1, 'kb': 1000, 'kib': 1024, 'mb': 1000, 'mib': 1024 * 1024, 'gb': 1000, 'gib': 1024 * 1024, 'tb': 1000, 'tib': 1024 * 1024, 'ns/iter': 1}

def convert(base_unit: str, unit: str, value: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Convert `value` from `base_unit` to `unit`.'
    base_unit = base_unit.lower()
    unit = unit.lower()
    if unit not in VALID_CONVERSIONS[base_unit]:
        raise Exception(f'invalid conversion from {base_unit} to {unit}')
    return value / UNITS[unit] * UNITS[base_unit]

def min_and_max(data: list[float]) -> (float, float):
    if False:
        print('Hello World!')
    min_value = float('inf')
    max_value = float('-inf')
    for value in data:
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value
    return (min_value, max_value)

def render_html(title: str, benchmarks: Benchmarks) -> str:
    if False:
        for i in range(10):
            print('nop')
    print(f'Rendering "{title}" benchmark…')

    def label(entry: BenchmarkEntry) -> str:
        if False:
            for i in range(10):
                print('nop')
        date = entry.date.strftime('%Y-%m-%d')
        if entry.is_duplicate:
            return f'{date}'
        else:
            return f'{entry.commit[0:7]} {date}'
    chartjs = {}
    for (name, benchmark) in benchmarks.items():
        if len(benchmark) == 0:
            chartjs[name] = None
        labels = [label(entry) for entry in benchmark]
        base_unit = benchmark[-1].unit
        data = [convert(base_unit, entry.unit, entry.value) for entry in benchmark]
        (min_value, max_value) = min_and_max(data)
        y_scale = {'min': max(0, min_value - min_value / 3), 'max': max_value + max_value / 3}
        chartjs[name] = {'y_scale': y_scale, 'unit': base_unit, 'labels': labels, 'data': data}
    with open(os.path.join(SCRIPT_PATH, 'templates/benchmark.html')) as template_file:
        html = template_file.read()
        html = html.replace('%%TITLE%%', title)
        html = html.replace('"%%CHARTS%%"', json.dumps(json.dumps(chartjs)))
    return html

class Target(Enum):
    CRATES = 'crates'
    SIZE = 'sizes'
    ALL = 'all'

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.value

    def includes(self, other: Target) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self is Target.ALL or self is other

    def render(self, gcs: storage.Client, after: datetime) -> dict[str, str]:
        if False:
            while True:
                i = 10
        commits = get_commits(after)
        print('commits', commits)
        out: dict[str, str] = {}
        if self.includes(Target.CRATES):
            data = get_crates_benchmark_data(gcs, commits)
            out[str(Target.CRATES)] = render_html('Rust Crates', data)
        if self.includes(Target.SIZE):
            data = get_size_benchmark_data(gcs, commits)
            out[str(Target.SIZE)] = render_html('Sizes', data)
        return out

def date_type(v: str) -> datetime:
    if False:
        return 10
    try:
        return datetime.strptime(v, DATE_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Date must be in {DATE_FORMAT} format')

class Output(Enum):
    STDOUT = 'stdout'
    GCS = 'gcs'
    FILE = 'file'

    def parse(o: str) -> Output:
        if False:
            while True:
                i = 10
        if o == '-':
            return Output.STDOUT
        if o.startswith('gs://'):
            return Output.GCS
        return Output.FILE

@dataclass
class GcsPath:
    bucket: str
    blob: str

def parse_gcs_path(path: str) -> GcsPath:
    if False:
        for i in range(10):
            print('nop')
    if not path.startswith('gs://'):
        raise ValueError(f'invalid gcs path: {path}')
    path = path.lstrip('gs://')
    try:
        (bucket, blob) = path.split('/', 1)
        return GcsPath(bucket, blob.rstrip('/'))
    except ValueError:
        raise ValueError(f'invalid gcs path: {path}')

def main() -> None:
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Render benchmarks from data in GCS', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('target', type=Target, choices=list(Target), help='Which benchmark to render')
    _30_days_ago = datetime.today() - timedelta(days=30)
    parser.add_argument('--after', type=date_type, help=f'The last date to fetch, in {ESCAPED_DATE_FORMAT} format. Default: today ({_30_days_ago.strftime(DATE_FORMAT)})')
    parser.add_argument('-o', '--output', type=str, required=True, help=textwrap.dedent("        Directory to save to. Accepts any of:\n          - '-' for stdout\n          - 'gs://' prefix for GCS\n          - local path\n        "))
    args = parser.parse_args()
    target: Target = args.target
    after: datetime = args.after or _30_days_ago
    output: str = args.output
    output_kind: Output = Output.parse(output)
    print({'target': str(target), 'after': str(after), 'output': output, 'output_kind': str(output_kind)})
    gcs = storage.Client()
    benchmarks = target.render(gcs, after)
    print('benchmarks', benchmarks)
    if output_kind is Output.STDOUT:
        for benchmark in benchmarks.values():
            print(benchmark)
    elif output_kind is Output.GCS:
        path = parse_gcs_path(output)
        print(f'Uploading to {path.bucket}/{path.blob}…')
        bucket = gcs.bucket(path.bucket)
        for (name, benchmark) in benchmarks.items():
            blob = bucket.blob(f'{path.blob}/{name}.html')
            blob.cache_control = 'no-cache, max-age=0'
            blob.upload_from_string(benchmark, content_type='text/html')
    elif output_kind is Output.FILE:
        dir = Path(output)
        dir.mkdir(parents=True, exist_ok=True)
        for (name, benchmark) in benchmarks.items():
            (dir / f'{name}.html').write_text(benchmark)
if __name__ == '__main__':
    main()