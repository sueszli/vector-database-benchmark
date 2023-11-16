import collections
import csv
import os
import pathlib
import subprocess
import conbench.runner
from conbench.machine_info import github_info

def _result_in_seconds(row):
    if False:
        for i in range(10):
            print('nop')
    count = int(row['iteration_count'])
    sample = float(row['sample_measured_value'])
    return sample / count / 10 ** 9

def _parse_benchmark_group(row):
    if False:
        while True:
            i = 10
    parts = row['group'].split(',')
    if len(parts) > 1:
        (suite, name) = (parts[0], ','.join(parts[1:]))
    else:
        (suite, name) = (row['group'], row['group'])
    return (suite.strip(), name.strip())

def _read_results(src_dir):
    if False:
        print('Hello World!')
    results = collections.defaultdict(lambda : collections.defaultdict(list))
    path = pathlib.Path(os.path.join(src_dir, 'target', 'criterion'))
    for path in list(path.glob('**/new/raw.csv')):
        with open(path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                (suite, name) = _parse_benchmark_group(row)
                results[suite][name].append(_result_in_seconds(row))
    return results

def _execute_command(command):
    if False:
        while True:
            i = 10
    try:
        print(command)
        result = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        raise e
    return (result.stdout.decode('utf-8'), result.stderr.decode('utf-8'))

class CriterionBenchmark(conbench.runner.Benchmark):
    external = True

    def run(self, **kwargs):
        if False:
            return 10
        src_dir = os.path.join(os.getcwd(), '..')
        self._cargo_bench(src_dir)
        results = _read_results(src_dir)
        for suite in results:
            self.conbench.mark_new_batch()
            for (name, data) in results[suite].items():
                yield self._record_result(suite, name, data, kwargs)

    def _cargo_bench(self, src_dir):
        if False:
            i = 10
            return i + 15
        os.chdir(src_dir)
        _execute_command(['cargo', 'bench'])

    def _record_result(self, suite, name, data, options):
        if False:
            i = 10
            return i + 15
        tags = {'suite': suite}
        result = {'data': data, 'unit': 's'}
        context = {'benchmark_language': 'Rust'}
        github = github_info()
        return self.conbench.record(result, name, tags=tags, context=context, github=github, options=options)