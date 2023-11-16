"""A module for running the pandas docs (such as the users guide) against our
dataframe implementation.

Run as python -m apache_beam.dataframe.pandas_docs_test [getting_started ...]
"""
import argparse
import contextlib
import io
import multiprocessing
import os
import sys
import time
import urllib.request
import zipfile
from apache_beam.dataframe import doctests
PANDAS_VERSION = '1.1.1'
PANDAS_DIR = os.path.expanduser('~/.apache_beam/cache/pandas-' + PANDAS_VERSION)
PANDAS_DOCS_SOURCE = os.path.join(PANDAS_DIR, 'doc', 'source')

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parallel', type=int, default=0, help='Number of tests to run in parallel. Defaults to 0, meaning the number of cores on the machine.')
    parser.add_argument('docs', nargs='*')
    args = parser.parse_args()
    if not os.path.exists(PANDAS_DIR):
        os.makedirs(os.path.dirname(PANDAS_DIR), exist_ok=True)
        zip = os.path.join(PANDAS_DIR + '.zip')
        if not os.path.exists(zip):
            url = 'https://github.com/pandas-dev/pandas/archive/v%s.zip' % PANDAS_VERSION
            print('Downloading', url)
            with urllib.request.urlopen(url) as fin:
                with open(zip + '.tmp', 'wb') as fout:
                    fout.write(fin.read())
                os.rename(zip + '.tmp', zip)
        print('Extracting', zip)
        with zipfile.ZipFile(zip, 'r') as handle:
            handle.extractall(os.path.dirname(PANDAS_DIR))
    tests = args.docs or ['getting_started', 'user_guide']
    paths = []
    filters = []
    for test in tests:
        if os.path.exists(test):
            paths.append(test)
        else:
            filters.append(test)
    for (root, _, files) in os.walk(PANDAS_DOCS_SOURCE):
        for name in files:
            if name.endswith('.rst'):
                path = os.path.join(root, name)
                if any((filter in path for filter in filters)):
                    paths.append(path)
    parallelism = max(args.parallel or multiprocessing.cpu_count(), len(paths))
    if parallelism > 1:
        pool_map = multiprocessing.pool.Pool(parallelism).imap_unordered
        run_tests = run_tests_capturing_stdout
        paths.sort(key=lambda path: ('enhancingperf' in path, os.path.getsize(path)), reverse=True)
    else:
        pool_map = map
        run_tests = run_tests_streaming_stdout
    running_summary = doctests.Summary()
    for (count, (summary, stdout)) in enumerate(pool_map(run_tests, paths)):
        running_summary += summary
        if stdout:
            print(stdout)
        print(count, '/', len(paths), 'done.')
    print('*' * 72)
    print('Final summary:')
    running_summary.summarize()

def run_tests_capturing_stdout(path):
    if False:
        i = 10
        return i + 15
    with deferred_stdout() as stdout:
        return (run_tests(path), stdout())

def run_tests_streaming_stdout(path):
    if False:
        return 10
    return (run_tests(path), None)

def run_tests(path):
    if False:
        while True:
            i = 10
    start = time.time()
    with open(path) as f:
        rst = f.read()
    res = doctests.test_rst_ipython(rst, path, report=True, wont_implement_ok=['*'], not_implemented_ok=['*'], use_beam=False).summary
    print('Total time for {}: {:.2f} secs'.format(path, time.time() - start))
    return res

@contextlib.contextmanager
def deferred_stdout():
    if False:
        while True:
            i = 10
    captured = io.StringIO()
    (old_stdout, sys.stdout) = (sys.stdout, captured)
    try:
        yield captured.getvalue
    finally:
        sys.stdout = old_stdout
if __name__ == '__main__':
    main()