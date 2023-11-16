"""
This file compares the output and runtime of running normal vs incremental mode
on the history of any arbitrary git repo as a way of performing a sanity check
to make sure incremental mode is working correctly and efficiently.

It does so by first running mypy without incremental mode on the specified range
of commits to find the expected result, then rewinds back to the first commit and
re-runs mypy on the commits with incremental mode enabled to make sure it returns
the same results.

This script will download and test the official mypy repo by default. Running:

    python3 misc/incremental_checker.py last 30

is equivalent to running

    python3 misc/incremental_checker.py last 30 \\
            --repo_url https://github.com/python/mypy.git \\
            --file-path mypy

You can chose to run this script against a specific commit id or against the
last n commits.

To run this script against the last 30 commits:

    python3 misc/incremental_checker.py last 30

To run this script starting from the commit id 2a432b:

    python3 misc/incremental_checker.py commit 2a432b
"""
from __future__ import annotations
import base64
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from typing import Any, Dict, Final
from typing_extensions import TypeAlias as _TypeAlias
CACHE_PATH: Final = '.incremental_checker_cache.json'
MYPY_REPO_URL: Final = 'https://github.com/python/mypy.git'
MYPY_TARGET_FILE: Final = 'mypy'
DAEMON_CMD: Final = ['python3', '-m', 'mypy.dmypy']
JsonDict: _TypeAlias = Dict[str, Any]

def print_offset(text: str, indent_length: int=4) -> None:
    if False:
        for i in range(10):
            print('nop')
    print()
    print(textwrap.indent(text, ' ' * indent_length))
    print()

def delete_folder(folder_path: str) -> None:
    if False:
        return 10
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def execute(command: list[str], fail_on_error: bool=True) -> tuple[str, str, int]:
    if False:
        i = 10
        return i + 15
    proc = subprocess.Popen(' '.join(command), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    (stdout_bytes, stderr_bytes) = proc.communicate()
    (stdout, stderr) = (stdout_bytes.decode('utf-8'), stderr_bytes.decode('utf-8'))
    if fail_on_error and proc.returncode != 0:
        print('EXECUTED COMMAND:', repr(command))
        print('RETURN CODE:', proc.returncode)
        print()
        print('STDOUT:')
        print_offset(stdout)
        print('STDERR:')
        print_offset(stderr)
        raise RuntimeError('Unexpected error from external tool.')
    return (stdout, stderr, proc.returncode)

def ensure_environment_is_ready(mypy_path: str, temp_repo_path: str, mypy_cache_path: str) -> None:
    if False:
        return 10
    os.chdir(mypy_path)
    delete_folder(temp_repo_path)
    delete_folder(mypy_cache_path)

def initialize_repo(repo_url: str, temp_repo_path: str, branch: str) -> None:
    if False:
        i = 10
        return i + 15
    print(f'Cloning repo {repo_url} to {temp_repo_path}')
    execute(['git', 'clone', repo_url, temp_repo_path])
    if branch is not None:
        print(f'Checking out branch {branch}')
        execute(['git', '-C', temp_repo_path, 'checkout', branch])

def get_commits(repo_folder_path: str, commit_range: str) -> list[tuple[str, str]]:
    if False:
        for i in range(10):
            print('nop')
    (raw_data, _stderr, _errcode) = execute(['git', '-C', repo_folder_path, 'log', '--reverse', '--oneline', commit_range])
    output = []
    for line in raw_data.strip().split('\n'):
        (commit_id, _, message) = line.partition(' ')
        output.append((commit_id, message))
    return output

def get_commits_starting_at(repo_folder_path: str, start_commit: str) -> list[tuple[str, str]]:
    if False:
        return 10
    print(f'Fetching commits starting at {start_commit}')
    return get_commits(repo_folder_path, f'{start_commit}^..HEAD')

def get_nth_commit(repo_folder_path: str, n: int) -> tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    print(f'Fetching last {n} commits (or all, if there are fewer commits than n)')
    return get_commits(repo_folder_path, f'-{n}')[0]

def run_mypy(target_file_path: str | None, mypy_cache_path: str, mypy_script: str | None, *, incremental: bool=False, daemon: bool=False, verbose: bool=False) -> tuple[float, str, dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    'Runs mypy against `target_file_path` and returns what mypy prints to stdout as a string.\n\n    If `incremental` is set to True, this function will use store and retrieve all caching data\n    inside `mypy_cache_path`. If `verbose` is set to True, this function will pass the "-v -v"\n    flags to mypy to make it output debugging information.\n\n    If `daemon` is True, we use daemon mode; the daemon must be started and stopped by the caller.\n    '
    stats: dict[str, Any] = {}
    if daemon:
        command = DAEMON_CMD + ['check', '-v']
    else:
        if mypy_script is None:
            command = ['python3', '-m', 'mypy']
        else:
            command = [mypy_script]
        command.extend(['--cache-dir', mypy_cache_path])
        if incremental:
            command.append('--incremental')
        if verbose:
            command.extend(['-v', '-v'])
    if target_file_path is not None:
        command.append(target_file_path)
    start = time.time()
    (output, stderr, _) = execute(command, False)
    if stderr != '':
        output = stderr
    elif daemon:
        (output, stats) = filter_daemon_stats(output)
    runtime = time.time() - start
    return (runtime, output, stats)

def filter_daemon_stats(output: str) -> tuple[str, dict[str, Any]]:
    if False:
        return 10
    stats: dict[str, Any] = {}
    lines = output.splitlines()
    output_lines = []
    for line in lines:
        m = re.match('(\\w+)\\s+:\\s+(.*)', line)
        if m:
            (key, value) = m.groups()
            stats[key] = value
        else:
            output_lines.append(line)
    if output_lines:
        output_lines.append('\n')
    return ('\n'.join(output_lines), stats)

def start_daemon(mypy_cache_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    cmd = DAEMON_CMD + ['restart', '--log-file', './@incr-chk-logs', '--', '--cache-dir', mypy_cache_path]
    execute(cmd)

def stop_daemon() -> None:
    if False:
        for i in range(10):
            print('nop')
    execute(DAEMON_CMD + ['stop'])

def load_cache(incremental_cache_path: str=CACHE_PATH) -> JsonDict:
    if False:
        return 10
    if os.path.exists(incremental_cache_path):
        with open(incremental_cache_path) as stream:
            cache = json.load(stream)
            assert isinstance(cache, dict)
            return cache
    else:
        return {}

def save_cache(cache: JsonDict, incremental_cache_path: str=CACHE_PATH) -> None:
    if False:
        return 10
    with open(incremental_cache_path, 'w') as stream:
        json.dump(cache, stream, indent=2)

def set_expected(commits: list[tuple[str, str]], cache: JsonDict, temp_repo_path: str, target_file_path: str | None, mypy_cache_path: str, mypy_script: str | None) -> None:
    if False:
        while True:
            i = 10
    'Populates the given `cache` with the expected results for all of the given `commits`.\n\n    This function runs mypy on the `target_file_path` inside the `temp_repo_path`, and stores\n    the result in the `cache`.\n\n    If `cache` already contains results for a particular commit, this function will\n    skip evaluating that commit and move on to the next.'
    for (commit_id, message) in commits:
        if commit_id in cache:
            print(f'Skipping commit (already cached): {commit_id}: "{message}"')
        else:
            print(f'Caching expected output for commit {commit_id}: "{message}"')
            execute(['git', '-C', temp_repo_path, 'checkout', commit_id])
            (runtime, output, stats) = run_mypy(target_file_path, mypy_cache_path, mypy_script, incremental=False)
            cache[commit_id] = {'runtime': runtime, 'output': output}
            if output == '':
                print(f'    Clean output ({runtime:.3f} sec)')
            else:
                print(f'    Output ({runtime:.3f} sec)')
                print_offset(output, 8)
    print()

def test_incremental(commits: list[tuple[str, str]], cache: JsonDict, temp_repo_path: str, target_file_path: str | None, mypy_cache_path: str, *, mypy_script: str | None=None, daemon: bool=False, exit_on_error: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Runs incremental mode on all `commits` to verify the output matches the expected output.\n\n    This function runs mypy on the `target_file_path` inside the `temp_repo_path`. The\n    expected output must be stored inside of the given `cache`.\n    '
    print('Note: first commit is evaluated twice to warm up cache')
    commits = [commits[0]] + commits
    overall_stats: dict[str, float] = {}
    for (commit_id, message) in commits:
        print(f'Now testing commit {commit_id}: "{message}"')
        execute(['git', '-C', temp_repo_path, 'checkout', commit_id])
        (runtime, output, stats) = run_mypy(target_file_path, mypy_cache_path, mypy_script, incremental=True, daemon=daemon)
        relevant_stats = combine_stats(overall_stats, stats)
        expected_runtime: float = cache[commit_id]['runtime']
        expected_output: str = cache[commit_id]['output']
        if output != expected_output:
            print('    Output does not match expected result!')
            print(f'    Expected output ({expected_runtime:.3f} sec):')
            print_offset(expected_output, 8)
            print(f'    Actual output: ({runtime:.3f} sec):')
            print_offset(output, 8)
            if exit_on_error:
                break
        else:
            print('    Output matches expected result!')
            print(f'    Incremental: {runtime:.3f} sec')
            print(f'    Original:    {expected_runtime:.3f} sec')
            if relevant_stats:
                print(f'    Stats:       {relevant_stats}')
    if overall_stats:
        print('Overall stats:', overall_stats)

def combine_stats(overall_stats: dict[str, float], new_stats: dict[str, Any]) -> dict[str, float]:
    if False:
        for i in range(10):
            print('nop')
    INTERESTING_KEYS = ['build_time', 'gc_time']
    relevant_stats: dict[str, float] = {}
    for key in INTERESTING_KEYS:
        if key in new_stats:
            value = float(new_stats[key])
            relevant_stats[key] = value
            overall_stats[key] = overall_stats.get(key, 0.0) + value
    return relevant_stats

def cleanup(temp_repo_path: str, mypy_cache_path: str) -> None:
    if False:
        return 10
    delete_folder(temp_repo_path)
    delete_folder(mypy_cache_path)

def test_repo(target_repo_url: str, temp_repo_path: str, target_file_path: str | None, mypy_path: str, incremental_cache_path: str, mypy_cache_path: str, range_type: str, range_start: str, branch: str, params: Namespace) -> None:
    if False:
        return 10
    "Tests incremental mode against the repo specified in `target_repo_url`.\n\n    This algorithm runs in five main stages:\n\n    1.  Clones `target_repo_url` into the `temp_repo_path` folder locally,\n        checking out the specified `branch` if applicable.\n    2.  Examines the repo's history to get the list of all commits to\n        to test incremental mode on.\n    3.  Runs mypy WITHOUT incremental mode against the `target_file_path` (which is\n        assumed to be located inside the `temp_repo_path`), testing each commit\n        discovered in stage two.\n        -   If the results of running mypy WITHOUT incremental mode on a\n            particular commit are already cached inside the `incremental_cache_path`,\n            skip that commit to save time.\n        -   Cache the results after finishing.\n    4.  Rewind back to the first commit, and run mypy WITH incremental mode\n        against the `target_file_path` commit-by-commit, and compare to the expected\n        results found in stage 3.\n    5.  Delete all unnecessary temp files.\n    "
    ensure_environment_is_ready(mypy_path, temp_repo_path, mypy_cache_path)
    initialize_repo(target_repo_url, temp_repo_path, branch)
    if range_type == 'last':
        start_commit = get_nth_commit(temp_repo_path, int(range_start))[0]
    elif range_type == 'commit':
        start_commit = range_start
    else:
        raise RuntimeError(f'Invalid option: {range_type}')
    commits = get_commits_starting_at(temp_repo_path, start_commit)
    if params.limit:
        commits = commits[:params.limit]
    if params.sample:
        seed = params.seed or base64.urlsafe_b64encode(os.urandom(15)).decode('ascii')
        random.seed(seed)
        commits = random.sample(commits, params.sample)
        print('Sampled down to %d commits using random seed %s' % (len(commits), seed))
    cache = load_cache(incremental_cache_path)
    set_expected(commits, cache, temp_repo_path, target_file_path, mypy_cache_path, mypy_script=params.mypy_script)
    save_cache(cache, incremental_cache_path)
    if params.daemon:
        print('Starting daemon')
        start_daemon(mypy_cache_path)
    test_incremental(commits, cache, temp_repo_path, target_file_path, mypy_cache_path, mypy_script=params.mypy_script, daemon=params.daemon, exit_on_error=params.exit_on_error)
    if not params.keep_temporary_files:
        cleanup(temp_repo_path, mypy_cache_path)
    if params.daemon:
        print('Stopping daemon')
        stop_daemon()

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    help_factory: Any = lambda prog: RawDescriptionHelpFormatter(prog=prog, max_help_position=32)
    parser = ArgumentParser(prog='incremental_checker', description=__doc__, formatter_class=help_factory)
    parser.add_argument('range_type', metavar='START_TYPE', choices=['last', 'commit'], help="must be one of 'last' or 'commit'")
    parser.add_argument('range_start', metavar='COMMIT_ID_OR_NUMBER', help='the commit id to start from, or the number of commits to move back (see above)')
    parser.add_argument('-r', '--repo_url', default=MYPY_REPO_URL, metavar='URL', help='the repo to clone and run tests on')
    parser.add_argument('-f', '--file-path', default=MYPY_TARGET_FILE, metavar='FILE', help='the name of the file or directory to typecheck')
    parser.add_argument('-x', '--exit-on-error', action='store_true', help='Exits as soon as an error occurs')
    parser.add_argument('--keep-temporary-files', action='store_true', help='Keep temporary files on exit')
    parser.add_argument('--cache-path', default=CACHE_PATH, metavar='DIR', help='sets a custom location to store cache data')
    parser.add_argument('--branch', default=None, metavar='NAME', help='check out and test a custom branch uses the default if not specified')
    parser.add_argument('--sample', type=int, help='use a random sample of size SAMPLE')
    parser.add_argument('--seed', type=str, help='random seed')
    parser.add_argument('--limit', type=int, help='maximum number of commits to use (default until end)')
    parser.add_argument('--mypy-script', type=str, help='alternate mypy script to run')
    parser.add_argument('--daemon', action='store_true', help='use mypy daemon instead of incremental (highly experimental)')
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    params = parser.parse_args(sys.argv[1:])
    script_path = os.path.abspath(sys.argv[0])
    mypy_path = os.path.abspath(os.path.dirname(os.path.dirname(script_path)))
    temp_repo_path = os.path.abspath(os.path.join(mypy_path, 'tmp_repo'))
    if params.file_path:
        target_file_path = os.path.abspath(os.path.join(temp_repo_path, params.file_path))
    else:
        target_file_path = None
    incremental_cache_path = os.path.abspath(params.cache_path)
    mypy_cache_path = os.path.abspath(os.path.join(mypy_path, 'misc', '.mypy_cache'))
    print(f'Assuming mypy is located at {mypy_path}')
    print(f'Temp repo will be cloned at {temp_repo_path}')
    print(f'Testing file/dir located at {target_file_path}')
    print(f'Using cache data located at {incremental_cache_path}')
    print()
    test_repo(params.repo_url, temp_repo_path, target_file_path, mypy_path, incremental_cache_path, mypy_cache_path, params.range_type, params.range_start, params.branch, params)
if __name__ == '__main__':
    main()