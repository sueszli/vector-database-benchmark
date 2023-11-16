import argparse
import datetime
import functools
import glob
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing.managers import SyncManager
import os
from os.path import basename
from os.path import dirname
from os.path import exists
from os.path import join
from os.path import realpath
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
from threading import Lock
import time
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
CERTBOT_DIR = dirname(dirname(dirname(realpath(__file__))))
PLUGINS = [basename(path) for path in glob.glob(join(CERTBOT_DIR, 'certbot-dns-*'))]
print = functools.partial(print, flush=True)

def _snap_log_name(target: str, arch: str):
    if False:
        return 10
    return f'{target}_{arch}.txt'

def _execute_build(target: str, archs: Set[str], status: Dict[str, Dict[str, str]], workspace: str, output_lock: Lock) -> Tuple[int, List[str]]:
    if False:
        print('Hello World!')
    random_string = ''.join((random.choice(string.ascii_lowercase + string.digits) for _ in range(32)))
    build_id = f'snapcraft-{target}-{random_string}'
    with tempfile.TemporaryDirectory() as tempdir:
        environ = os.environ.copy()
        environ['XDG_CACHE_HOME'] = tempdir
        process = subprocess.Popen(['snapcraft', 'remote-build', '--launchpad-accept-public-upload', '--build-for', ','.join(archs), '--build-id', build_id], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=environ, cwd=workspace)
    killed = False
    process_output: List[str] = []
    for line in process.stdout:
        process_output.append(line)
        _extract_state(target, line, status)
        if not killed and any((state for state in status[target].values() if state == 'Chroot problem')):
            with output_lock:
                print(f"Chroot problem encountered for build {target} for {','.join(archs)}.\nLaunchpad seems to be unable to recover from this state so we are terminating the build.")
            process.kill()
            killed = True
    process_state = process.wait()
    return (process_state, process_output)

def _build_snap(target: str, archs: Set[str], status: Dict[str, Dict[str, str]], running: Dict[str, bool], output_lock: Lock) -> bool:
    if False:
        i = 10
        return i + 15
    if target == 'certbot':
        workspace = CERTBOT_DIR
    else:
        workspace = join(CERTBOT_DIR, target)
    build_success = False
    retry = 3
    while retry:
        status[target] = {arch: '...' for arch in archs}
        (exit_code, process_output) = _execute_build(target, archs, status, workspace, output_lock)
        with output_lock:
            print(f"Build {target} for {','.join(archs)} (attempt {4 - retry}/3) ended with exit code {exit_code}.")
            failed_archs = [arch for arch in archs if status[target][arch] != 'Successfully built']
            dump_output = exit_code != 0 or failed_archs
            if exit_code == 0 and (not failed_archs):
                snaps_list = glob.glob(join(workspace, '*.snap'))
                if not len(snaps_list) == len(archs):
                    print(f'Some of the expected snaps for a successful build are missing (current list: {snaps_list}).')
                    dump_output = True
                else:
                    build_success = True
                    break
            if dump_output:
                print(f'Dumping snapcraft remote-build output build for {target}:')
                print('\n'.join(process_output))
                _dump_failed_build_logs(target, archs, status, workspace)
        retry = retry - 1
    running[target] = False
    return build_success

def _extract_state(project: str, output: str, status: Dict[str, Dict[str, str]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    state = status[project]
    if 'Sending build data to Launchpad...' in output:
        for arch in state.keys():
            state[arch] = 'Sending build data'
    match = re.match('^.*arch=(\\w+)\\s+state=([\\w ]+).*$', output)
    if match:
        arch = match.group(1)
        state[arch] = match.group(2)
    status[project] = state

def _dump_status_helper(archs: Set[str], status: Dict[str, Dict[str, str]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    headers = ['project', *archs]
    print(''.join((f'| {item:<25}' for item in headers)))
    print(f"|{'-' * 26}" * len(headers))
    for (project, states) in sorted(status.items()):
        print(''.join((f'| {item:<25}' for item in [project, *[states[arch] for arch in archs]])))
    print(f"|{'-' * 26}" * len(headers))
    print()

def _dump_status(archs: Set[str], status: Dict[str, Dict[str, str]], running: Dict[str, bool], output_lock: Lock) -> None:
    if False:
        i = 10
        return i + 15
    while any(running.values()):
        with output_lock:
            print(f'Remote build status at {datetime.datetime.now()}')
            _dump_status_helper(archs, status)
        time.sleep(10)

def _dump_failed_build_logs(target: str, archs: Set[str], status: Dict[str, Dict[str, str]], workspace: str) -> None:
    if False:
        while True:
            i = 10
    for arch in archs:
        result = status[target][arch]
        if result != 'Successfully built':
            failures = True
            build_output_name = _snap_log_name(target, arch)
            build_output_path = join(workspace, build_output_name)
            if not exists(build_output_path):
                build_output = f'No output has been dumped by snapcraft remote-build.'
            else:
                with open(build_output_path) as file_h:
                    build_output = file_h.read()
            print(f'Output for failed build target={target} arch={arch}')
            print('-------------------------------------------')
            print(build_output)
            print('-------------------------------------------')
            print()

def _dump_results(archs: Set[str], status: Dict[str, Dict[str, str]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    print(f'Results for remote build finished at {datetime.datetime.now()}')
    _dump_status_helper(archs, status)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='+', choices=['ALL', 'DNS_PLUGINS', 'certbot', *PLUGINS], help='the list of snaps to build')
    parser.add_argument('--archs', nargs='+', choices=['amd64', 'arm64', 'armhf'], default=['amd64'], help='the architectures for which snaps are built')
    parser.add_argument('--timeout', type=int, default=None, help='build process will fail after the provided timeout (in seconds)')
    args = parser.parse_args()
    archs = set(args.archs)
    targets = set(args.targets)
    if 'ALL' in targets:
        targets.remove('ALL')
        targets.update(['certbot', 'DNS_PLUGINS'])
    if 'DNS_PLUGINS' in targets:
        targets.remove('DNS_PLUGINS')
        targets.update(PLUGINS)
    if targets != {'certbot'}:
        subprocess.run(['tools/snap/generate_dnsplugins_all.sh'], check=True, cwd=CERTBOT_DIR)
    print('Start remote snap builds...')
    print(f" - archs: {', '.join(archs)}")
    print(f" - projects: {', '.join(sorted(targets))}")
    print()
    manager: SyncManager = Manager()
    pool = Pool(processes=len(targets))
    with manager, pool:
        status: Dict[str, Dict[str, str]] = manager.dict()
        running = manager.dict({target: True for target in targets})
        output_lock = manager.Lock()
        async_results = [pool.apply_async(_build_snap, (target, archs, status, running, output_lock)) for target in targets]
        process = Process(target=_dump_status, args=(archs, status, running, output_lock))
        process.start()
        try:
            process.join(args.timeout)
            if process.is_alive():
                for target in targets:
                    if target == 'certbot':
                        workspace = CERTBOT_DIR
                    else:
                        workspace = join(CERTBOT_DIR, target)
                    _dump_failed_build_logs(target, archs, status, workspace)
                raise ValueError(f'Timeout out reached ({args.timeout} seconds) during the build!')
            build_success = True
            for async_result in async_results:
                if not async_result.get():
                    build_success = False
            _dump_results(archs, status)
            if build_success:
                print('All builds succeeded.')
            else:
                print('Some builds failed.')
                raise ValueError('There were failures during the build!')
        finally:
            process.terminate()
if __name__ == '__main__':
    sys.exit(main())