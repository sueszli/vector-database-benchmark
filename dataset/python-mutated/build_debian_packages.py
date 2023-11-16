import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import FrameType
from typing import Collection, Optional, Sequence, Set
DISTS = ('debian:bullseye', 'debian:bookworm', 'debian:sid', 'ubuntu:focal', 'ubuntu:jammy', 'ubuntu:lunar', 'ubuntu:mantic', 'debian:trixie')
DESC = 'Builds .debs for synapse, using a Docker image for the build environment.\n\nBy default, builds for all known distributions, but a list of distributions\ncan be passed on the commandline for debugging.\n'
projdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class Builder:

    def __init__(self, redirect_stdout: bool=False, docker_build_args: Optional[Sequence[str]]=None):
        if False:
            print('Hello World!')
        self.redirect_stdout = redirect_stdout
        self._docker_build_args = tuple(docker_build_args or ())
        self.active_containers: Set[str] = set()
        self._lock = threading.Lock()
        self._failed = False

    def run_build(self, dist: str, skip_tests: bool=False) -> None:
        if False:
            print('Hello World!')
        'Build deb for a single distribution'
        if self._failed:
            print('not building %s due to earlier failure' % (dist,))
            raise Exception('failed')
        try:
            self._inner_build(dist, skip_tests)
        except Exception as e:
            print('build of %s failed: %s' % (dist, e), file=sys.stderr)
            self._failed = True
            raise

    def _inner_build(self, dist: str, skip_tests: bool=False) -> None:
        if False:
            print('Hello World!')
        tag = dist.split(':', 1)[1]
        debsdir = os.path.join(projdir, '../debs')
        os.makedirs(debsdir, exist_ok=True)
        if self.redirect_stdout:
            logfile = os.path.join(debsdir, '%s.buildlog' % (tag,))
            print('building %s: directing output to %s' % (dist, logfile))
            stdout = open(logfile, 'w')
        else:
            stdout = None
        build_args = ('docker', 'build', '--tag', 'dh-venv-builder:' + tag, '--build-arg', 'distro=' + dist, '-f', 'docker/Dockerfile-dhvirtualenv') + self._docker_build_args + ('docker',)
        subprocess.check_call(build_args, stdout=stdout, stderr=subprocess.STDOUT, cwd=projdir)
        container_name = 'synapse_build_' + tag
        with self._lock:
            self.active_containers.add(container_name)
        subprocess.check_call(['docker', 'run', '--rm', '--name', container_name, '--volume=' + projdir + ':/synapse/source:ro', '--volume=' + debsdir + ':/debs', '-e', 'TARGET_USERID=%i' % (os.getuid(),), '-e', 'TARGET_GROUPID=%i' % (os.getgid(),), '-e', 'DEB_BUILD_OPTIONS=%s' % ('nocheck' if skip_tests else ''), 'dh-venv-builder:' + tag], stdout=stdout, stderr=subprocess.STDOUT)
        with self._lock:
            self.active_containers.remove(container_name)
        if stdout is not None:
            stdout.close()
            print('Completed build of %s' % (dist,))

    def kill_containers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            active = list(self.active_containers)
        for c in active:
            print('killing container %s' % (c,))
            subprocess.run(['docker', 'kill', c], stdout=subprocess.DEVNULL)
            with self._lock:
                self.active_containers.remove(c)

def run_builds(builder: Builder, dists: Collection[str], jobs: int=1, skip_tests: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')

    def sig(signum: int, _frame: Optional[FrameType]) -> None:
        if False:
            print('Hello World!')
        print('Caught SIGINT')
        builder.kill_containers()
    signal.signal(signal.SIGINT, sig)
    with ThreadPoolExecutor(max_workers=jobs) as e:
        res = e.map(lambda dist: builder.run_build(dist, skip_tests), dists)
    for _ in res:
        pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('-j', '--jobs', type=int, default=1, help='specify the number of builds to run in parallel')
    parser.add_argument('--no-check', action='store_true', help='skip running tests after building')
    parser.add_argument('--docker-build-arg', action='append', help='specify an argument to pass to docker build')
    parser.add_argument('--show-dists-json', action='store_true', help='instead of building the packages, just list the dists to build for, as a json array')
    parser.add_argument('dist', nargs='*', default=DISTS, help='a list of distributions to build for. Default: %(default)s')
    args = parser.parse_args()
    if args.show_dists_json:
        print(json.dumps(DISTS))
    else:
        builder = Builder(redirect_stdout=args.jobs > 1, docker_build_args=args.docker_build_arg)
        run_builds(builder, dists=args.dist, jobs=args.jobs, skip_tests=args.no_check)