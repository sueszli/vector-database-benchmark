import argparse
import contextlib
import os
import platform
import signal
import struct
import subprocess
import sys
from typing import Dict
import scalene

class ScalenePreload:

    @staticmethod
    def get_preload_environ(args: argparse.Namespace) -> Dict[str, str]:
        if False:
            print('Hello World!')
        env = {'SCALENE_ALLOCATION_SAMPLING_WINDOW': str(args.allocation_sampling_window)}
        if sys.platform == 'darwin':
            if args.memory:
                env['DYLD_INSERT_LIBRARIES'] = os.path.join(scalene.__path__[0], 'libscalene.dylib')
                if 'PYTHONMALLOC' in env:
                    del env['PYTHONMALLOC']
            env['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        elif sys.platform == 'linux':
            if args.memory:
                new_ld_preload = os.path.join(scalene.__path__[0].replace(' ', '\\ '), 'libscalene.so')
                if 'LD_PRELOAD' in env:
                    old_ld_preload = env['LD_PRELOAD']
                    env['LD_PRELOAD'] = new_ld_preload + ':' + old_ld_preload
                else:
                    env['LD_PRELOAD'] = new_ld_preload
                if 'PYTHONMALLOC' in env:
                    del env['PYTHONMALLOC']
        elif sys.platform == 'win32':
            args.memory = False
        return env

    @staticmethod
    def setup_preload(args: argparse.Namespace) -> bool:
        if False:
            while True:
                i = 10
        '\n        Ensures that Scalene runs with libscalene preloaded, if necessary,\n        as well as any other required environment variables.\n        Returns true iff we had to run another process.\n        '
        if args.memory and (platform.machine() not in ['x86_64', 'AMD64', 'arm64', 'aarch64'] or struct.calcsize('P') != 8):
            args.memory = False
            print('Scalene warning: currently only 64-bit x86-64 and ARM platforms are supported for memory and copy profiling.')
        with contextlib.suppress(Exception):
            from IPython import get_ipython
            if get_ipython():
                sys.exit = Scalene.clean_exit
                sys._exit = Scalene.clean_exit
        req_env = ScalenePreload.get_preload_environ(args)
        if any((k_v not in os.environ.items() for k_v in req_env.items())):
            os.environ.update(req_env)
            new_args = [sys.executable, '-m', 'scalene'] + sys.argv[1:]
            result = subprocess.Popen(new_args, close_fds=True, shell=False)
            with contextlib.suppress(Exception):
                if os.getpgrp() != os.tcgetpgrp(sys.stdout.fileno()):
                    print(f'Scalene now profiling process {result.pid}')
                    print(f'  to disable profiling: python3 -m scalene.profile --off --pid {result.pid}')
                    print(f'  to resume profiling:  python3 -m scalene.profile --on  --pid {result.pid}')
            try:
                result.wait()
            except KeyboardInterrupt:
                result.returncode = 0
            if result.returncode < 0:
                print('Scalene error: received signal', signal.Signals(-result.returncode).name)
            sys.exit(result.returncode)
            return True
        return False