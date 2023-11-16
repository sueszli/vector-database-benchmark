from bigdl.nano.utils.common import Backend
from bigdl.nano.utils.common import invalidInputError
import os
from tempfile import TemporaryDirectory
from typing import Any

class HorovodBackend(Backend):

    def setup(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def shutdown(self) -> None:
        if False:
            return 10
        pass

    def run(self, target, args=..., nprocs=1, envs=None) -> Any:
        if False:
            while True:
                i = 10
        if envs is not None:
            if isinstance(envs, list):
                invalidInputError(nprocs == len(envs), 'envs must have the same length with nprocs')
            elif isinstance(envs, dict):
                envs = [envs] * nprocs
            else:
                invalidInputError(False, 'envs must be a dict or a list of dict')
        return self.run_subprocess(target, args=args, nprocs=nprocs, envs=envs)

    def run_subprocess(self, target, args=..., nprocs=1, envs=None) -> Any:
        if False:
            return 10
        from bigdl.nano.utils.common import SafePickle
        import subprocess
        import sys
        with TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'args.pkl'), 'wb') as f:
                SafePickle.dump((envs,) + args, f)
            with open(os.path.join(temp_dir, 'target.pkl'), 'wb') as f:
                SafePickle.dump(target, f)
            cwd_path = os.path.split(os.path.realpath(__file__))[0]
            invalidInputError(os.path.isdir(cwd_path), 'cwd_path should be a valid directory path.')
            invalidInputError(os.path.isdir(temp_dir), 'temp_dir should be a valid directory path.')
            invalidInputError(nprocs > 0, 'nprocs must be greater than 0')
            p = subprocess.Popen(['horovodrun', '-np', str(nprocs), '-H', f'localhost:{nprocs}', sys.executable, f'{cwd_path}/horovod_worker.py', temp_dir])
            p.wait()
            if p.returncode != 0:
                invalidInputError(False, 'horovodrun failed')
            results = []
            for i in range(nprocs):
                with open(os.path.join(temp_dir, f'history_{i}'), 'rb') as f:
                    results.append(SafePickle.load(f))
        return results