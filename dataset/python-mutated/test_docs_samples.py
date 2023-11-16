"""
Test sample scripts in docs/samples/.
"""
import glob
import os
import pytest
import runpy
root = os.path.abspath(f'{__file__}/../..')
samples = []
for p in glob.glob(f'{root}/docs/samples/*.py'):
    if os.path.basename(p) in ('make-bold.py', 'multiprocess-gui.py', 'multiprocess-render.py', 'text-lister.py'):
        print(f'Not testing: {p}')
    else:
        samples.append(p)

def _test_all():
    if False:
        while True:
            i = 10
    import subprocess
    import sys
    e = 0
    for sample in samples:
        print(f'Running: {sample}')
        try:
            if 0:
                print(f'os.environ is:')
                for (n, v) in os.environ.items():
                    print(f'    {n}: {v!r}')
                command = f'{sys.executable} {sample}'
                print(f'command is: {command!r}')
                sys.stdout.flush()
                subprocess.check_call(command, shell=1, text=1)
            else:
                runpy.run_path(sample)
        except Exception:
            print(f'Failed: {sample}')
            e += 1
    if e:
        raise Exception(f'Errors: {e}')

@pytest.mark.parametrize('sample', samples)
def test_docs_samples(sample):
    if False:
        while True:
            i = 10
    runpy.run_path(sample)