"""
Runs a command three times:

1.  Run unchanged. This will use the default `fitz` implementation of PyMuPDF.

2.  Run with PYTHONPATH set up to use the `fitz_new` implementation of PyMuPDF.

3.  As 2 but also set PYMUPDF_USE_EXTRA=0 to disable use of C++ optimisations.

Example usage:

    ./PyMuPDF/tests/run_compound.py python -m pytest -s PyMuPDF
"""
import shlex
import os
import platform
import subprocess
import sys
import textwrap

def log(text):
    if False:
        print('Hello World!')
    print('#' * 40)
    print(f'{__file__} python-{platform.python_version()}: {text}')
    print('#' * 40)
    sys.stdout.flush()

def main():
    if False:
        while True:
            i = 10
    args = sys.argv[1:]
    log(f'Running using fitz: {shlex.join(args)}')
    e1 = subprocess.run(args, shell=0, check=0).returncode
    d = os.path.abspath(f'{__file__}/../resources')
    with open(f'{d}/fitz.py', 'w') as f:
        f.write(textwrap.dedent(f"\n                #import sys\n                #print(f'{{__file__}}: {{sys.path=}}')\n                #print(f'{{__file__}}: Importing * from fitz_new')\n                #sys.stdout.flush()\n                from fitz_new import *\n                "))
    env = os.environ.copy()
    pp = env.get('PYTHONPATH')
    pp = d if pp is None else f'{d}:{pp}'
    env['PYTHONPATH'] = pp
    log(f'Running using fitz_new, PYTHONPATH={pp}: {shlex.join(args)}')
    e2 = subprocess.run(args, shell=0, check=0, env=env).returncode
    env['PYMUPDF_USE_EXTRA'] = '0'
    log(f'Running using fitz_new without optimisations, PYTHONPATH={pp}: {shlex.join(args)}')
    e3 = subprocess.run(args, shell=0, check=0, env=env).returncode
    log(f'e1={e1!r} e2={e2!r} e3={e3!r}')
    if e1 or e2 or e3:
        raise Exception(f'Failure: e1={e1!r} e2={e2!r} e3={e3!r}')
if __name__ == '__main__':
    main()