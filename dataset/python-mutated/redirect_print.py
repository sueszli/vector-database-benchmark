"""Redirecting writing

If using a library that can print messages to the console, editing the library
by  replacing `print()` with `tqdm.write()` may not be desirable.
In that case, redirecting `sys.stdout` to `tqdm.write()` is an option.

To redirect `sys.stdout`, create a file-like class that will write
any input string to `tqdm.write()`, and supply the arguments
`file=sys.stdout, dynamic_ncols=True`.

A reusable canonical example is given below:
"""
import contextlib
import sys
from time import sleep
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    if False:
        for i in range(10):
            print('nop')
    orig_out_err = (sys.stdout, sys.stderr)
    try:
        (sys.stdout, sys.stderr) = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    except Exception as exc:
        raise exc
    finally:
        (sys.stdout, sys.stderr) = orig_out_err

def some_fun(i):
    if False:
        i = 10
        return i + 15
    print('Fee, fi, fo,'.split()[i])
with std_out_err_redirect_tqdm() as orig_stdout:
    for i in tqdm(range(3), file=orig_stdout, dynamic_ncols=True):
        some_fun(i)
        sleep(0.5)
print('Done!')