"""
Entrypoint for testing from the top-level namespace.
"""
from __future__ import annotations
import os
import sys
from pandas.compat._optional import import_optional_dependency
PKG = os.path.dirname(os.path.dirname(__file__))

def test(extra_args: list[str] | None=None, run_doctests: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Run the pandas test suite using pytest.\n\n    By default, runs with the marks -m "not slow and not network and not db"\n\n    Parameters\n    ----------\n    extra_args : list[str], default None\n        Extra marks to run the tests.\n    run_doctests : bool, default False\n        Whether to only run the Python and Cython doctests. If you would like to run\n        both doctests/regular tests, just append "--doctest-modules"/"--doctest-cython"\n        to extra_args.\n\n    Examples\n    --------\n    >>> pd.test()  # doctest: +SKIP\n    running: pytest...\n    '
    pytest = import_optional_dependency('pytest')
    import_optional_dependency('hypothesis')
    cmd = ['-m not slow and not network and not db']
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args
    if run_doctests:
        cmd = ['--doctest-modules', '--doctest-cython', f"--ignore={os.path.join(PKG, 'tests')}"]
    cmd += [PKG]
    joined = ' '.join(cmd)
    print(f'running: pytest {joined}')
    sys.exit(pytest.main(cmd))
__all__ = ['test']