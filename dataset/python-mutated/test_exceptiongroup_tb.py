import unittest
import re
from IPython.utils.capture import capture_output
import sys
import pytest
from tempfile import TemporaryDirectory
from IPython.testing import tools as tt

def _exceptiongroup_common(outer_chain: str, inner_chain: str, native: bool) -> None:
    if False:
        while True:
            i = 10
    pre_raise = 'exceptiongroup.' if not native else ''
    pre_catch = pre_raise if sys.version_info < (3, 11) else ''
    filestr = f'''\n    {('import exceptiongroup' if not native else '')}\n    import pytest\n\n    def f(): raise ValueError("From f()")\n    def g(): raise BaseException("From g()")\n\n    def inner(inner_chain):\n        excs = []\n        for callback in [f, g]:\n            try:\n                callback()\n            except BaseException as err:\n                excs.append(err)\n        if excs:\n            if inner_chain == "none":\n                raise {pre_raise}BaseExceptionGroup("Oops", excs)\n            try:\n                raise SyntaxError()\n            except SyntaxError as e:\n                if inner_chain == "from":\n                    raise {pre_raise}BaseExceptionGroup("Oops", excs) from e\n                else:\n                    raise {pre_raise}BaseExceptionGroup("Oops", excs)\n\n    def outer(outer_chain, inner_chain):\n        try:\n            inner(inner_chain)\n        except {pre_catch}BaseExceptionGroup as e:\n            if outer_chain == "none":\n                raise\n            if outer_chain == "from":\n                raise IndexError() from e\n            else:\n                raise IndexError\n\n\n    outer("{outer_chain}", "{inner_chain}")\n    '''
    with capture_output() as cap:
        ip.run_cell(filestr)
    match_lines = []
    if inner_chain == 'another':
        match_lines += ['During handling of the above exception, another exception occurred:']
    elif inner_chain == 'from':
        match_lines += ['The above exception was the direct cause of the following exception:']
    match_lines += ['  + Exception Group Traceback (most recent call last):', f'  | {pre_catch}BaseExceptionGroup: Oops (2 sub-exceptions)', '    | ValueError: From f()', '    | BaseException: From g()']
    if outer_chain == 'another':
        match_lines += ['During handling of the above exception, another exception occurred:', 'IndexError']
    elif outer_chain == 'from':
        match_lines += ['The above exception was the direct cause of the following exception:', 'IndexError']
    error_lines = cap.stderr.split('\n')
    err_index = match_index = 0
    for expected in match_lines:
        for (i, actual) in enumerate(error_lines):
            if actual == expected:
                error_lines = error_lines[i + 1:]
                break
        else:
            assert False, f'{expected} not found in cap.stderr'

@pytest.mark.skipif(sys.version_info < (3, 11), reason='Native ExceptionGroup not implemented')
@pytest.mark.parametrize('outer_chain', ['none', 'from', 'another'])
@pytest.mark.parametrize('inner_chain', ['none', 'from', 'another'])
def test_native_exceptiongroup(outer_chain, inner_chain) -> None:
    if False:
        print('Hello World!')
    _exceptiongroup_common(outer_chain, inner_chain, native=True)

@pytest.mark.parametrize('outer_chain', ['none', 'from', 'another'])
@pytest.mark.parametrize('inner_chain', ['none', 'from', 'another'])
def test_native_exceptiongroup(outer_chain, inner_chain) -> None:
    if False:
        return 10
    pytest.importorskip('exceptiongroup')
    _exceptiongroup_common(outer_chain, inner_chain, native=False)