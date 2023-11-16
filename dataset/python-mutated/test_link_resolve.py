import os.path
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from string import ascii_lowercase
from _pytest.pytester import Pytester

@contextmanager
def subst_path_windows(filepath: Path):
    if False:
        i = 10
        return i + 15
    for c in ascii_lowercase[7:]:
        c += ':'
        if not os.path.exists(c):
            drive = c
            break
    else:
        raise AssertionError('Unable to find suitable drive letter for subst.')
    directory = filepath.parent
    basename = filepath.name
    args = ['subst', drive, str(directory)]
    subprocess.check_call(args)
    assert os.path.exists(drive)
    try:
        filename = Path(drive, os.sep, basename)
        yield filename
    finally:
        args = ['subst', '/D', drive]
        subprocess.check_call(args)

@contextmanager
def subst_path_linux(filepath: Path):
    if False:
        print('Hello World!')
    directory = filepath.parent
    basename = filepath.name
    target = directory / '..' / 'sub2'
    os.symlink(str(directory), str(target), target_is_directory=True)
    try:
        filename = target / basename
        yield filename
    finally:
        pass

def test_link_resolve(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'See: https://github.com/pytest-dev/pytest/issues/5965.'
    sub1 = pytester.mkpydir('sub1')
    p = sub1.joinpath('test_foo.py')
    p.write_text(textwrap.dedent('\n        import pytest\n        def test_foo():\n            raise AssertionError()\n        '), encoding='utf-8')
    subst = subst_path_linux
    if sys.platform == 'win32':
        subst = subst_path_windows
    with subst(p) as subst_p:
        result = pytester.runpytest(str(subst_p), '-v')
        stdout = result.stdout.str()
        assert 'sub1/test_foo.py' not in stdout
        expect = f'*{subst_p}*' if sys.platform == 'win32' else '*sub2/test_foo.py*'
        result.stdout.fnmatch_lines([expect])