import subprocess
BEFORE = '\nfrom hypothesis.strategies import complex_numbers, complex_numbers as cn\n\ncomplex_numbers(min_magnitude=None)  # simple call to fix\ncomplex_numbers(min_magnitude=None, max_magnitude=1)  # plus arg after\ncomplex_numbers(allow_infinity=False, min_magnitude=None)  # plus arg before\ncn(min_magnitude=None)  # imported as alias\n'
AFTER = BEFORE.replace('None', '0')
_unchanged = '\ncomplex_numbers(min_magnitude=1)  # value OK\n\nclass Foo:\n    def complex_numbers(self, **kw): pass\n\n    complex_numbers(min_magnitude=None)  # defined in a different scope\n'
BEFORE += _unchanged
AFTER += _unchanged
del _unchanged

def run(command, *, cwd=None, input=None):
    if False:
        for i in range(10):
            print('nop')
    return subprocess.run(command, input=input, capture_output=True, shell=True, text=True, cwd=cwd, encoding='utf-8')

def test_codemod_single_file(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    fname = tmp_path / 'mycode.py'
    fname.write_text(BEFORE, encoding='utf-8')
    result = run('hypothesis codemod mycode.py', cwd=tmp_path)
    assert result.returncode == 0
    assert fname.read_text(encoding='utf-8') == AFTER

def test_codemod_multiple_files(tmp_path):
    if False:
        return 10
    files = [tmp_path / 'mycode1.py', tmp_path / 'mycode2.py']
    for f in files:
        f.write_text(BEFORE, encoding='utf-8')
    result = run('hypothesis codemod mycode1.py mycode2.py', cwd=tmp_path)
    assert result.returncode == 0
    for f in files:
        assert f.read_text(encoding='utf-8') == AFTER

def test_codemod_from_stdin():
    if False:
        while True:
            i = 10
    result = run('hypothesis codemod -', input=BEFORE)
    assert result.returncode == 0
    assert result.stdout.rstrip() == AFTER.rstrip()