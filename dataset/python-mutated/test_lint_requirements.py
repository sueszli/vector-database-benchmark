import pytest
from tools import lint_requirements

def test_ok(tmp_path):
    if False:
        return 10
    f = tmp_path.joinpath('f.txt')
    f.write_text('# allow comments\n# and allow pip settings\n--index-url https://pypi.devinfra.sentry.io/simple\na==1\nb==2\n')
    assert lint_requirements.main((str(f),)) == 0

def test_not_ok_classic_git_url(tmp_path):
    if False:
        print('Hello World!')
    f = tmp_path.joinpath('f.txt')
    f.write_text('git+https://github.com/asottile/astpretty@3.0.0#egg=astpretty')
    with pytest.raises(SystemExit) as excinfo:
        lint_requirements.main((str(f),))
    expected = f'You cannot use dependencies that are not on PyPI directly.\nSee PEP440: https://www.python.org/dev/peps/pep-0440/#direct-references\n\n{f}:1: git+https://github.com/asottile/astpretty@3.0.0#egg=astpretty\n'
    (msg,) = excinfo.value.args
    assert msg == expected.rstrip()

def test_not_ok_new_style_git_url(tmp_path):
    if False:
        while True:
            i = 10
    f = tmp_path.joinpath('f.txt')
    f.write_text('astpretty @ git+https://github.com/asottile/astpretty@3.0.0')
    with pytest.raises(SystemExit) as excinfo:
        lint_requirements.main((str(f),))
    expected = f'You cannot use dependencies that are not on PyPI directly.\nSee PEP440: https://www.python.org/dev/peps/pep-0440/#direct-references\n\n{f}:1: astpretty @ git+https://github.com/asottile/astpretty@3.0.0\n'
    (msg,) = excinfo.value.args
    assert msg == expected.rstrip()