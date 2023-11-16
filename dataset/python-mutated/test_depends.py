from __future__ import annotations
import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from tox.pytest import ToxProjectCreator

def test_depends(tox_project: ToxProjectCreator, patch_prev_py: Callable[[bool], tuple[str, str]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    (prev_ver, impl) = patch_prev_py(True)
    ver = sys.version_info[0:2]
    py = f"py{''.join((str(i) for i in ver))}"
    prev_py = f'py{prev_ver}'
    ini = f'\n    [tox]\n    env_list = py,{py},{prev_py},py31,cov2,cov\n    [testenv]\n    package = wheel\n    [testenv:cov]\n    depends = py,{py},{prev_py},py31\n    skip_install = true\n    [testenv:cov2]\n    depends = cov\n    skip_install = true\n    '
    project = tox_project({'tox.ini': ini, 'pyproject.toml': ''})
    outcome = project.run('de')
    outcome.assert_success()
    expected = f'\n    Execution order: py, {py}, {prev_py}, py31, cov, cov2\n    ALL\n       py ~ .pkg\n       {py} ~ .pkg\n       {prev_py} ~ .pkg | .pkg-{impl}{prev_ver}\n       py31 ~ .pkg | ... (could not find python interpreter with spec(s): py31)\n       cov2\n          cov\n             py ~ .pkg\n             {py} ~ .pkg\n             {prev_py} ~ .pkg | .pkg-{impl}{prev_ver}\n             py31 ~ .pkg | ... (could not find python interpreter with spec(s): py31)\n       cov\n          py ~ .pkg\n          {py} ~ .pkg\n          {prev_py} ~ .pkg | .pkg-{impl}{prev_ver}\n          py31 ~ .pkg | ... (could not find python interpreter with spec(s): py31)\n    '
    assert outcome.out == dedent(expected).lstrip()

def test_depends_help(tox_project: ToxProjectCreator) -> None:
    if False:
        for i in range(10):
            print('nop')
    outcome = tox_project({'tox.ini': ''}).run('de', '-h')
    outcome.assert_success()