import pytest
from xonsh.pytest.tools import skip_if_not_has
pytestmark = skip_if_not_has('gh')

@pytest.mark.parametrize('line, exp', [['gh rep', {'repo'}], ['gh repo ', {'archive', 'clone', 'create', 'delete', 'edit', 'fork'}]])
def test_completions(line, exp, check_completer, xsh_with_env):
    if False:
        while True:
            i = 10
    comps = check_completer(line, prefix=None)
    if callable(exp):
        exp = exp()
    assert comps.intersection(exp)