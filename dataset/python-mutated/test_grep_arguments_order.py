import pytest
from thefuck.rules.grep_arguments_order import get_new_command, match
from thefuck.types import Command
output = 'grep: {}: No such file or directory'.format

@pytest.fixture(autouse=True)
def os_path(monkeypatch):
    if False:
        return 10
    monkeypatch.setattr('os.path.isfile', lambda x: not x.startswith('-'))

@pytest.mark.parametrize('script, file', [('grep test.py test', 'test'), ('grep -lir . test', 'test'), ('egrep test.py test', 'test'), ('egrep -lir . test', 'test')])
def test_match(script, file):
    if False:
        print('Hello World!')
    assert match(Command(script, output(file)))

@pytest.mark.parametrize('script, output', [('cat test.py', output('test')), ('grep test test.py', ''), ('grep -lir test .', ''), ('egrep test test.py', ''), ('egrep -lir test .', '')])
def test_not_match(script, output):
    if False:
        i = 10
        return i + 15
    assert not match(Command(script, output))

@pytest.mark.parametrize('script, output, result', [('grep test.py test', output('test'), 'grep test test.py'), ('grep -lir . test', output('test'), 'grep -lir test .'), ('grep . test -lir', output('test'), 'grep test -lir .'), ('egrep test.py test', output('test'), 'egrep test test.py'), ('egrep -lir . test', output('test'), 'egrep -lir test .'), ('egrep . test -lir', output('test'), 'egrep test -lir .')])
def test_get_new_command(script, output, result):
    if False:
        while True:
            i = 10
    assert get_new_command(Command(script, output)) == result