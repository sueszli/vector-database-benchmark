import os
import pytest
from xonsh.prompt import gitstatus
from xonsh.prompt.base import _format_value

@pytest.fixture(autouse=True)
def git_no_stash(mocker):
    if False:
        return 10
    return mocker.patch.object(gitstatus, 'get_stash_count', return_value=0)

@pytest.fixture
def prompts(xession):
    if False:
        i = 10
        return i + 15
    fields = xession.env['PROMPT_FIELDS']
    yield fields
    fields.clear()
    fields.reset()

@pytest.fixture
def fake_proc(fake_process):
    if False:
        print('Hello World!')

    def wrap(map: 'dict'):
        if False:
            print('Hello World!')
        for (command, stdout) in map.items():
            fake_process.register_subprocess(command=command, stdout=stdout)
        return fake_process
    return wrap

@pytest.mark.parametrize('hidden, exp', [((), '{CYAN}gitstatus-opt↑·7↓·2{RESET}|{RED}●1{RESET}{BLUE}+3{RESET}{BLUE}+49{RESET}{RED}-26{RESET}'), (('.lines_added', '.lines_removed'), '{CYAN}gitstatus-opt↑·7↓·2{RESET}|{RED}●1{RESET}{BLUE}+3{RESET}')])
def test_gitstatus_dirty(prompts, fake_proc, hidden, exp, xession):
    if False:
        return 10
    prompts['gitstatus'].hidden = hidden
    dirty = {'git status --porcelain --branch': b'## gitstatus-opt...origin/gitstatus-opt [ahead 7, behind 2]\n M requirements/tests.txt\nAM tests/prompt/test_gitstatus.py\n M tests/prompt/test_vc.py', 'git rev-parse --git-dir': b'.git', 'git diff --numstat': b'1       0       requirements/tests.txt\n26      0       tests/prompt/test_gitstatus.py\n22      26      tests/prompt/test_vc.py'}
    fake_proc(dirty)
    assert format(prompts.pick('gitstatus')) == exp

def test_gitstatus_clean(prompts, fake_proc):
    if False:
        i = 10
        return i + 15
    clean = {'git status --porcelain --branch': b'## gitstatus-opt...origin/gitstatus-opt [ahead 7, behind 2]', 'git rev-parse --git-dir': b'.git', 'git diff --numstat': b''}
    fake_proc(clean)
    exp = '{CYAN}gitstatus-opt↑·7↓·2{RESET}|{BOLD_GREEN}✓{RESET}'
    assert format(prompts.pick('gitstatus')) == exp
    assert _format_value(prompts.pick('gitstatus'), None, None) == exp
    assert _format_value(prompts.pick('gitstatus'), '{}', None) == exp

def test_no_git(prompts, fake_process, tmp_path):
    if False:
        while True:
            i = 10
    os.chdir(tmp_path)
    err = b'fatal: not a git repository (or any of the parent directories): .git'
    fake_process.register_subprocess(command='git rev-parse --git-dir', stderr=err, returncode=128)
    for field in prompts.get_fields(gitstatus):
        assert prompts.pick_val(field) is None
        assert _format_value(prompts.pick(field), None, None) == ''
        assert _format_value(prompts.pick(field), 'hello {}', None) == ''