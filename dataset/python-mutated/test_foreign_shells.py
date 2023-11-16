"""Tests foreign shells."""
import os
import subprocess
import pytest
from xonsh.foreign_shells import foreign_shell_data, parse_aliases, parse_env
from xonsh.pytest.tools import skip_if_on_unix, skip_if_on_windows

def test_parse_env():
    if False:
        while True:
            i = 10
    exp = {'X': 'YES', 'Y': 'NO'}
    s = 'some garbage\n__XONSH_ENV_BEG__\nY=NO\nX=YES\n__XONSH_ENV_END__\nmore filth'
    obs = parse_env(s)
    assert exp == obs

def test_parse_env_newline():
    if False:
        while True:
            i = 10
    exp = {'X': 'YES', 'Y': 'NO', 'PROMPT': 'why\nme '}
    s = 'some garbage\n__XONSH_ENV_BEG__\nY=NO\nPROMPT=why\nme \nX=YES\n__XONSH_ENV_END__\nmore filth'
    obs = parse_env(s)
    assert exp == obs

def test_parse_env_equals():
    if False:
        print('Hello World!')
    exp = {'X': 'YES', 'Y': 'NO', 'LS_COLORS': '*.tar=5'}
    s = 'some garbage\n__XONSH_ENV_BEG__\nY=NO\nLS_COLORS=*.tar=5\nX=YES\n__XONSH_ENV_END__\nmore filth'
    obs = parse_env(s)
    assert exp == obs

def test_parse_aliases():
    if False:
        i = 10
        return i + 15
    exp = {'x': ['yes', '-1'], 'y': ['echo', 'no'], 'z': ['echo', 'True', '&&', 'echo', 'Next', '||', 'echo', 'False']}
    s = "some garbage\n__XONSH_ALIAS_BEG__\nalias x='yes -1'\nalias y='echo    no'\nalias z='echo True && \\\n echo Next || \\\n echo False'\n__XONSH_ALIAS_END__\nmore filth"
    obs = parse_aliases(s, 'bash')
    assert exp == obs

@skip_if_on_windows
def test_foreign_bash_data():
    if False:
        for i in range(10):
            print('nop')
    expenv = {'EMERALD': 'SWORD', 'MIGHTY': 'WARRIOR'}
    expaliases = {'l': ['ls', '-CF'], 'la': ['ls', '-A'], 'll': ['ls', '-a', '-lF']}
    rcfile = os.path.join(os.path.dirname(__file__), 'bashrc.sh')
    try:
        (obsenv, obsaliases) = foreign_shell_data('bash', currenv=(), extra_args=('--rcfile', rcfile), safe=False)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for (key, expval) in expenv.items():
        assert expval == obsenv.get(key, False)
    for (key, expval) in expaliases.items():
        assert expval == obsaliases.get(key, False)

@skip_if_on_unix
def test_foreign_cmd_data():
    if False:
        for i in range(10):
            print('nop')
    env = (('ENV_TO_BE_REMOVED', 'test'),)
    batchfile = os.path.join(os.path.dirname(__file__), 'batch.bat')
    source_cmd = f'call "{batchfile}"\necho off'
    try:
        (obsenv, _) = foreign_shell_data('cmd', prevcmd=source_cmd, currenv=env, interactive=False, sourcer='call', envcmd='set', use_tmpfile=True, safe=False)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    assert 'ENV_TO_BE_ADDED' in obsenv
    assert obsenv['ENV_TO_BE_ADDED'] == 'Hallo world'
    assert 'ENV_TO_BE_REMOVED' not in obsenv