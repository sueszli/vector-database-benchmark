import pytest
import subprocess
import json
import sys
from numpy.distutils import _shell_utils
from numpy.testing import IS_WASM
argv_cases = [['exe'], ['path/exe'], ['path\\exe'], ['\\\\server\\path\\exe'], ['path to/exe'], ['path to\\exe'], ['exe', '--flag'], ['path/exe', '--flag'], ['path\\exe', '--flag'], ['path to/exe', '--flag'], ['path to\\exe', '--flag'], ['path to/exe', '--flag-"quoted"'], ['path to\\exe', '--flag-"quoted"'], ['path to/exe', '"--flag-quoted"'], ['path to\\exe', '"--flag-quoted"']]

@pytest.fixture(params=[_shell_utils.WindowsParser, _shell_utils.PosixParser])
def Parser(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.fixture
def runner(Parser):
    if False:
        i = 10
        return i + 15
    if Parser != _shell_utils.NativeParser:
        pytest.skip('Unable to run with non-native parser')
    if Parser == _shell_utils.WindowsParser:
        return lambda cmd: subprocess.check_output(cmd)
    elif Parser == _shell_utils.PosixParser:
        return lambda cmd: subprocess.check_output(cmd, shell=True)
    else:
        raise NotImplementedError

@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
@pytest.mark.parametrize('argv', argv_cases)
def test_join_matches_subprocess(Parser, runner, argv):
    if False:
        return 10
    '\n    Test that join produces strings understood by subprocess\n    '
    cmd = [sys.executable, '-c', 'import json, sys; print(json.dumps(sys.argv[1:]))']
    joined = Parser.join(cmd + argv)
    json_out = runner(joined).decode()
    assert json.loads(json_out) == argv

@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
@pytest.mark.parametrize('argv', argv_cases)
def test_roundtrip(Parser, argv):
    if False:
        while True:
            i = 10
    '\n    Test that split is the inverse operation of join\n    '
    try:
        joined = Parser.join(argv)
        assert argv == Parser.split(joined)
    except NotImplementedError:
        pytest.skip('Not implemented')