import eventlet
import pytest
from mock import DEFAULT, patch
from nameko.cli.commands import Backdoor
from nameko.cli.main import setup_parser
from nameko.cli.run import setup_backdoor
from nameko.exceptions import CommandError

@pytest.fixture
def running_backdoor():
    if False:
        while True:
            i = 10
    runner = object()
    (green_socket, gt) = setup_backdoor(runner, 0)
    eventlet.sleep(0)
    socket_name = green_socket.fd.getsockname()
    return socket_name

def test_no_telnet():
    if False:
        i = 10
        return i + 15
    parser = setup_parser()
    args = parser.parse_args(['backdoor', '0'])
    with patch('nameko.cli.backdoor.os') as mock_os:
        mock_os.system.return_value = -1
        with pytest.raises(CommandError) as exc:
            Backdoor.main(args)
    assert 'Could not find an installed telnet' in str(exc)

def test_no_running_backdoor():
    if False:
        print('Hello World!')
    parser = setup_parser()
    args = parser.parse_args(['backdoor', '0'])
    with patch.multiple('nameko.cli.backdoor', call=DEFAULT, os=DEFAULT) as mocks:
        mocks['os'].system.return_value = 0
        mocks['call'].return_value = -1
        with pytest.raises(CommandError) as exc:
            Backdoor.main(args)
    assert 'Backdoor unreachable' in str(exc)

def test_basic(running_backdoor):
    if False:
        i = 10
        return i + 15
    socket_arg = '{}:{}'.format(*running_backdoor)
    parser = setup_parser()
    args = parser.parse_args(['backdoor', socket_arg])
    with patch.multiple('nameko.cli.backdoor', call=DEFAULT, os=DEFAULT) as mocks:
        mock_call = mocks['call']
        mocks['os'].system.return_value = 0
        mock_call.return_value = 0
        Backdoor.main(args)
    ((cmd,), _) = mock_call.call_args
    expected = ['rlwrap', 'netcat'] + list(map(str, running_backdoor)) + ['--close']
    assert cmd == expected

def test_default_host(running_backdoor):
    if False:
        return 10
    (_, port) = running_backdoor
    parser = setup_parser()
    args = parser.parse_args(['backdoor', str(port)])
    with patch.multiple('nameko.cli.backdoor', call=DEFAULT, os=DEFAULT) as mocks:
        mock_call = mocks['call']
        mocks['os'].system.return_value = 0
        mock_call.return_value = 0
        Backdoor.main(args)
    ((cmd,), _) = mock_call.call_args
    expected = ['rlwrap', 'netcat', 'localhost'] + [str(port)] + ['--close']
    assert cmd == expected

def test_stop(running_backdoor):
    if False:
        return 10
    (_, port) = running_backdoor
    parser = setup_parser()
    args = parser.parse_args(['backdoor', str(port)])
    with patch.multiple('nameko.cli.backdoor', call=DEFAULT, os=DEFAULT) as mocks:
        mocks['os'].system.side_effect = [-1, -1, 0, 0]
        mocks['call'].side_effect = [KeyboardInterrupt, 0]
        Backdoor.main(args)