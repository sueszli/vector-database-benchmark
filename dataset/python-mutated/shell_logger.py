import json
import os
import socket
try:
    from shutil import get_terminal_size
except ImportError:
    from backports.shutil_get_terminal_size import get_terminal_size
import pyte
from .. import const, logs

def _get_socket_path():
    if False:
        for i in range(10):
            print('nop')
    return os.environ.get(const.SHELL_LOGGER_SOCKET_ENV)

def is_available():
    if False:
        while True:
            i = 10
    'Returns `True` if shell logger socket available.\n\n    :rtype: book\n\n    '
    path = _get_socket_path()
    if not path:
        return False
    return os.path.exists(path)

def _get_last_n(n):
    if False:
        i = 10
        return i + 15
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(_get_socket_path())
        request = json.dumps({'type': 'list', 'count': n}) + '\n'
        client.sendall(request.encode('utf-8'))
        response = client.makefile().readline()
        return json.loads(response)['commands']

def _get_output_lines(output):
    if False:
        print('Hello World!')
    lines = output.split('\n')
    screen = pyte.Screen(get_terminal_size().columns, len(lines))
    stream = pyte.Stream(screen)
    stream.feed('\n'.join(lines))
    return screen.display

def get_output(script):
    if False:
        return 10
    'Gets command output from shell logger.'
    with logs.debug_time(u'Read output from external shell logger'):
        commands = _get_last_n(const.SHELL_LOGGER_LIMIT)
        for command in commands:
            if command['command'] == script:
                lines = _get_output_lines(command['output'])
                output = '\n'.join(lines).strip()
                return output
            else:
                logs.warn("Output isn't available in shell logger")
                return None