import re
from subprocess import Popen, PIPE
from thefuck.utils import memoize, which
from thefuck.shells import shell
enabled_by_default = bool(which('lsof'))
patterns = ["bind on address \\('.*', (?P<port>\\d+)\\)", 'Unable to bind [^ ]*:(?P<port>\\d+)', "can't listen on port (?P<port>\\d+)", 'listen EADDRINUSE [^ ]*:(?P<port>\\d+)']

@memoize
def _get_pid_by_port(port):
    if False:
        i = 10
        return i + 15
    proc = Popen(['lsof', '-i', ':{}'.format(port)], stdout=PIPE)
    lines = proc.stdout.read().decode().split('\n')
    if len(lines) > 1:
        return lines[1].split()[1]
    else:
        return None

@memoize
def _get_used_port(command):
    if False:
        print('Hello World!')
    for pattern in patterns:
        matched = re.search(pattern, command.output)
        if matched:
            return matched.group('port')

def match(command):
    if False:
        return 10
    port = _get_used_port(command)
    return port and _get_pid_by_port(port)

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    port = _get_used_port(command)
    pid = _get_pid_by_port(port)
    return shell.and_(u'kill {}'.format(pid), command.script)