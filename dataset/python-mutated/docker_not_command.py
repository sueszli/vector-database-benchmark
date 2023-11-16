from itertools import dropwhile, takewhile, islice
import re
import subprocess
from thefuck.utils import replace_command, for_app, which, cache
from thefuck.specific.sudo import sudo_support

@sudo_support
@for_app('docker')
def match(command):
    if False:
        return 10
    return 'is not a docker command' in command.output or 'Usage:\tdocker' in command.output

def _parse_commands(lines, starts_with):
    if False:
        print('Hello World!')
    lines = dropwhile(lambda line: not line.startswith(starts_with), lines)
    lines = islice(lines, 1, None)
    lines = list(takewhile(lambda line: line.strip(), lines))
    return [line.strip().split(' ')[0] for line in lines]

def get_docker_commands():
    if False:
        for i in range(10):
            print('nop')
    proc = subprocess.Popen('docker', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.readlines() or proc.stderr.readlines()
    lines = [line.decode('utf-8') for line in lines]
    if 'Management Commands:\n' in lines:
        management_commands = _parse_commands(lines, 'Management Commands:')
    else:
        management_commands = []
    regular_commands = _parse_commands(lines, 'Commands:')
    return management_commands + regular_commands
if which('docker'):
    get_docker_commands = cache(which('docker'))(get_docker_commands)

@sudo_support
def get_new_command(command):
    if False:
        print('Hello World!')
    if 'Usage:' in command.output and len(command.script_parts) > 1:
        management_subcommands = _parse_commands(command.output.split('\n'), 'Commands:')
        return replace_command(command, command.script_parts[2], management_subcommands)
    wrong_command = re.findall("docker: '(\\w+)' is not a docker command.", command.output)[0]
    return replace_command(command, wrong_command, get_docker_commands())