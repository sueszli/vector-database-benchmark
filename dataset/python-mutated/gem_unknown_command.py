import re
import subprocess
from thefuck.utils import for_app, eager, replace_command, cache, which

@for_app('gem')
def match(command):
    if False:
        return 10
    return 'ERROR:  While executing gem ... (Gem::CommandLineError)' in command.output and 'Unknown command' in command.output

def _get_unknown_command(command):
    if False:
        while True:
            i = 10
    return re.findall('Unknown command (.*)$', command.output)[0]

@eager
def _get_all_commands():
    if False:
        while True:
            i = 10
    proc = subprocess.Popen(['gem', 'help', 'commands'], stdout=subprocess.PIPE)
    for line in proc.stdout.readlines():
        line = line.decode()
        if line.startswith('    '):
            yield line.strip().split(' ')[0]
if which('gem'):
    _get_all_commands = cache(which('gem'))(_get_all_commands)

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    unknown_command = _get_unknown_command(command)
    all_commands = _get_all_commands()
    return replace_command(command, unknown_command, all_commands)