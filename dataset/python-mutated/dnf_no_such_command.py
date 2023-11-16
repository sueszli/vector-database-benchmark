import subprocess
import re
from thefuck.specific.sudo import sudo_support
from thefuck.utils import for_app, replace_command
from thefuck.specific.dnf import dnf_available
regex = re.compile('No such command: (.*)\\.')

@sudo_support
@for_app('dnf')
def match(command):
    if False:
        i = 10
        return i + 15
    return 'no such command' in command.output.lower()

def _parse_operations(help_text_lines):
    if False:
        print('Hello World!')
    operation_regex = re.compile('^([a-z-]+) +', re.MULTILINE)
    return operation_regex.findall(help_text_lines)

def _get_operations():
    if False:
        return 10
    proc = subprocess.Popen(['dnf', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.read().decode('utf-8')
    return _parse_operations(lines)

@sudo_support
def get_new_command(command):
    if False:
        print('Hello World!')
    misspelled_command = regex.findall(command.output)[0]
    return replace_command(command, misspelled_command, _get_operations())
enabled_by_default = dnf_available