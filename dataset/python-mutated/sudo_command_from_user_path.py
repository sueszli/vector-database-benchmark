import re
from thefuck.utils import for_app, which, replace_argument

def _get_command_name(command):
    if False:
        i = 10
        return i + 15
    found = re.findall('sudo: (.*): command not found', command.output)
    if found:
        return found[0]

@for_app('sudo')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    if 'command not found' in command.output:
        command_name = _get_command_name(command)
        return which(command_name)

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    command_name = _get_command_name(command)
    return replace_argument(command.script, command_name, u'env "PATH=$PATH" {}'.format(command_name))