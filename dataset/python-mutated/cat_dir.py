import os
from thefuck.utils import for_app

@for_app('cat', at_least=1)
def match(command):
    if False:
        return 10
    return command.output.startswith('cat: ') and os.path.isdir(command.script_parts[1])

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return command.script.replace('cat', 'ls', 1)