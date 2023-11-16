import os
from thefuck.utils import for_app

def _get_actual_file(parts):
    if False:
        print('Hello World!')
    for part in parts[1:]:
        if os.path.isfile(part) or os.path.isdir(part):
            return part

@for_app('grep', 'egrep')
def match(command):
    if False:
        while True:
            i = 10
    return ': No such file or directory' in command.output and _get_actual_file(command.script_parts)

def get_new_command(command):
    if False:
        while True:
            i = 10
    actual_file = _get_actual_file(command.script_parts)
    parts = command.script_parts[:]
    parts.remove(actual_file)
    parts.append(actual_file)
    return ' '.join(parts)