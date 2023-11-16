import os
from thefuck.utils import for_app

def _is_recursive(part):
    if False:
        while True:
            i = 10
    if part == '--recurse':
        return True
    elif not part.startswith('--') and part.startswith('-') and ('r' in part):
        return True

def _isdir(part):
    if False:
        while True:
            i = 10
    return not part.startswith('-') and os.path.isdir(part)

@for_app('prove')
def match(command):
    if False:
        print('Hello World!')
    return 'NOTESTS' in command.output and (not any((_is_recursive(part) for part in command.script_parts[1:]))) and any((_isdir(part) for part in command.script_parts[1:]))

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    parts = command.script_parts[:]
    parts.insert(1, '-r')
    return u' '.join(parts)