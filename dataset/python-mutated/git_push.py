import re
from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        i = 10
        return i + 15
    return 'push' in command.script_parts and 'git push --set-upstream' in command.output

def _get_upstream_option_index(command_parts):
    if False:
        return 10
    if '--set-upstream' in command_parts:
        return command_parts.index('--set-upstream')
    elif '-u' in command_parts:
        return command_parts.index('-u')
    else:
        return None

@git_support
def get_new_command(command):
    if False:
        while True:
            i = 10
    command_parts = command.script_parts[:]
    upstream_option_index = _get_upstream_option_index(command_parts)
    if upstream_option_index is not None:
        command_parts.pop(upstream_option_index)
        if len(command_parts) > upstream_option_index:
            command_parts.pop(upstream_option_index)
    else:
        push_idx = command_parts.index('push') + 1
        while len(command_parts) > push_idx and command_parts[len(command_parts) - 1][0] != '-':
            command_parts.pop(len(command_parts) - 1)
    arguments = re.findall('git push (.*)', command.output)[-1].replace("'", "\\'").strip()
    return replace_argument(' '.join(command_parts), 'push', 'push {}'.format(arguments))