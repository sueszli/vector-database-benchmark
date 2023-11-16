from thefuck.utils import get_all_executables, get_close_matches, get_valid_history_without_current, get_closest, which
from thefuck.specific.sudo import sudo_support

@sudo_support
def match(command):
    if False:
        return 10
    return not which(command.script_parts[0]) and ('not found' in command.output or 'is not recognized as' in command.output) and bool(get_close_matches(command.script_parts[0], get_all_executables()))

def _get_used_executables(command):
    if False:
        for i in range(10):
            print('nop')
    for script in get_valid_history_without_current(command):
        yield script.split(' ')[0]

@sudo_support
def get_new_command(command):
    if False:
        return 10
    old_command = command.script_parts[0]
    already_used = get_closest(old_command, _get_used_executables(command), fallback_to_first=False)
    if already_used:
        new_cmds = [already_used]
    else:
        new_cmds = []
    new_cmds += [cmd for cmd in get_close_matches(old_command, get_all_executables()) if cmd not in new_cmds]
    return [command.script.replace(old_command, cmd, 1) for cmd in new_cmds]
priority = 3000