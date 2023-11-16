from thefuck import utils
from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        while True:
            i = 10
    if command.script_parts and len(command.script_parts) > 1:
        return command.script_parts[1] == 'stash' and 'usage:' in command.output
    else:
        return False
stash_commands = ('apply', 'branch', 'clear', 'drop', 'list', 'pop', 'save', 'show')

@git_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    stash_cmd = command.script_parts[2]
    fixed = utils.get_closest(stash_cmd, stash_commands, fallback_to_first=False)
    if fixed is not None:
        return replace_argument(command.script, stash_cmd, fixed)
    else:
        cmd = command.script_parts[:]
        cmd.insert(2, 'save')
        return ' '.join(cmd)