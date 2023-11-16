from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'tag' in command.script_parts and 'already exists' in command.output

@git_support
def get_new_command(command):
    if False:
        i = 10
        return i + 15
    return replace_argument(command.script, 'tag', 'tag --force')