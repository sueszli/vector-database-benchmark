from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        print('Hello World!')
    return 'add' in command.script_parts and 'Use -f if you really want to add them.' in command.output

@git_support
def get_new_command(command):
    if False:
        return 10
    return replace_argument(command.script, 'add', 'add --force')