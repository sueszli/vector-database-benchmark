from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        return 10
    return 'branch -d' in command.script and 'If you are sure you want to delete it' in command.output

@git_support
def get_new_command(command):
    if False:
        while True:
            i = 10
    return replace_argument(command.script, '-d', '-D')