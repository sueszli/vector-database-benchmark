from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        print('Hello World!')
    return 'diff' in command.script and '--staged' not in command.script

@git_support
def get_new_command(command):
    if False:
        print('Hello World!')
    return replace_argument(command.script, 'diff', 'diff --staged')