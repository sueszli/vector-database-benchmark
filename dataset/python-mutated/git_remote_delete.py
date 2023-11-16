import re
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'remote delete' in command.script

@git_support
def get_new_command(command):
    if False:
        return 10
    return re.sub('delete', 'remove', command.script, 1)