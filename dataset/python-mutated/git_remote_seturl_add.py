from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        i = 10
        return i + 15
    return 'set-url' in command.script and 'fatal: No such remote' in command.output

def get_new_command(command):
    if False:
        return 10
    return replace_argument(command.script, 'set-url', 'add')