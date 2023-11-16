from thefuck.shells import shell
from thefuck.specific.git import git_support
from thefuck.utils import replace_argument

@git_support
def match(command):
    if False:
        print('Hello World!')
    return ('branch -d' in command.script or 'branch -D' in command.script) and "error: Cannot delete branch '" in command.output and ("' checked out at '" in command.output)

@git_support
def get_new_command(command):
    if False:
        while True:
            i = 10
    return shell.and_('git checkout master', '{}').format(replace_argument(command.script, '-d', '-D'))