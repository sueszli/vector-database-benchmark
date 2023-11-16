from thefuck.shells import shell
from thefuck.specific.git import git_support
from thefuck.utils import memoize

@memoize
def first_0flag(script_parts):
    if False:
        while True:
            i = 10
    return next((p for p in script_parts if len(p) == 2 and p.startswith('0')), None)

@git_support
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return command.script_parts[1] == 'branch' and first_0flag(command.script_parts)

@git_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    branch_name = first_0flag(command.script_parts)
    fixed_flag = branch_name.replace('0', '-')
    fixed_script = command.script.replace(branch_name, fixed_flag)
    if "A branch named '" in command.output and "' already exists." in command.output:
        delete_branch = u'git branch -D {}'.format(branch_name)
        return shell.and_(delete_branch, fixed_script)
    return fixed_script