from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        return 10
    return 'commit' in command.script_parts

@git_support
def get_new_command(command):
    if False:
        while True:
            i = 10
    return 'git reset HEAD~'