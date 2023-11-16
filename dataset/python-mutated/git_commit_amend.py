from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'commit' in command.script_parts

@git_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return 'git commit --amend'