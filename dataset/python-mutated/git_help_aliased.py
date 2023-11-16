from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        i = 10
        return i + 15
    return 'help' in command.script and ' is aliased to ' in command.output

@git_support
def get_new_command(command):
    if False:
        while True:
            i = 10
    aliased = command.output.split('`', 2)[2].split("'", 1)[0].split(' ', 1)[0]
    return 'git help {}'.format(aliased)