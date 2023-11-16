from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        print('Hello World!')
    return ' git clone ' in command.script and 'fatal: Too many arguments.' in command.output

@git_support
def get_new_command(command):
    if False:
        print('Hello World!')
    return command.script.replace(' git clone ', ' ', 1)