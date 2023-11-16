from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        print('Hello World!')
    return 'merge' in command.script and 'fatal: refusing to merge unrelated histories' in command.output

@git_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return command.script + ' --allow-unrelated-histories'