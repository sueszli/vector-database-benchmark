import re
from thefuck.specific.sudo import sudo_support

@sudo_support
def match(command):
    if False:
        return 10
    return 'mkdir' in command.script and 'No such file or directory' in command.output

@sudo_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return re.sub('\\bmkdir (.*)', 'mkdir -p \\1', command.script)