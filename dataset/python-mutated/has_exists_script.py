import os
from thefuck.specific.sudo import sudo_support

@sudo_support
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return command.script_parts and os.path.exists(command.script_parts[0]) and ('command not found' in command.output)

@sudo_support
def get_new_command(command):
    if False:
        print('Hello World!')
    return u'./{}'.format(command.script)