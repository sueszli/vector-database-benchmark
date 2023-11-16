import re
from thefuck.specific.sudo import sudo_support
from thefuck.utils import for_app

@sudo_support
@for_app('cp')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    output = command.output.lower()
    return 'omitting directory' in output or 'is a directory' in output

@sudo_support
def get_new_command(command):
    if False:
        print('Hello World!')
    return re.sub('^cp', 'cp -a', command.script)