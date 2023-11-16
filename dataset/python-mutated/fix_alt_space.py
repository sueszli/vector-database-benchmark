import re
from thefuck.specific.sudo import sudo_support

@sudo_support
def match(command):
    if False:
        i = 10
        return i + 15
    return 'command not found' in command.output.lower() and u'\xa0' in command.script

@sudo_support
def get_new_command(command):
    if False:
        return 10
    return re.sub(u'\xa0', ' ', command.script)