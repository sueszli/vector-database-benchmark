import re
from thefuck.utils import replace_argument, for_app
from thefuck.specific.sudo import sudo_support

@sudo_support
@for_app('pip', 'pip2', 'pip3')
def match(command):
    if False:
        while True:
            i = 10
    return 'pip' in command.script and 'unknown command' in command.output and ('maybe you meant' in command.output)

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    broken_cmd = re.findall('ERROR: unknown command "([^"]+)"', command.output)[0]
    new_cmd = re.findall('maybe you meant "([^"]+)"', command.output)[0]
    return replace_argument(command.script, broken_cmd, new_cmd)