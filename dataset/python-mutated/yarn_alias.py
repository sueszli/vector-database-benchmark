import re
from thefuck.utils import replace_argument, for_app

@for_app('yarn', at_least=1)
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'Did you mean' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    broken = command.script_parts[1]
    fix = re.findall('Did you mean [`"](?:yarn )?([^`"]*)[`"]', command.output)[0]
    return replace_argument(command.script, broken, fix)