import re
from thefuck.utils import replace_argument, for_app

@for_app('cargo', at_least=1)
def match(command):
    if False:
        i = 10
        return i + 15
    return 'no such subcommand' in command.output.lower() and 'Did you mean' in command.output

def get_new_command(command):
    if False:
        return 10
    broken = command.script_parts[1]
    fix = re.findall('Did you mean `([^`]*)`', command.output)[0]
    return replace_argument(command.script, broken, fix)