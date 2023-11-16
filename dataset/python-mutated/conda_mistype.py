import re
from thefuck.utils import replace_command, for_app

@for_app('conda')
def match(command):
    if False:
        return 10
    '\n    Match a mistyped command\n    '
    return "Did you mean 'conda" in command.output

def get_new_command(command):
    if False:
        return 10
    match = re.findall("'conda ([^']*)'", command.output)
    broken_cmd = match[0]
    correct_cmd = match[1]
    return replace_command(command, broken_cmd, [correct_cmd])