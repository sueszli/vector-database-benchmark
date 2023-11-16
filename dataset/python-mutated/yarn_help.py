import re
from thefuck.utils import for_app
from thefuck.system import open_command

@for_app('yarn', at_least=2)
def match(command):
    if False:
        print('Hello World!')
    return command.script_parts[1] == 'help' and 'for documentation about this command.' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    url = re.findall('Visit ([^ ]*) for documentation about this command.', command.output)[0]
    return open_command(url)