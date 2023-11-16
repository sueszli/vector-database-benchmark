import re
from thefuck.utils import for_app, replace_command
from thefuck.specific.npm import get_scripts, npm_available
enabled_by_default = npm_available

@for_app('npm')
def match(command):
    if False:
        return 10
    return any((part.startswith('ru') for part in command.script_parts)) and 'npm ERR! missing script: ' in command.output

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    misspelled_script = re.findall('.*missing script: (.*)\\n', command.output)[0]
    return replace_command(command, misspelled_script, get_scripts())