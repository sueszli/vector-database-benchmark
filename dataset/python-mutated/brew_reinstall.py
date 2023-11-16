import re
from thefuck.utils import for_app
warning_regex = re.compile('Warning: (?:.(?!is ))+ is already installed and up-to-date')
message_regex = re.compile('To reinstall (?:(?!, ).)+, run `brew reinstall [^`]+`')

@for_app('brew', at_least=2)
def match(command):
    if False:
        while True:
            i = 10
    return 'install' in command.script and warning_regex.search(command.output) and message_regex.search(command.output)

def get_new_command(command):
    if False:
        while True:
            i = 10
    return command.script.replace('install', 'reinstall')