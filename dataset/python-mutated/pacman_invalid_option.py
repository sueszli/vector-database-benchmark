from thefuck.specific.archlinux import archlinux_env
from thefuck.specific.sudo import sudo_support
from thefuck.utils import for_app
import re

@sudo_support
@for_app('pacman')
def match(command):
    if False:
        while True:
            i = 10
    return command.output.startswith("error: invalid option '-") and any((' -{}'.format(option) in command.script for option in 'surqfdvt'))

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    option = re.findall(' -[dfqrstuv]', command.script)[0]
    return re.sub(option, option.upper(), command.script)
enabled_by_default = archlinux_env()