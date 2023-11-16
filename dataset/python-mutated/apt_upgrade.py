from thefuck.specific.apt import apt_available
from thefuck.specific.sudo import sudo_support
from thefuck.utils import for_app
enabled_by_default = apt_available

@sudo_support
@for_app('apt')
def match(command):
    if False:
        while True:
            i = 10
    return command.script == 'apt list --upgradable' and len(command.output.strip().split('\n')) > 1

@sudo_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return 'apt upgrade'