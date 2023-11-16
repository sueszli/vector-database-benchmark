import re
from thefuck.specific.apt import apt_available
from thefuck.utils import for_app
enabled_by_default = apt_available

@for_app('apt-get')
def match(command):
    if False:
        i = 10
        return i + 15
    return command.script.startswith('apt-get search')

def get_new_command(command):
    if False:
        print('Hello World!')
    return re.sub('^apt-get', 'apt-cache', command.script)