from thefuck.utils import for_app
from thefuck.shells import shell

@for_app('docker')
def match(command):
    if False:
        i = 10
        return i + 15
    return 'docker' in command.script and 'access denied' in command.output and ("may require 'docker login'" in command.output)

def get_new_command(command):
    if False:
        print('Hello World!')
    return shell.and_('docker login', command.script)