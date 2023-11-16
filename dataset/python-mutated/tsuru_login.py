from thefuck.shells import shell
from thefuck.utils import for_app

@for_app('tsuru')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'not authenticated' in command.output and 'session has expired' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    return shell.and_('tsuru login', command.script)