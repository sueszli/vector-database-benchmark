from thefuck.shells import shell
from thefuck.utils import for_app

@for_app('terraform')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'this module is not yet installed' in command.output.lower() or 'initialization required' in command.output.lower()

def get_new_command(command):
    if False:
        return 10
    return shell.and_('terraform init', command.script)