from thefuck.utils import for_app
from thefuck.shells import shell

@for_app('docker')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    "\n    Matches a command's output with docker's output\n    warning you that you need to remove a container before removing an image.\n    "
    return 'image is being used by running container' in command.output

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    '\n    Prepends docker container rm -f {container ID} to\n    the previous docker image rm {image ID} command\n    '
    container_id = command.output.strip().split(' ')
    return shell.and_('docker container rm -f {}', '{}').format(container_id[-1], command.script)