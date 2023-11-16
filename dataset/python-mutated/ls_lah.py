from thefuck.utils import for_app

@for_app('ls')
def match(command):
    if False:
        return 10
    return command.script_parts and 'ls -' not in command.script

def get_new_command(command):
    if False:
        print('Hello World!')
    command = command.script_parts[:]
    command[0] = 'ls -lah'
    return ' '.join(command)