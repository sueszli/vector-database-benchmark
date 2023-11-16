from thefuck.utils import for_app

@for_app('ls')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return command.output.strip() == ''

def get_new_command(command):
    if False:
        while True:
            i = 10
    return ' '.join(['ls', '-A'] + command.script_parts[1:])