from thefuck.utils import for_app

@for_app('ag')
def match(command):
    if False:
        print('Hello World!')
    return command.output.endswith('run ag with -Q\n')

def get_new_command(command):
    if False:
        return 10
    return command.script.replace('ag', 'ag -Q', 1)