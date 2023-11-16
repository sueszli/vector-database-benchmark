from thefuck.utils import for_app

@for_app('go')
def match(command):
    if False:
        while True:
            i = 10
    return command.script.startswith('go run ') and (not command.script.endswith('.go'))

def get_new_command(command):
    if False:
        while True:
            i = 10
    return command.script + '.go'