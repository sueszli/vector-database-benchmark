from thefuck.utils import for_app

@for_app('python')
def match(command):
    if False:
        return 10
    return not command.script.endswith('.py')

def get_new_command(command):
    if False:
        return 10
    return command.script + '.py'