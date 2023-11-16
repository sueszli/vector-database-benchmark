from thefuck.utils import for_app

@for_app('grep')
def match(command):
    if False:
        i = 10
        return i + 15
    return 'is a directory' in command.output.lower()

def get_new_command(command):
    if False:
        while True:
            i = 10
    return u'grep -r {}'.format(command.script[5:])