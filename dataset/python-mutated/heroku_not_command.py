import re
from thefuck.utils import for_app

@for_app('heroku')
def match(command):
    if False:
        print('Hello World!')
    return 'Run heroku _ to run' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    return re.findall('Run heroku _ to run ([^.]*)', command.output)[0]