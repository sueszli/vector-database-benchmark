import re
from thefuck.utils import for_app
regex = re.compile('Run "(.*)" instead')

@for_app('yarn', at_least=1)
def match(command):
    if False:
        i = 10
        return i + 15
    return regex.findall(command.output)

def get_new_command(command):
    if False:
        return 10
    return regex.findall(command.output)[0]