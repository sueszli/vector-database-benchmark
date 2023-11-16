import re
from thefuck.utils import replace_command

def match(command):
    if False:
        for i in range(10):
            print('nop')
    return re.search('([^:]*): Unknown command.*', command.output) is not None and re.search('Did you mean ([^?]*)?', command.output) is not None

def get_new_command(command):
    if False:
        print('Hello World!')
    broken_cmd = re.findall('([^:]*): Unknown command.*', command.output)[0]
    matched = re.findall('Did you mean ([^?]*)?', command.output)
    return replace_command(command, broken_cmd, matched)