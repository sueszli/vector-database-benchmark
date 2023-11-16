import re
from thefuck.utils import for_app
MISTAKE = '(?<=Terraform has no command named ")([^"]+)(?="\\.)'
FIX = '(?<=Did you mean ")([^"]+)(?="\\?)'

@for_app('terraform')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return re.search(MISTAKE, command.output) and re.search(FIX, command.output)

def get_new_command(command):
    if False:
        return 10
    mistake = re.search(MISTAKE, command.output).group(0)
    fix = re.search(FIX, command.output).group(0)
    return command.script.replace(mistake, fix)