import re
from thefuck.shells import shell
from thefuck.utils import for_app

@for_app('touch')
def match(command):
    if False:
        return 10
    return 'No such file or directory' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    path = re.findall("touch: (?:cannot touch ')?(.+)/.+'?:", command.output)[0]
    return shell.and_(u'mkdir -p {}'.format(path), command.script)