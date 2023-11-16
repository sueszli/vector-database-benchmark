import re
from thefuck.shells import shell
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        print('Hello World!')
    return bool(re.search('src refspec \\w+ does not match any', command.output))

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return shell.and_('git commit -m "Initial commit"', command.script)