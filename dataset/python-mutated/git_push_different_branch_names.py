import re
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        return 10
    return 'push' in command.script and 'The upstream branch of your current branch does not match' in command.output

@git_support
def get_new_command(command):
    if False:
        i = 10
        return i + 15
    return re.findall('^ +(git push [^\\s]+ [^\\s]+)', command.output, re.MULTILINE)[0]