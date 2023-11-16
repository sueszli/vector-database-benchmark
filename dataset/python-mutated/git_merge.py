import re
from thefuck.utils import replace_argument
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        return 10
    return 'merge' in command.script and ' - not something we can merge' in command.output and ('Did you mean this?' in command.output)

@git_support
def get_new_command(command):
    if False:
        i = 10
        return i + 15
    unknown_branch = re.findall('merge: (.+) - not something we can merge', command.output)[0]
    remote_branch = re.findall('Did you mean this\\?\\n\\t([^\\n]+)', command.output)[0]
    return replace_argument(command.script, unknown_branch, remote_branch)