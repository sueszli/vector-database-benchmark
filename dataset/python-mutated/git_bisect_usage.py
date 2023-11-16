import re
from thefuck.utils import replace_command
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        while True:
            i = 10
    return 'bisect' in command.script_parts and 'usage: git bisect' in command.output

@git_support
def get_new_command(command):
    if False:
        i = 10
        return i + 15
    broken = re.findall('git bisect ([^ $]*).*', command.script)[0]
    usage = re.findall('usage: git bisect \\[([^\\]]+)\\]', command.output)[0]
    return replace_command(command, broken, usage.split('|'))