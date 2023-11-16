import re
from thefuck.utils import get_all_matched_commands, replace_command
from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        i = 10
        return i + 15
    '\n    Match a mistyped command\n    '
    return 'lfs' in command.script and 'Did you mean this?' in command.output

@git_support
def get_new_command(command):
    if False:
        print('Hello World!')
    broken_cmd = re.findall('Error: unknown command "([^"]*)" for "git-lfs"', command.output)[0]
    matched = get_all_matched_commands(command.output, ['Did you mean', ' for usage.'])
    return replace_command(command, broken_cmd, matched)