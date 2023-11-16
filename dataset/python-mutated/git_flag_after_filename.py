import re
from thefuck.specific.git import git_support
error_pattern = "fatal: bad flag '(.*?)' used after filename"
error_pattern2 = "fatal: option '(.*?)' must come before non-option arguments"

@git_support
def match(command):
    if False:
        while True:
            i = 10
    return re.search(error_pattern, command.output) or re.search(error_pattern2, command.output)

@git_support
def get_new_command(command):
    if False:
        return 10
    command_parts = command.script_parts[:]
    bad_flag = match(command).group(1)
    bad_flag_index = command_parts.index(bad_flag)
    for index in reversed(range(bad_flag_index)):
        if command_parts[index][0] != '-':
            filename_index = index
            break
    (command_parts[bad_flag_index], command_parts[filename_index]) = (command_parts[filename_index], command_parts[bad_flag_index])
    return u' '.join(command_parts)