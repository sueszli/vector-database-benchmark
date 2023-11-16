from thefuck.utils import replace_argument
import re
help_regex = "(?:Run|Try) '([^']+)'(?: or '[^']+')? for (?:details|more information)."

def match(command):
    if False:
        while True:
            i = 10
    if re.search(help_regex, command.output, re.I) is not None:
        return True
    if '--help' in command.output:
        return True
    return False

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    if re.search(help_regex, command.output) is not None:
        match_obj = re.search(help_regex, command.output, re.I)
        return match_obj.group(1)
    return replace_argument(command.script, '-h', '--help')
enabled_by_default = True
priority = 5000