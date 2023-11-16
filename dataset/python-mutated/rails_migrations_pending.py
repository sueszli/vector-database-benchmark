import re
from thefuck.shells import shell
SUGGESTION_REGEX = 'To resolve this issue, run:\\s+(.*?)\\n'

def match(command):
    if False:
        return 10
    return 'Migrations are pending. To resolve this issue, run:' in command.output

def get_new_command(command):
    if False:
        return 10
    migration_script = re.search(SUGGESTION_REGEX, command.output).group(1)
    return shell.and_(migration_script, command.script)