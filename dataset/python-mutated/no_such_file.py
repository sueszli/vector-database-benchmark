import re
from thefuck.shells import shell
patterns = ("mv: cannot move '[^']*' to '([^']*)': No such file or directory", "mv: cannot move '[^']*' to '([^']*)': Not a directory", "cp: cannot create regular file '([^']*)': No such file or directory", "cp: cannot create regular file '([^']*)': Not a directory")

def match(command):
    if False:
        print('Hello World!')
    for pattern in patterns:
        if re.search(pattern, command.output):
            return True
    return False

def get_new_command(command):
    if False:
        print('Hello World!')
    for pattern in patterns:
        file = re.findall(pattern, command.output)
        if file:
            file = file[0]
            dir = file[0:file.rfind('/')]
            formatme = shell.and_('mkdir -p {}', '{}')
            return formatme.format(dir, command.script)