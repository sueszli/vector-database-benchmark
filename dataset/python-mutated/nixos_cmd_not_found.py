import re
from thefuck.specific.nix import nix_available
from thefuck.shells import shell
regex = re.compile('nix-env -iA ([^\\s]*)')
enabled_by_default = nix_available

def match(command):
    if False:
        i = 10
        return i + 15
    return regex.findall(command.output)

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    name = regex.findall(command.output)[0]
    return shell.and_('nix-env -iA {}'.format(name), command.script)