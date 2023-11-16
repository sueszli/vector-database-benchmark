"""Suggest creating symbolic link if hard link is not allowed.

Example:
> ln barDir barLink
ln: ‘barDir’: hard link not allowed for directory

--> ln -s barDir barLink
"""
import re
from thefuck.specific.sudo import sudo_support

@sudo_support
def match(command):
    if False:
        i = 10
        return i + 15
    return command.output.endswith('hard link not allowed for directory') and command.script_parts[0] == 'ln'

@sudo_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return re.sub('^ln ', 'ln -s ', command.script)