"""Fixes error for commands containing one or more occurrences of the shell
prompt symbol '$'.

This usually happens when commands are copied from documentations
including them in their code blocks.

Example:
> $ git clone https://github.com/nvbn/thefuck.git
bash: $: command not found...
"""
import re

def match(command):
    if False:
        for i in range(10):
            print('nop')
    return '$: command not found' in command.output and re.search('^[\\s]*\\$ [\\S]+', command.script) is not None

def get_new_command(command):
    if False:
        print('Hello World!')
    return command.script.lstrip('$ ')