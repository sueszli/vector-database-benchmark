import re
from thefuck.utils import for_app
from thefuck.specific.sudo import sudo_support
from thefuck.shells import shell

@sudo_support
@for_app('cd')
def match(command):
    if False:
        while True:
            i = 10
    return command.script.startswith('cd ') and any(('no such file or directory' in command.output.lower(), "cd: can't cd to" in command.output.lower(), 'does not exist' in command.output.lower()))

@sudo_support
def get_new_command(command):
    if False:
        print('Hello World!')
    repl = shell.and_('mkdir -p \\1', 'cd \\1')
    return re.sub('^cd (.*)', repl, command.script)