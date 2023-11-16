import re
from thefuck.specific.sudo import sudo_support

@sudo_support
def match(command):
    if False:
        while True:
            i = 10
    return 'rm' in command.script and 'is a directory' in command.output.lower()

@sudo_support
def get_new_command(command):
    if False:
        print('Hello World!')
    arguments = '-rf'
    if 'hdfs' in command.script:
        arguments = '-r'
    return re.sub('\\brm (.*)', 'rm ' + arguments + ' \\1', command.script)