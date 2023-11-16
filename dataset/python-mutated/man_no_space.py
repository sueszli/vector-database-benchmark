def match(command):
    if False:
        print('Hello World!')
    return command.script.startswith(u'man') and u'command not found' in command.output.lower()

def get_new_command(command):
    if False:
        print('Hello World!')
    return u'man {}'.format(command.script[3:])
priority = 2000