def match(command):
    if False:
        print('Hello World!')
    return command.script == 'test.py' and 'not found' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    return 'pytest'
priority = 900