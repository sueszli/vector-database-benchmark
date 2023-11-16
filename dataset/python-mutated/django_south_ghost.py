def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'manage.py' in command.script and 'migrate' in command.script and ('or pass --delete-ghost-migrations' in command.output)

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return u'{} --delete-ghost-migrations'.format(command.script)