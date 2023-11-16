def match(command):
    if False:
        return 10
    return 'manage.py' in command.script and 'migrate' in command.script and ('--merge: will just attempt the migration' in command.output)

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    return u'{} --merge'.format(command.script)