def match(command):
    if False:
        i = 10
        return i + 15
    if command.script_parts[0] == 'cs':
        return True

def get_new_command(command):
    if False:
        return 10
    return 'cd' + ''.join(command.script[2:])
priority = 900