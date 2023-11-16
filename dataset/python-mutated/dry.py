def match(command):
    if False:
        while True:
            i = 10
    split_command = command.script_parts
    return split_command and len(split_command) >= 2 and (split_command[0] == split_command[1])

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    return ' '.join(command.script_parts[1:])
priority = 900