from thefuck.specific.sudo import sudo_support
enabled_by_default = False

@sudo_support
def match(command):
    if False:
        i = 10
        return i + 15
    return command.script_parts and {'rm', '/'}.issubset(command.script_parts) and ('--no-preserve-root' not in command.script) and ('--no-preserve-root' in command.output)

@sudo_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    return u'{} --no-preserve-root'.format(command.script)