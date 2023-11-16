from thefuck.specific.sudo import sudo_support

@sudo_support
def match(command):
    if False:
        i = 10
        return i + 15
    return command.script_parts and command.script_parts[0].endswith('.py') and ('Permission denied' in command.output or 'command not found' in command.output)

@sudo_support
def get_new_command(command):
    if False:
        while True:
            i = 10
    return 'python ' + command.script