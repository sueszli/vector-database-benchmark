from thefuck.utils import for_app

@for_app('brew', at_least=2)
def match(command):
    if False:
        print('Hello World!')
    return command.script_parts[1] in ['uninstall', 'rm', 'remove'] and 'brew uninstall --force' in command.output

def get_new_command(command):
    if False:
        return 10
    command_parts = command.script_parts[:]
    command_parts[1] = 'uninstall'
    command_parts.insert(2, '--force')
    return ' '.join(command_parts)