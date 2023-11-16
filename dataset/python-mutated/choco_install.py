from thefuck.utils import for_app, which

@for_app('choco', 'cinst')
def match(command):
    if False:
        return 10
    return (command.script.startswith('choco install') or 'cinst' in command.script_parts) and 'Installing the following packages' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    for script_part in command.script_parts:
        if script_part not in ['choco', 'cinst', 'install'] and (not script_part.startswith('-')) and ('=' not in script_part) and ('/' not in script_part):
            return command.script.replace(script_part, script_part + '.install')
    return []
enabled_by_default = bool(which('choco')) or bool(which('cinst'))