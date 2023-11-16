import re
from thefuck.utils import for_app
from thefuck.specific.brew import brew_available
enabled_by_default = brew_available

def _get_suggestions(str):
    if False:
        i = 10
        return i + 15
    suggestions = str.replace(' or ', ', ').split(', ')
    return suggestions

@for_app('brew', at_least=2)
def match(command):
    if False:
        return 10
    is_proper_command = 'install' in command.script and 'No available formula' in command.output and ('Did you mean' in command.output)
    return is_proper_command

def get_new_command(command):
    if False:
        i = 10
        return i + 15
    matcher = re.search('Warning: No available formula with the name "(?:[^"]+)". Did you mean (.+)\\?', command.output)
    suggestions = _get_suggestions(matcher.group(1))
    return ['brew install ' + formula for formula in suggestions]