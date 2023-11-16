from thefuck.utils import for_app, get_close_matches, replace_command
import re

def _get_failed_lifecycle(command):
    if False:
        for i in range(10):
            print('nop')
    return re.search('\\[ERROR\\] Unknown lifecycle phase "(.+)"', command.output)

def _getavailable_lifecycles(command):
    if False:
        return 10
    return re.search('Available lifecycle phases are: (.+) -> \\[Help 1\\]', command.output)

@for_app('mvn')
def match(command):
    if False:
        return 10
    failed_lifecycle = _get_failed_lifecycle(command)
    available_lifecycles = _getavailable_lifecycles(command)
    return available_lifecycles and failed_lifecycle

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    failed_lifecycle = _get_failed_lifecycle(command)
    available_lifecycles = _getavailable_lifecycles(command)
    if available_lifecycles and failed_lifecycle:
        selected_lifecycle = get_close_matches(failed_lifecycle.group(1), available_lifecycles.group(1).split(', '))
        return replace_command(command, failed_lifecycle.group(1), selected_lifecycle)
    else:
        return []