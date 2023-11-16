from thefuck.utils import eager, get_closest, for_app

@for_app('fab')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return 'Warning: Command(s) not found:' in command.output

@eager
def _get_between(content, start, end=None):
    if False:
        while True:
            i = 10
    should_yield = False
    for line in content.split('\n'):
        if start in line:
            should_yield = True
            continue
        if end and end in line:
            return
        if should_yield and line:
            yield line.strip().split(' ')[0]

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    not_found_commands = _get_between(command.output, 'Warning: Command(s) not found:', 'Available commands:')
    possible_commands = _get_between(command.output, 'Available commands:')
    script = command.script
    for not_found in not_found_commands:
        fix = get_closest(not_found, possible_commands)
        script = script.replace(' {}'.format(not_found), ' {}'.format(fix))
    return script