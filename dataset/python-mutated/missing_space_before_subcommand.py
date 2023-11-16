from thefuck.utils import get_all_executables, memoize

@memoize
def _get_executable(script_part):
    if False:
        for i in range(10):
            print('nop')
    for executable in get_all_executables():
        if len(executable) > 1 and script_part.startswith(executable):
            return executable

def match(command):
    if False:
        print('Hello World!')
    return not command.script_parts[0] in get_all_executables() and _get_executable(command.script_parts[0])

def get_new_command(command):
    if False:
        return 10
    executable = _get_executable(command.script_parts[0])
    return command.script.replace(executable, u'{} '.format(executable), 1)
priority = 4000