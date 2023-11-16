from types import ModuleType
from thefuck.specific.apt import apt_available
from thefuck.utils import memoize, which
from thefuck.shells import shell
try:
    from CommandNotFound import CommandNotFound
    enabled_by_default = apt_available
    if isinstance(CommandNotFound, ModuleType):
        _get_packages = CommandNotFound.CommandNotFound().get_packages
    else:
        _get_packages = CommandNotFound().getPackages
except ImportError:
    enabled_by_default = False

def _get_executable(command):
    if False:
        i = 10
        return i + 15
    if command.script_parts[0] == 'sudo':
        return command.script_parts[1]
    else:
        return command.script_parts[0]

@memoize
def get_package(executable):
    if False:
        while True:
            i = 10
    try:
        packages = _get_packages(executable)
        return packages[0][0]
    except IndexError:
        return None

def match(command):
    if False:
        return 10
    if 'not found' in command.output or 'not installed' in command.output:
        executable = _get_executable(command)
        return not which(executable) and get_package(executable)
    else:
        return False

def get_new_command(command):
    if False:
        print('Hello World!')
    executable = _get_executable(command)
    name = get_package(executable)
    formatme = shell.and_('sudo apt-get install {}', '{}')
    return formatme.format(name, command.script)