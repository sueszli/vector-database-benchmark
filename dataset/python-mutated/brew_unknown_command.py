import os
import re
from thefuck.utils import get_closest, replace_command
from thefuck.specific.brew import get_brew_path_prefix, brew_available
BREW_CMD_PATH = '/Homebrew/Library/Homebrew/cmd'
TAP_PATH = '/Homebrew/Library/Taps'
TAP_CMD_PATH = '/%s/%s/cmd'
enabled_by_default = brew_available

def _get_brew_commands(brew_path_prefix):
    if False:
        while True:
            i = 10
    'To get brew default commands on local environment'
    brew_cmd_path = brew_path_prefix + BREW_CMD_PATH
    return [name[:-3] for name in os.listdir(brew_cmd_path) if name.endswith(('.rb', '.sh'))]

def _get_brew_tap_specific_commands(brew_path_prefix):
    if False:
        i = 10
        return i + 15
    "To get tap's specific commands\n    https://github.com/Homebrew/homebrew/blob/master/Library/brew.rb#L115"
    commands = []
    brew_taps_path = brew_path_prefix + TAP_PATH
    for user in _get_directory_names_only(brew_taps_path):
        taps = _get_directory_names_only(brew_taps_path + '/%s' % user)
        taps = (tap for tap in taps if tap.startswith('homebrew-'))
        for tap in taps:
            tap_cmd_path = brew_taps_path + TAP_CMD_PATH % (user, tap)
            if os.path.isdir(tap_cmd_path):
                commands += (name.replace('brew-', '').replace('.rb', '') for name in os.listdir(tap_cmd_path) if _is_brew_tap_cmd_naming(name))
    return commands

def _is_brew_tap_cmd_naming(name):
    if False:
        for i in range(10):
            print('nop')
    return name.startswith('brew-') and name.endswith('.rb')

def _get_directory_names_only(path):
    if False:
        print('Hello World!')
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def _brew_commands():
    if False:
        return 10
    brew_path_prefix = get_brew_path_prefix()
    if brew_path_prefix:
        try:
            return _get_brew_commands(brew_path_prefix) + _get_brew_tap_specific_commands(brew_path_prefix)
        except OSError:
            pass
    return ['info', 'home', 'options', 'install', 'uninstall', 'search', 'list', 'update', 'upgrade', 'pin', 'unpin', 'doctor', 'create', 'edit', 'cask']

def match(command):
    if False:
        for i in range(10):
            print('nop')
    is_proper_command = 'brew' in command.script and 'Unknown command' in command.output
    if is_proper_command:
        broken_cmd = re.findall('Error: Unknown command: ([a-z]+)', command.output)[0]
        return bool(get_closest(broken_cmd, _brew_commands()))
    return False

def get_new_command(command):
    if False:
        print('Hello World!')
    broken_cmd = re.findall('Error: Unknown command: ([a-z]+)', command.output)[0]
    return replace_command(command, broken_cmd, _brew_commands())