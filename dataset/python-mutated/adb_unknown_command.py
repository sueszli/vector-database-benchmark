from thefuck.utils import is_app, get_closest, replace_argument
_ADB_COMMANDS = ('backup', 'bugreport', 'connect', 'devices', 'disable-verity', 'disconnect', 'enable-verity', 'emu', 'forward', 'get-devpath', 'get-serialno', 'get-state', 'install', 'install-multiple', 'jdwp', 'keygen', 'kill-server', 'logcat', 'pull', 'push', 'reboot', 'reconnect', 'restore', 'reverse', 'root', 'run-as', 'shell', 'sideload', 'start-server', 'sync', 'tcpip', 'uninstall', 'unroot', 'usb', 'wait-for')

def match(command):
    if False:
        for i in range(10):
            print('nop')
    return is_app(command, 'adb') and command.output.startswith('Android Debug Bridge version')

def get_new_command(command):
    if False:
        print('Hello World!')
    for (idx, arg) in enumerate(command.script_parts[1:]):
        if not arg[0] == '-' and (not command.script_parts[idx] in ('-s', '-H', '-P', '-L')):
            adb_cmd = get_closest(arg, _ADB_COMMANDS)
            return replace_argument(command.script, arg, adb_cmd)