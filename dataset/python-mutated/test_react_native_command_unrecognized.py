import pytest
from io import BytesIO
from thefuck.rules.react_native_command_unrecognized import match, get_new_command
from thefuck.types import Command
output = "Unrecognized command '{}'".format
stdout = b'\nScanning 615 folders for symlinks in /home/nvbn/work/zcho/BookkaWebView/node_modules (6ms)\n\n  Usage: react-native [options] [command]\n\n\n  Options:\n\n    -V, --version  output the version number\n    -h, --help     output usage information\n\n\n  Commands:\n\n    start [options]                    starts the webserver\n    run-ios [options]                  builds your app and starts it on iOS simulator\n    run-android [options]              builds your app and starts it on a connected Android emulator or device\n    new-library [options]              generates a native library bridge\n    bundle [options]                   builds the javascript bundle for offline use\n    unbundle [options]                 builds javascript as "unbundle" for offline use\n    eject [options]                    Re-create the iOS and Android folders and native code\n    link [options] [packageName]       links all native dependencies (updates native build files)\n    unlink [options] <packageName>     unlink native dependency\n    install [options] <packageName>    install and link native dependencies\n    uninstall [options] <packageName>  uninstall and unlink native dependencies\n    upgrade [options]                  upgrade your app\'s template files to the latest version; run this after updating the react-native version in your package.json and running npm install\n    log-android [options]              starts adb logcat\n    log-ios [options]                  starts iOS device syslog tail\n'

@pytest.mark.parametrize('command', [Command('react-native star', output('star')), Command('react-native android-logs', output('android-logs'))])
def test_match(command):
    if False:
        for i in range(10):
            print('nop')
    assert match(command)

@pytest.mark.parametrize('command', [Command('gradle star', output('star')), Command('react-native start', '')])
def test_not_match(command):
    if False:
        print('Hello World!')
    assert not match(command)

@pytest.mark.parametrize('command, result', [(Command('react-native star', output('star')), 'react-native start'), (Command('react-native logsandroid -f', output('logsandroid')), 'react-native log-android -f')])
def test_get_new_command(mocker, command, result):
    if False:
        for i in range(10):
            print('nop')
    patch = mocker.patch('thefuck.rules.react_native_command_unrecognized.Popen')
    patch.return_value.stdout = BytesIO(stdout)
    assert get_new_command(command)[0] == result