import platform, os
ON_WINDOWS = platform.system() == 'Windows'
ON_LINUX = not ON_WINDOWS
if ON_WINDOWS:
    from .localplatformwin import *
    from . import localplatformwin as localplatform
else:
    from .localplatformlinux import *
    from . import localplatformlinux as localplatform

def get_privileged_path() -> str:
    if False:
        while True:
            i = 10
    'Get path accessible by elevated user. Holds plugins, decky loader and decky loader configs'
    return localplatform.get_privileged_path()

def get_unprivileged_path() -> str:
    if False:
        return 10
    "Get path accessible by non-elevated user. Holds plugin configuration, plugin data and plugin logs. Externally referred to as the 'Homebrew' directory"
    return localplatform.get_unprivileged_path()

def get_unprivileged_user() -> str:
    if False:
        return 10
    'Get user that should own files made in unprivileged path'
    return localplatform.get_unprivileged_user()

def get_chown_plugin_path() -> bool:
    if False:
        return 10
    return os.getenv('CHOWN_PLUGIN_PATH', '1') == '1'

def get_server_host() -> str:
    if False:
        return 10
    return os.getenv('SERVER_HOST', '127.0.0.1')

def get_server_port() -> int:
    if False:
        return 10
    return int(os.getenv('SERVER_PORT', '1337'))

def get_live_reload() -> bool:
    if False:
        return 10
    return os.getenv('LIVE_RELOAD', '1') == '1'

def get_keep_systemd_service() -> bool:
    if False:
        while True:
            i = 10
    return os.getenv('KEEP_SYSTEMD_SERVICE', '0') == '1'

def get_log_level() -> int:
    if False:
        for i in range(10):
            print('nop')
    return {'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20, 'DEBUG': 10}[os.getenv('LOG_LEVEL', 'INFO')]

def get_selinux() -> bool:
    if False:
        i = 10
        return i + 15
    if ON_LINUX:
        from subprocess import check_output
        try:
            if check_output('getenforce').decode('ascii').strip('\n') == 'Enforcing':
                return True
        except FileNotFoundError:
            pass
    return False