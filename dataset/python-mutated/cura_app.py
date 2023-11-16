import sys
if '' in sys.path:
    sys.path.remove('')
import argparse
import faulthandler
import os
if sys.platform != 'linux':
    os.environ['QT_PLUGIN_PATH'] = ''
    os.environ['QML2_IMPORT_PATH'] = ''
    os.environ['QT_OPENGL_DLL'] = ''
from PyQt6.QtNetwork import QSslConfiguration, QSslSocket
from UM.Platform import Platform
from cura import ApplicationMetadata
from cura.ApplicationMetadata import CuraAppName
from cura.CrashHandler import CrashHandler
try:
    import sentry_sdk
    with_sentry_sdk = True
except ImportError:
    with_sentry_sdk = False
parser = argparse.ArgumentParser(prog='cura', add_help=False)
parser.add_argument('--debug', action='store_true', default=False, help='Turn on the debug mode by setting this option.')
known_args = vars(parser.parse_known_args()[0])
if with_sentry_sdk:
    sentry_env = 'unknown'
    if hasattr(sys, 'frozen'):
        sentry_env = 'production'
    if ApplicationMetadata.CuraVersion == 'master':
        sentry_env = 'development'
    elif 'beta' in ApplicationMetadata.CuraVersion or 'BETA' in ApplicationMetadata.CuraVersion:
        sentry_env = 'beta'
    elif 'alpha' in ApplicationMetadata.CuraVersion or 'ALPHA' in ApplicationMetadata.CuraVersion:
        sentry_env = 'alpha'
    try:
        if ApplicationMetadata.CuraVersion.split('.')[2] == '99':
            sentry_env = 'nightly'
    except IndexError:
        pass
    ignore_errors = [KeyboardInterrupt, MemoryError]
    try:
        sentry_sdk.init('https://5034bf0054fb4b889f82896326e79b13@sentry.io/1821564', before_send=CrashHandler.sentryBeforeSend, environment=sentry_env, release='cura%s' % ApplicationMetadata.CuraVersion, default_integrations=False, max_breadcrumbs=300, server_name='cura', ignore_errors=ignore_errors)
    except Exception:
        with_sentry_sdk = False
if not known_args['debug']:

    def get_cura_dir_path():
        if False:
            while True:
                i = 10
        if Platform.isWindows():
            appdata_path = os.getenv('APPDATA')
            if not appdata_path:
                appdata_path = '.'
            return os.path.join(appdata_path, CuraAppName)
        elif Platform.isLinux():
            return os.path.expanduser('~/.local/share/' + CuraAppName)
        elif Platform.isOSX():
            return os.path.expanduser('~/Library/Logs/' + CuraAppName)
    if hasattr(sys, 'frozen') and 'cli' not in os.path.basename(sys.argv[0]).lower():
        dirpath = get_cura_dir_path()
        os.makedirs(dirpath, exist_ok=True)
        sys.stdout = open(os.path.join(dirpath, 'stdout.log'), 'w', encoding='utf-8')
        sys.stderr = open(os.path.join(dirpath, 'stderr.log'), 'w', encoding='utf-8')
if Platform.isLinux():
    try:
        import ctypes
        from ctypes.util import find_library
        libGL = find_library('GL')
        ctypes.CDLL(libGL, ctypes.RTLD_GLOBAL)
    except:
        pass
if Platform.isWindows() and hasattr(sys, 'frozen'):
    try:
        del os.environ['PYTHONPATH']
    except KeyError:
        pass
if Platform.isLinux() and hasattr(sys, 'frozen'):
    os.chdir(os.path.expanduser('~'))
if 'PYTHONPATH' in os.environ.keys():
    PYTHONPATH = os.environ['PYTHONPATH'].split(os.pathsep)
    PYTHONPATH.reverse()
    for PATH in PYTHONPATH:
        PATH_real = os.path.realpath(PATH)
        if PATH_real in sys.path:
            sys.path.remove(PATH_real)
        sys.path.insert(1, PATH_real)

def exceptHook(hook_type, value, traceback):
    if False:
        return 10
    from cura.CrashHandler import CrashHandler
    from cura.CuraApplication import CuraApplication
    has_started = False
    if CuraApplication.Created:
        has_started = CuraApplication.getInstance().started
    from PyQt6.QtWidgets import QApplication
    if CuraApplication.Created:
        _crash_handler = CrashHandler(hook_type, value, traceback, has_started)
        if CuraApplication.splash is not None:
            CuraApplication.splash.close()
        if not has_started:
            CuraApplication.getInstance().removePostedEvents(None)
            _crash_handler.early_crash_dialog.show()
            sys.exit(CuraApplication.getInstance().exec())
        else:
            _crash_handler.show()
    else:
        application = QApplication(sys.argv)
        application.removePostedEvents(None)
        _crash_handler = CrashHandler(hook_type, value, traceback, has_started)
        if CuraApplication.splash is not None:
            CuraApplication.splash.close()
        _crash_handler.early_crash_dialog.show()
        sys.exit(application.exec())
sys.excepthook = exceptHook
if sys.stderr and (not sys.stderr.closed):
    faulthandler.enable(file=sys.stderr, all_threads=True)
elif sys.stdout and (not sys.stdout.closed):
    faulthandler.enable(file=sys.stdout, all_threads=True)
from cura.CuraApplication import CuraApplication
if Platform.isOSX() and getattr(sys, 'frozen', False):
    old_env = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
    search_path = os.path.join(CuraApplication.getInstallPrefix(), 'MacOS')
    path_list = old_env.split(':')
    if search_path not in path_list:
        path_list.append(search_path)
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(path_list)
    import trimesh.exchange.load
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = old_env
if Platform.isLinux() and getattr(sys, 'frozen', False):
    old_env = os.environ.get('LD_LIBRARY_PATH', '')
    search_path = os.path.join(CuraApplication.getInstallPrefix(), 'bin')
    path_list = old_env.split(':')
    if search_path not in path_list:
        path_list.append(search_path)
    os.environ['LD_LIBRARY_PATH'] = ':'.join(path_list)
    import trimesh.exchange.load
    os.environ['LD_LIBRARY_PATH'] = old_env
if Platform.isLinux():
    os.environ['QT_QUICK_CONTROLS_STYLE'] = 'default'
if ApplicationMetadata.CuraDebugMode:
    ssl_conf = QSslConfiguration.defaultConfiguration()
    ssl_conf.setPeerVerifyMode(QSslSocket.PeerVerifyMode.VerifyNone)
    QSslConfiguration.setDefaultConfiguration(ssl_conf)
app = CuraApplication()
app.run()