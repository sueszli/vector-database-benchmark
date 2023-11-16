"""Functions to check if certain parts of InvenTree are ready."""
import os
import sys

def isInTestMode():
    if False:
        print('Hello World!')
    'Returns True if the database is in testing mode.'
    return 'test' in sys.argv

def isImportingData():
    if False:
        return 10
    "Returns True if the database is currently importing data, e.g. 'loaddata' command is performed."
    return 'loaddata' in sys.argv

def isRunningMigrations():
    if False:
        for i in range(10):
            print('nop')
    'Return True if the database is currently running migrations.'
    return 'migrate' in sys.argv or 'makemigrations' in sys.argv

def isInMainThread():
    if False:
        for i in range(10):
            print('nop')
    'Django runserver starts two processes, one for the actual dev server and the other to reload the application.\n\n    - The RUN_MAIN env is set in that case. However if --noreload is applied, this variable\n    is not set because there are no different threads.\n    '
    if 'runserver' in sys.argv and '--noreload' not in sys.argv:
        return os.environ.get('RUN_MAIN', None) == 'true'
    return True

def canAppAccessDatabase(allow_test: bool=False, allow_plugins: bool=False, allow_shell: bool=False):
    if False:
        i = 10
        return i + 15
    "Returns True if the apps.py file can access database records.\n\n    There are some circumstances where we don't want the ready function in apps.py\n    to touch the database\n    "
    excluded_commands = ['flush', 'loaddata', 'dumpdata', 'check', 'createsuperuser', 'wait_for_db', 'prerender', 'rebuild_models', 'rebuild_thumbnails', 'makemessages', 'compilemessages', 'backup', 'dbbackup', 'mediabackup', 'restore', 'dbrestore', 'mediarestore']
    if not allow_shell:
        excluded_commands.append('shell')
    if not allow_test:
        excluded_commands.append('test')
    if not allow_plugins:
        excluded_commands.extend(['makemigrations', 'migrate', 'collectstatic'])
    for cmd in excluded_commands:
        if cmd in sys.argv:
            return False
    return True

def isPluginRegistryLoaded():
    if False:
        return 10
    "Ensures that the plugin registry is already loaded.\n\n    The plugin registry reloads all apps onetime after starting if there are AppMixin plugins,\n    so that the discovered AppConfigs are added to Django. This triggers the ready function of\n    AppConfig to execute twice. Add this check to prevent from running two times.\n\n    Note: All apps using this check need to be registered after the plugins app in settings.py\n\n    Returns: 'False' if the registry has not fully loaded the plugins yet.\n    "
    from plugin import registry
    return registry.plugins_loaded