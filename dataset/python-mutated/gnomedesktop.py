"""
GNOME implementations
"""
import logging
import re
import salt.utils.path
try:
    import pwd
    HAS_PWD = True
except ImportError:
    HAS_PWD = False
try:
    from gi.repository import Gio, GLib
    HAS_GLIB = True
except ImportError:
    HAS_GLIB = False
log = logging.getLogger(__name__)
__virtualname__ = 'gnome'
__func_alias__ = {'set_': 'set'}

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the Gio and Glib modules are available\n    '
    if HAS_PWD and HAS_GLIB:
        return __virtualname__
    return (False, 'The gnome_desktop execution module cannot be loaded: The Gio and GLib modules are not available')

class _GSettings:

    def __init__(self, user, schema, key):
        if False:
            while True:
                i = 10
        self.SCHEMA = schema
        self.KEY = key
        self.USER = user
        self.UID = None
        self.HOME = None

    @property
    def gsetting_command(self):
        if False:
            print('Hello World!')
        '\n        return the command to run the gsettings binary\n        '
        if salt.utils.path.which_bin(['dbus-run-session']):
            cmd = ['dbus-run-session', '--', 'gsettings']
        else:
            cmd = ['dbus-launch', '--exit-with-session', 'gsettings']
        return cmd

    def _get(self):
        if False:
            return 10
        '\n        get the value for user in gsettings\n\n        '
        user = self.USER
        try:
            uid = pwd.getpwnam(user).pw_uid
        except KeyError:
            log.info('User does not exist')
            return False
        cmd = self.gsetting_command + ['get', str(self.SCHEMA), str(self.KEY)]
        environ = {}
        environ['XDG_RUNTIME_DIR'] = '/run/user/{}'.format(uid)
        result = __salt__['cmd.run_all'](cmd, runas=user, env=environ, python_shell=False)
        if 'stdout' in result:
            if 'uint32' in result['stdout']:
                return re.sub('uint32 ', '', result['stdout'])
            else:
                return result['stdout']
        else:
            return False

    def _set(self, value):
        if False:
            i = 10
            return i + 15
        '\n        set the value for user in gsettings\n\n        '
        user = self.USER
        try:
            uid = pwd.getpwnam(user).pw_uid
        except KeyError:
            log.info('User does not exist')
            result = {}
            result['retcode'] = 1
            result['stdout'] = 'User {} does not exist'.format(user)
            return result
        cmd = self.gsetting_command + ['set', self.SCHEMA, self.KEY, value]
        environ = {}
        environ['XDG_RUNTIME_DIR'] = '/run/user/{}'.format(uid)
        result = __salt__['cmd.run_all'](cmd, runas=user, env=environ, python_shell=False)
        return result

def ping(**kwargs):
    if False:
        print('Hello World!')
    "\n    A test to ensure the GNOME module is loaded\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.ping user=<username>\n\n    "
    return True

def getIdleDelay(**kwargs):
    if False:
        return 10
    "\n    Return the current idle delay setting in seconds\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.getIdleDelay user=<username>\n\n    "
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.session', key='idle-delay')
    return _gsession._get()

def setIdleDelay(delaySeconds, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the current idle delay setting in seconds\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.setIdleDelay <seconds> user=<username>\n\n    "
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.session', key='idle-delay')
    return _gsession._set(delaySeconds)

def getClockFormat(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the current clock format, either 12h or 24h format.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.getClockFormat user=<username>\n\n    "
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.interface', key='clock-format')
    return _gsession._get()

def setClockFormat(clockFormat, **kwargs):
    if False:
        print('Hello World!')
    "\n    Set the clock format, either 12h or 24h format.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.setClockFormat <12h|24h> user=<username>\n\n    "
    if clockFormat != '12h' and clockFormat != '24h':
        return False
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.interface', key='clock-format')
    return _gsession._set(clockFormat)

def getClockShowDate(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the current setting, if the date is shown in the clock\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.getClockShowDate user=<username>\n\n    "
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.interface', key='clock-show-date')
    return _gsession._get()

def setClockShowDate(kvalue, **kwargs):
    if False:
        print('Hello World!')
    "\n    Set whether the date is visible in the clock\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.setClockShowDate <True|False> user=<username>\n\n    "
    if kvalue is not True and kvalue is not False:
        return False
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.interface', key='clock-show-date')
    return _gsession._set(kvalue)

def getIdleActivation(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get whether the idle activation is enabled\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.getIdleActivation user=<username>\n\n    "
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.screensaver', key='idle-activation-enabled')
    return _gsession._get()

def setIdleActivation(kvalue, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set whether the idle activation is enabled\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.setIdleActivation <True|False> user=<username>\n\n    "
    if kvalue is not True and kvalue is not False:
        return False
    _gsession = _GSettings(user=kwargs.get('user'), schema='org.gnome.desktop.screensaver', key='idle-activation-enabled')
    return _gsession._set(kvalue)

def get(schema=None, key=None, user=None, **kwargs):
    if False:
        return 10
    "\n    Get key in a particular GNOME schema\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.get user=<username> schema=org.gnome.desktop.screensaver key=idle-activation-enabled\n\n    "
    _gsession = _GSettings(user=user, schema=schema, key=key)
    return _gsession._get()

def set_(schema=None, key=None, user=None, value=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Set key in a particular GNOME schema\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' gnome.set user=<username> schema=org.gnome.desktop.screensaver key=idle-activation-enabled value=False\n\n    "
    _gsession = _GSettings(user=user, schema=schema, key=key)
    return _gsession._set(value)