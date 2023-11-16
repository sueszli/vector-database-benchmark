"""
Support IPMI commands over LAN. This module does not talk to the local
systems hardware through IPMI drivers. It uses a python module `pyghmi`.

:depends: Python module pyghmi.
    You can install pyghmi using pip:

    .. code-block:: bash

        pip install pyghmi

:configuration: The following configuration defaults can be
    define (pillar or config files):

    .. code-block:: python

        ipmi.config:
            api_host: 127.0.0.1
            api_user: admin
            api_pass: apassword
            api_port: 623
            api_kg: None

    Usage can override the config defaults:

    .. code-block:: bash

            salt-call ipmi.get_user api_host=myipmienabled.system
                                    api_user=admin api_pass=pass
                                    uid=1
"""
IMPORT_ERR = None
try:
    from pyghmi.ipmi import command
    from pyghmi.ipmi.private import session
except Exception as ex:
    IMPORT_ERR = str(ex)
__virtualname__ = 'ipmi'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    return (IMPORT_ERR is None, IMPORT_ERR)

def _get_config(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return configuration\n    '
    config = {'api_host': 'localhost', 'api_port': 623, 'api_user': 'admin', 'api_pass': '', 'api_kg': None, 'api_login_timeout': 2}
    if '__salt__' in globals():
        config_key = '{}.config'.format(__virtualname__)
        config.update(__salt__['config.get'](config_key, {}))
    for k in set(config) & set(kwargs):
        config[k] = kwargs[k]
    return config

class _IpmiCommand:
    o = None

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        config = _get_config(**kwargs)
        self.o = command.Command(bmc=config['api_host'], userid=config['api_user'], password=config['api_pass'], port=config['api_port'], kg=config['api_kg'])

    def __enter__(self):
        if False:
            print('Hello World!')
        return self.o

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        if self.o:
            self.o.ipmi_session.logout()

class _IpmiSession:
    o = None

    def _onlogon(self, response):
        if False:
            while True:
                i = 10
        if 'error' in response:
            raise Exception(response['error'])

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        config = _get_config(**kwargs)
        self.o = session.Session(bmc=config['api_host'], userid=config['api_user'], password=config['api_pass'], port=config['api_port'], kg=config['api_kg'], onlogon=self._onlogon)
        while not self.o.logged:
            self.o.maxtimeout = config['api_login_timeout']
            self.o.wait_for_rsp(timeout=1)
        self.o.maxtimeout = 5

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self.o

    def __exit__(self, type, value, traceback):
        if False:
            i = 10
            return i + 15
        if self.o:
            self.o.logout()

def raw_command(netfn, command, bridge_request=None, data=(), retry=True, delay_xmit=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Send raw ipmi command\n\n    This allows arbitrary IPMI bytes to be issued.  This is commonly used\n    for certain vendor specific commands.\n\n    :param netfn: Net function number\n    :param command: Command value\n    :param bridge_request: The target slave address and channel number for\n                        the bridge request.\n    :param data: Command data as a tuple or list\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    :returns: dict -- The response from IPMI device\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.raw_command netfn=0x06 command=0x46 data=[0x02]\n        # this will return the name of the user with id 2 in bytes\n    '
    with _IpmiSession(**kwargs) as s:
        r = s.raw_command(netfn=int(netfn), command=int(command), bridge_request=bridge_request, data=data, retry=retry, delay_xmit=delay_xmit)
        return r

def fast_connect_test(**kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Returns True if connection success.\n    This uses an aggressive timeout value!\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.fast_connect_test api_host=172.168.0.9\n    '
    try:
        if 'api_login_timeout' not in kwargs:
            kwargs['api_login_timeout'] = 0
        with _IpmiSession(**kwargs) as s:
            return True
    except Exception as e:
        return False
    return True

def set_channel_access(channel=14, access_update_mode='non_volatile', alerting=False, per_msg_auth=False, user_level_auth=False, access_mode='always', privilege_update_mode='non_volatile', privilege_level='administrator', **kwargs):
    if False:
        return 10
    "\n    Set channel access\n\n    :param channel: number [1:7]\n\n    :param access_update_mode:\n        - 'dont_change'  = don't set or change Channel Access\n        - 'non_volatile' = set non-volatile Channel Access\n        - 'volatile'     = set volatile (active) setting of Channel Access\n\n    :param alerting:\n        PEF Alerting Enable/Disable\n\n        - True  = enable PEF Alerting\n        - False = disable PEF Alerting on this channel\n          (Alert Immediate command can still be used to generate alerts)\n\n    :param per_msg_auth:\n        Per-message Authentication\n\n        - True  = enable\n        - False = disable Per-message Authentication. [Authentication required to\n          activate any session on this channel, but authentication not\n          used on subsequent packets for the session.]\n\n    :param user_level_auth:\n        User Level Authentication Enable/Disable\n\n        - True  = enable User Level Authentication. All User Level commands are\n          to be authenticated per the Authentication Type that was\n          negotiated when the session was activated.\n        - False = disable User Level Authentication. Allow User Level commands to\n          be executed without being authenticated.\n          If the option to disable User Level Command authentication is\n          accepted, the BMC will accept packets with Authentication Type\n          set to None if they contain user level commands.\n          For outgoing packets, the BMC returns responses with the same\n          Authentication Type that was used for the request.\n\n    :param access_mode:\n        Access Mode for IPMI messaging (PEF Alerting is enabled/disabled\n        separately from IPMI messaging)\n\n        - disabled = disabled for IPMI messaging\n        - pre_boot = pre-boot only channel only available when system is\n          in a powered down state or in BIOS prior to start of boot.\n        - always   = channel always available regardless of system mode.\n          BIOS typically dedicates the serial connection to the BMC.\n        - shared   = same as always available, but BIOS typically leaves the\n          serial port available for software use.\n\n    :param privilege_update_mode:\n        Channel Privilege Level Limit. This value sets the maximum privilege\n        level that can be accepted on the specified channel.\n\n        - dont_change  = don't set or change channel Privilege Level Limit\n        - non_volatile = non-volatile Privilege Level Limit according\n        - volatile     = volatile setting of Privilege Level Limit\n\n    :param privilege_level:\n        Channel Privilege Level Limit\n\n        - reserved      = unused\n        - callback\n        - user\n        - operator\n        - administrator\n        - proprietary   = used by OEM\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.set_channel_access privilege_level='administrator'\n    "
    with _IpmiCommand(**kwargs) as s:
        return s.set_channel_access(channel, access_update_mode, alerting, per_msg_auth, user_level_auth, access_mode, privilege_update_mode, privilege_level)

def get_channel_access(channel=14, read_mode='non_volatile', **kwargs):
    if False:
        return 10
    "\n    :param kwargs:api_host='127.0.0.1' api_user='admin' api_pass='example' api_port=623\n\n    :param channel: number [1:7]\n\n    :param read_mode:\n        - non_volatile  = get non-volatile Channel Access\n        - volatile      = get present volatile (active) setting of Channel Access\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    Return Data\n\n        A Python dict with the following keys/values:\n\n        .. code-block:: python\n\n            {\n                alerting:\n                per_msg_auth:\n                user_level_auth:\n                access_mode:{ (ONE OF)\n                    0: 'disabled',\n                    1: 'pre_boot',\n                    2: 'always',\n                    3: 'shared'\n                }\n                privilege_level: { (ONE OF)\n                    1: 'callback',\n                    2: 'user',\n                    3: 'operator',\n                    4: 'administrator',\n                    5: 'proprietary',\n                }\n            }\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_channel_access channel=1\n    "
    with _IpmiCommand(**kwargs) as s:
        return s.get_channel_access(channel)

def get_channel_info(channel=14, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Get channel info\n\n    :param channel: number [1:7]\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    Return Data\n        channel session supports\n\n        .. code-block:: none\n\n                - no_session: channel is session-less\n                - single: channel is single-session\n                - multi: channel is multi-session\n                - auto: channel is session-based (channel could alternate between\n                    single- and multi-session operation, as can occur with a\n                    serial/modem channel that supports connection mode auto-detect)\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_channel_info\n    '
    with _IpmiCommand(**kwargs) as s:
        return s.get_channel_info(channel)

def set_user_access(uid, channel=14, callback=True, link_auth=True, ipmi_msg=True, privilege_level='administrator', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set user access\n\n    :param uid: user number [1:16]\n\n    :param channel: number [1:7]\n\n    :param callback:\n        User Restricted to Callback\n\n        - False = User Privilege Limit is determined by the User Privilege Limit\n          parameter, below, for both callback and non-callback connections.\n        - True  = User Privilege Limit is determined by the User Privilege Limit\n          parameter for callback connections, but is restricted to Callback\n          level for non-callback connections. Thus, a user can only initiate\n          a Callback when they 'call in' to the BMC, but once the callback\n          connection has been made, the user could potentially establish a\n          session as an Operator.\n\n    :param link_auth: User Link authentication enable/disable (used to enable\n        whether this user's name and password information will be used for link\n        authentication, e.g. PPP CHAP) for the given channel. Link\n        authentication itself is a global setting for the channel and is\n        enabled/disabled via the serial/modem configuration parameters.\n\n    :param ipmi_msg: User IPMI Messaging: (used to enable/disable whether\n        this user's name and password information will be used for IPMI\n        Messaging. In this case, 'IPMI Messaging' refers to the ability to\n        execute generic IPMI commands that are not associated with a\n        particular payload type. For example, if IPMI Messaging is disabled for\n        a user, but that user is enabled for activating the SOL\n        payload type, then IPMI commands associated with SOL and session\n        management, such as Get SOL Configuration Parameters and Close Session\n        are available, but generic IPMI commands such as Get SEL Time are\n        unavailable.)\n\n    :param privilege_level:\n        User Privilege Limit. (Determines the maximum privilege level that the\n        user is allowed to switch to on the specified channel.)\n\n        - callback\n        - user\n        - operator\n        - administrator\n        - proprietary\n        - no_access\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.set_user_access uid=2 privilege_level='operator'\n    "
    with _IpmiCommand(**kwargs) as s:
        return s.set_user_access(uid, channel, callback, link_auth, ipmi_msg, privilege_level)

def get_user_access(uid, channel=14, **kwargs):
    if False:
        return 10
    '\n    Get user access\n\n    :param uid: user number [1:16]\n    :param channel: number [1:7]\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    Return Data\n\n    .. code-block:: none\n\n        channel_info:\n            - max_user_count = maximum number of user IDs on this channel\n            - enabled_users = count of User ID slots presently in use\n            - users_with_fixed_names = count of user IDs with fixed names\n        access:\n            - callback\n            - link_auth\n            - ipmi_msg\n            - privilege_level: [reserved, callback, user, operator\n                               administrator, proprietary, no_access]\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_user_access uid=2\n    '
    with _IpmiCommand(**kwargs) as s:
        return s.get_user_access(uid, channel=channel)

def set_user_name(uid, name, **kwargs):
    if False:
        return 10
    "\n    Set user name\n\n    :param uid: user number [1:16]\n    :param name: username (limit of 16bytes)\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.set_user_name uid=2 name='steverweber'\n    "
    with _IpmiCommand(**kwargs) as s:
        return s.set_user_name(uid, name)

def get_user_name(uid, return_none_on_error=True, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Get user name\n\n    :param uid: user number [1:16]\n    :param return_none_on_error: return None on error\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_user_name uid=2\n    '
    with _IpmiCommand(**kwargs) as s:
        return s.get_user_name(uid, return_none_on_error=True)

def set_user_password(uid, mode='set_password', password=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Set user password and (modes)\n\n    :param uid: id number of user.  see: get_names_uid()['name']\n\n    :param mode:\n        - disable       = disable user connections\n        - enable        = enable user connections\n        - set_password  = set or ensure password\n        - test_password = test password is correct\n    :param password: max 16 char string\n        (optional when mode is [disable or enable])\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    :return:\n        True on success\n        when mode = test_password, return False on bad password\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call ipmi.set_user_password api_host=127.0.0.1 api_user=admin api_pass=pass\n                                         uid=1 password=newPass\n        salt-call ipmi.set_user_password uid=1 mode=enable\n    "
    with _IpmiCommand(**kwargs) as s:
        s.set_user_password(uid, mode='set_password', password=password)
    return True

def get_health(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Get Summarize health\n\n    This provides a summary of the health of the managed system.\n    It additionally provides an iterable list of reasons for\n    warning, critical, or failed assessments.\n\n    good health: {'badreadings': [], 'health': 0}\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_health api_host=127.0.0.1 api_user=admin api_pass=pass\n    "
    with _IpmiCommand(**kwargs) as s:
        return s.get_health()

def get_power(**kwargs):
    if False:
        print('Hello World!')
    "\n    Get current power state\n\n    The response, if successful, should contain 'powerstate' key and\n    either 'on' or 'off' to indicate current state.\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_power api_host=127.0.0.1 api_user=admin api_pass=pass\n    "
    with _IpmiCommand(**kwargs) as s:
        return s.get_power()['powerstate']

def get_sensor_data(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get sensor readings\n\n    Iterates sensor reading objects\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_sensor_data api_host=127.0.0.1 api_user=admin api_pass=pass\n    '
    import ast
    with _IpmiCommand(**kwargs) as s:
        data = {}
        for reading in s.get_sensor_data():
            if reading:
                r = ast.literal_eval(repr(reading))
                data[r.pop('name')] = r
    return data

def get_bootdev(**kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Get current boot device override information.\n\n    Provides the current requested boot device.  Be aware that not all IPMI\n    devices support this.  Even in BMCs that claim to, occasionally the\n    BIOS or UEFI fail to honor it. This is usually only applicable to the\n    next reboot.\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_bootdev api_host=127.0.0.1 api_user=admin api_pass=pass\n    '
    with _IpmiCommand(**kwargs) as s:
        return s.get_bootdev()

def set_power(state='power_on', wait=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Request power state change\n\n    :param name:\n        * power_on -- system turn on\n        * power_off -- system turn off (without waiting for OS)\n        * shutdown -- request OS proper shutdown\n        * reset -- reset (without waiting for OS)\n        * boot -- If system is off, then 'on', else 'reset'\n\n    :param ensure: If (bool True), do not return until system actually completes\n                requested state change for 300 seconds.\n                If a non-zero (int), adjust the wait time to the\n                requested number of seconds\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    :returns: dict -- A dict describing the response retrieved\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.set_power state=shutdown wait=True\n    "
    if state is True or state == 'power_on':
        state = 'on'
    if state is False or state == 'power_off':
        state = 'off'
    with _IpmiCommand(**kwargs) as s:
        return s.set_power(state, wait=wait)

def set_bootdev(bootdev='default', persist=False, uefiboot=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Set boot device to use on next reboot\n\n    :param bootdev:\n        - network: Request network boot\n        - hd: Boot from hard drive\n        - safe: Boot from hard drive, requesting \'safe mode\'\n        - optical: boot from CD/DVD/BD drive\n        - setup: Boot into setup utility\n        - default: remove any IPMI directed boot device\n          request\n\n    :param persist: If true, ask that system firmware use this device\n                    beyond next boot.  Be aware many systems do not honor\n                    this\n\n    :param uefiboot: If true, request UEFI boot explicitly.  Strictly\n                    speaking, the spec suggests that if not set, the system\n                    should BIOS boot and offers no "don\'t care" option.\n                    In practice, this flag not being set does not preclude\n                    UEFI boot on any system I\'ve encountered.\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    :returns: dict or True -- If callback is not provided, the response\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.set_bootdev bootdev=network persist=True\n    '
    with _IpmiCommand(**kwargs) as s:
        return s.set_bootdev(bootdev)

def set_identify(on=True, duration=600, **kwargs):
    if False:
        return 10
    '\n    Request identify light\n\n    Request the identify light to turn off, on for a duration,\n    or on indefinitely.  Other than error exceptions,\n\n    :param on: Set to True to force on or False to force off\n    :param duration: Set if wanting to request turn on for a duration\n                    in seconds, None = indefinitely.\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.set_identify\n    '
    with _IpmiCommand(**kwargs) as s:
        return s.set_identify(on=on, duration=duration)

def get_channel_max_user_count(channel=14, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Get max users in channel\n\n    :param channel: number [1:7]\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n    :return: int -- often 16\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_channel_max_user_count\n    '
    access = get_user_access(channel=channel, uid=1, **kwargs)
    return access['channel_info']['max_user_count']

def get_user(uid, channel=14, **kwargs):
    if False:
        print('Hello World!')
    '\n    Get user from uid and access on channel\n\n    :param uid: user number [1:16]\n    :param channel: number [1:7]\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    Return Data\n\n    .. code-block:: none\n\n        name: (str)\n        uid: (int)\n        channel: (int)\n        access:\n            - callback (bool)\n            - link_auth (bool)\n            - ipmi_msg (bool)\n            - privilege_level: (str)[callback, user, operatorm administrator,\n                                    proprietary, no_access]\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_user uid=2\n    '
    name = get_user_name(uid, **kwargs)
    access = get_user_access(uid, channel, **kwargs)
    data = {'name': name, 'uid': uid, 'channel': channel, 'access': access['access']}
    return data

def get_users(channel=14, **kwargs):
    if False:
        return 10
    '\n    get list of users and access information\n\n    :param channel: number [1:7]\n\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    :return:\n        - name: (str)\n        - uid: (int)\n        - channel: (int)\n        - access:\n            - callback (bool)\n            - link_auth (bool)\n            - ipmi_msg (bool)\n            - privilege_level: (str)[callback, user, operatorm administrator,\n              proprietary, no_access]\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.get_users api_host=172.168.0.7\n    '
    with _IpmiCommand(**kwargs) as c:
        return c.get_users(channel)

def create_user(uid, name, password, channel=14, callback=False, link_auth=True, ipmi_msg=True, privilege_level='administrator', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    create/ensure a user is created with provided settings.\n\n    :param privilege_level:\n        User Privilege Limit. (Determines the maximum privilege level that\n        the user is allowed to switch to on the specified channel.)\n        * callback\n        * user\n        * operator\n        * administrator\n        * proprietary\n        * no_access\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.create_user uid=2 name=steverweber api_host=172.168.0.7 api_pass=nevertell\n    '
    with _IpmiCommand(**kwargs) as c:
        return c.create_user(uid, name, password, channel, callback, link_auth, ipmi_msg, privilege_level)

def user_delete(uid, channel=14, **kwargs):
    if False:
        return 10
    '\n    Delete user (helper)\n\n    :param uid: user number [1:16]\n    :param channel: number [1:7]\n    :param kwargs:\n        - api_host=127.0.0.1\n        - api_user=admin\n        - api_pass=example\n        - api_port=623\n        - api_kg=None\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-call ipmi.user_delete uid=2\n    '
    with _IpmiCommand(**kwargs) as c:
        return c.user_delete(uid, channel)