"""
Support for modifying make.conf under Gentoo

"""
import salt.utils.data
import salt.utils.files

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on Gentoo\n    '
    if __grains__['os'] == 'Gentoo':
        return 'makeconf'
    return (False, 'The makeconf execution module cannot be loaded: only available on Gentoo systems.')

def _get_makeconf():
    if False:
        return 10
    '\n    Find the correct make.conf. Gentoo recently moved the make.conf\n    but still supports the old location, using the old location first\n    '
    old_conf = '/etc/make.conf'
    new_conf = '/etc/portage/make.conf'
    if __salt__['file.file_exists'](old_conf):
        return old_conf
    elif __salt__['file.file_exists'](new_conf):
        return new_conf

def _add_var(var, value):
    if False:
        i = 10
        return i + 15
    '\n    Add a new var to the make.conf. If using layman, the source line\n    for the layman make.conf needs to be at the very end of the\n    config. This ensures that the new var will be above the source\n    line.\n    '
    makeconf = _get_makeconf()
    layman = 'source /var/lib/layman/make.conf'
    fullvar = '{}="{}"'.format(var, value)
    if __salt__['file.contains'](makeconf, layman):
        cmd = ['sed', '-i', '/{}/ i\\{}'.format(layman.replace('/', '\\/'), fullvar), makeconf]
        __salt__['cmd.run'](cmd)
    else:
        __salt__['file.append'](makeconf, fullvar)

def set_var(var, value):
    if False:
        print('Hello World!')
    "\n    Set a variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_var 'LINGUAS' 'en'\n    "
    makeconf = _get_makeconf()
    old_value = get_var(var)
    if old_value is not None:
        __salt__['file.sed'](makeconf, '^{}=.*'.format(var), '{}="{}"'.format(var, value))
    else:
        _add_var(var, value)
    new_value = get_var(var)
    return {var: {'old': old_value, 'new': new_value}}

def remove_var(var):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove a variable from the make.conf\n\n    Return a dict containing the new value for the variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.remove_var 'LINGUAS'\n    "
    makeconf = _get_makeconf()
    old_value = get_var(var)
    if old_value is not None:
        __salt__['file.sed'](makeconf, '^{}=.*'.format(var), '')
    new_value = get_var(var)
    return {var: {'old': old_value, 'new': new_value}}

def append_var(var, value):
    if False:
        i = 10
        return i + 15
    "\n    Add to or create a new variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.append_var 'LINGUAS' 'en'\n    "
    makeconf = _get_makeconf()
    old_value = get_var(var)
    if old_value is not None:
        appended_value = '{} {}'.format(old_value, value)
        __salt__['file.sed'](makeconf, '^{}=.*'.format(var), '{}="{}"'.format(var, appended_value))
    else:
        _add_var(var, value)
    new_value = get_var(var)
    return {var: {'old': old_value, 'new': new_value}}

def trim_var(var, value):
    if False:
        print('Hello World!')
    "\n    Remove a value from a variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.trim_var 'LINGUAS' 'en'\n    "
    makeconf = _get_makeconf()
    old_value = get_var(var)
    if old_value is not None:
        __salt__['file.sed'](makeconf, value, '', limit=var)
    new_value = get_var(var)
    return {var: {'old': old_value, 'new': new_value}}

def get_var(var):
    if False:
        print('Hello World!')
    "\n    Get the value of a variable in make.conf\n\n    Return the value of the variable or None if the variable is not in\n    make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_var 'LINGUAS'\n    "
    makeconf = _get_makeconf()
    with salt.utils.files.fopen(makeconf) as fn_:
        conf_file = salt.utils.data.decode(fn_.readlines())
    for line in conf_file:
        if line.startswith(var):
            ret = line.split('=', 1)[1]
            if '"' in ret:
                ret = ret.split('"')[1]
            elif '#' in ret:
                ret = ret.split('#')[0]
            ret = ret.strip()
            return ret
    return None

def var_contains(var, value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Verify if variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.var_contains 'LINGUAS' 'en'\n    "
    setval = get_var(var)
    value = value.replace('\\', '')
    if setval is None:
        return False
    return value in setval.split()

def set_cflags(value):
    if False:
        while True:
            i = 10
    "\n    Set the CFLAGS variable\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_cflags '-march=native -O2 -pipe'\n    "
    return set_var('CFLAGS', value)

def get_cflags():
    if False:
        return 10
    "\n    Get the value of CFLAGS variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_cflags\n    "
    return get_var('CFLAGS')

def append_cflags(value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add to or create a new CFLAGS in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.append_cflags '-pipe'\n    "
    return append_var('CFLAGS', value)

def trim_cflags(value):
    if False:
        while True:
            i = 10
    "\n    Remove a value from CFLAGS variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.trim_cflags '-pipe'\n    "
    return trim_var('CFLAGS', value)

def cflags_contains(value):
    if False:
        return 10
    "\n    Verify if CFLAGS variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.cflags_contains '-pipe'\n    "
    return var_contains('CFLAGS', value)

def set_cxxflags(value):
    if False:
        print('Hello World!')
    "\n    Set the CXXFLAGS variable\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_cxxflags '-march=native -O2 -pipe'\n    "
    return set_var('CXXFLAGS', value)

def get_cxxflags():
    if False:
        print('Hello World!')
    "\n    Get the value of CXXFLAGS variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_cxxflags\n    "
    return get_var('CXXFLAGS')

def append_cxxflags(value):
    if False:
        i = 10
        return i + 15
    "\n    Add to or create a new CXXFLAGS in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.append_cxxflags '-pipe'\n    "
    return append_var('CXXFLAGS', value)

def trim_cxxflags(value):
    if False:
        while True:
            i = 10
    "\n    Remove a value from CXXFLAGS variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.trim_cxxflags '-pipe'\n    "
    return trim_var('CXXFLAGS', value)

def cxxflags_contains(value):
    if False:
        i = 10
        return i + 15
    "\n    Verify if CXXFLAGS variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.cxxflags_contains '-pipe'\n    "
    return var_contains('CXXFLAGS', value)

def set_chost(value):
    if False:
        i = 10
        return i + 15
    "\n    Set the CHOST variable\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_chost 'x86_64-pc-linux-gnu'\n    "
    return set_var('CHOST', value)

def get_chost():
    if False:
        i = 10
        return i + 15
    "\n    Get the value of CHOST variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_chost\n    "
    return get_var('CHOST')

def chost_contains(value):
    if False:
        i = 10
        return i + 15
    "\n    Verify if CHOST variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.chost_contains 'x86_64-pc-linux-gnu'\n    "
    return var_contains('CHOST', value)

def set_makeopts(value):
    if False:
        return 10
    "\n    Set the MAKEOPTS variable\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_makeopts '-j3'\n    "
    return set_var('MAKEOPTS', value)

def get_makeopts():
    if False:
        return 10
    "\n    Get the value of MAKEOPTS variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_makeopts\n    "
    return get_var('MAKEOPTS')

def append_makeopts(value):
    if False:
        while True:
            i = 10
    "\n    Add to or create a new MAKEOPTS in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.append_makeopts '-j3'\n    "
    return append_var('MAKEOPTS', value)

def trim_makeopts(value):
    if False:
        i = 10
        return i + 15
    "\n    Remove a value from MAKEOPTS variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.trim_makeopts '-j3'\n    "
    return trim_var('MAKEOPTS', value)

def makeopts_contains(value):
    if False:
        while True:
            i = 10
    "\n    Verify if MAKEOPTS variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.makeopts_contains '-j3'\n    "
    return var_contains('MAKEOPTS', value)

def set_emerge_default_opts(value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the EMERGE_DEFAULT_OPTS variable\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_emerge_default_opts '--jobs'\n    "
    return set_var('EMERGE_DEFAULT_OPTS', value)

def get_emerge_default_opts():
    if False:
        i = 10
        return i + 15
    "\n    Get the value of EMERGE_DEFAULT_OPTS variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_emerge_default_opts\n    "
    return get_var('EMERGE_DEFAULT_OPTS')

def append_emerge_default_opts(value):
    if False:
        while True:
            i = 10
    "\n    Add to or create a new EMERGE_DEFAULT_OPTS in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.append_emerge_default_opts '--jobs'\n    "
    return append_var('EMERGE_DEFAULT_OPTS', value)

def trim_emerge_default_opts(value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove a value from EMERGE_DEFAULT_OPTS variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.trim_emerge_default_opts '--jobs'\n    "
    return trim_var('EMERGE_DEFAULT_OPTS', value)

def emerge_default_opts_contains(value):
    if False:
        return 10
    "\n    Verify if EMERGE_DEFAULT_OPTS variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.emerge_default_opts_contains '--jobs'\n    "
    return var_contains('EMERGE_DEFAULT_OPTS', value)

def set_gentoo_mirrors(value):
    if False:
        while True:
            i = 10
    "\n    Set the GENTOO_MIRRORS variable\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_gentoo_mirrors 'http://distfiles.gentoo.org'\n    "
    return set_var('GENTOO_MIRRORS', value)

def get_gentoo_mirrors():
    if False:
        return 10
    "\n    Get the value of GENTOO_MIRRORS variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_gentoo_mirrors\n    "
    return get_var('GENTOO_MIRRORS')

def append_gentoo_mirrors(value):
    if False:
        return 10
    "\n    Add to or create a new GENTOO_MIRRORS in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.append_gentoo_mirrors 'http://distfiles.gentoo.org'\n    "
    return append_var('GENTOO_MIRRORS', value)

def trim_gentoo_mirrors(value):
    if False:
        print('Hello World!')
    "\n    Remove a value from GENTOO_MIRRORS variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.trim_gentoo_mirrors 'http://distfiles.gentoo.org'\n    "
    return trim_var('GENTOO_MIRRORS', value)

def gentoo_mirrors_contains(value):
    if False:
        i = 10
        return i + 15
    "\n    Verify if GENTOO_MIRRORS variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.gentoo_mirrors_contains 'http://distfiles.gentoo.org'\n    "
    return var_contains('GENTOO_MIRRORS', value)

def set_sync(value):
    if False:
        while True:
            i = 10
    "\n    Set the SYNC variable\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.set_sync 'rsync://rsync.namerica.gentoo.org/gentoo-portage'\n    "
    return set_var('SYNC', value)

def get_sync():
    if False:
        return 10
    "\n    Get the value of SYNC variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_sync\n    "
    return get_var('SYNC')

def sync_contains(value):
    if False:
        return 10
    "\n    Verify if SYNC variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.sync_contains 'rsync://rsync.namerica.gentoo.org/gentoo-portage'\n    "
    return var_contains('SYNC', value)

def get_features():
    if False:
        print('Hello World!')
    "\n    Get the value of FEATURES variable in the make.conf\n\n    Return the value of the variable or None if the variable is\n    not in the make.conf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.get_features\n    "
    return get_var('FEATURES')

def append_features(value):
    if False:
        return 10
    "\n    Add to or create a new FEATURES in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.append_features 'webrsync-gpg'\n    "
    return append_var('FEATURES', value)

def trim_features(value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove a value from FEATURES variable in the make.conf\n\n    Return a dict containing the new value for variable::\n\n        {'<variable>': {'old': '<old-value>',\n                        'new': '<new-value>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.trim_features 'webrsync-gpg'\n    "
    return trim_var('FEATURES', value)

def features_contains(value):
    if False:
        i = 10
        return i + 15
    "\n    Verify if FEATURES variable contains a value in make.conf\n\n    Return True if value is set for var\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' makeconf.features_contains 'webrsync-gpg'\n    "
    return var_contains('FEATURES', value)