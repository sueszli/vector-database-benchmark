"""
znc - An advanced IRC bouncer

.. versionadded:: 2014.7.0

Provides an interface to basic ZNC functionality
"""
import hashlib
import logging
import os.path
import random
import signal
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load the module if znc is installed\n    '
    if salt.utils.path.which('znc'):
        return 'znc'
    return (False, 'Module znc: znc binary not found')

def _makepass(password, hasher='sha256'):
    if False:
        print('Hello World!')
    '\n    Create a znc compatible hashed password\n    '
    if hasher == 'sha256':
        h = hashlib.sha256(password)
    elif hasher == 'md5':
        h = hashlib.md5(password)
    else:
        return NotImplemented
    c = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.,:;/*-+_()'
    r = {'Method': h.name, 'Salt': ''.join((random.SystemRandom().choice(c) for x in range(20)))}
    h.update(r['Salt'])
    r['Hash'] = h.hexdigest()
    return r

def buildmod(*modules):
    if False:
        return 10
    "\n    Build module using znc-buildmod\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' znc.buildmod module.cpp [...]\n    "
    missing = [module for module in modules if not os.path.exists(module)]
    if missing:
        return 'Error: The file ({}) does not exist.'.format(', '.join(missing))
    cmd = ['znc-buildmod']
    cmd.extend(modules)
    out = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    return out[-1]

def dumpconf():
    if False:
        print('Hello World!')
    "\n    Write the active configuration state to config file\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' znc.dumpconf\n    "
    return __salt__['ps.pkill']('znc', signal=signal.SIGUSR1)

def rehashconf():
    if False:
        while True:
            i = 10
    "\n    Rehash the active configuration state from config file\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' znc.rehashconf\n    "
    return __salt__['ps.pkill']('znc', signal=signal.SIGHUP)

def version():
    if False:
        while True:
            i = 10
    "\n    Return server version from znc --version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' znc.version\n    "
    cmd = ['znc', '--version']
    out = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    ret = out[0].split(' - ')
    return ret[0]