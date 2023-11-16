"""
This module allows you to install certificates into the windows certificate
manager.

.. code-block:: bash

    salt '*' certutil.add_store salt://cert.cer "TrustedPublisher"
"""
import logging
import os
import re
import salt.utils.platform
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'certutil'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only works on Windows\n    '
    if salt.utils.platform.is_windows():
        return __virtualname__
    return (False, 'Module win_certutil: module only works on Windows systems.')

def get_cert_serial(cert_file, saltenv='base'):
    if False:
        return 10
    "\n    Get the serial number of a certificate file\n\n    cert_file (str):\n        The certificate file to find the serial for. Can be a local file or a\n        a file on the file server (``salt://``)\n\n    Returns:\n        str: The serial number of the certificate if found, otherwise None\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' certutil.get_cert_serial <certificate name>\n    "
    cert_file = __salt__['cp.cache_file'](cert_file, saltenv)
    if not os.path.exists(cert_file):
        msg = 'cert_file not found: {}'.format(cert_file)
        raise CommandExecutionError(msg)
    cmd = 'certutil.exe -silent -verify "{}"'.format(cert_file)
    out = __salt__['cmd.run'](cmd)
    matches = re.search(':\\s*(\\w*)\\r\\n\\r\\n', out)
    if matches is not None:
        return matches.groups()[0].strip()
    else:
        return None

def get_stored_cert_serials(store):
    if False:
        return 10
    "\n    Get all of the certificate serials in the specified store\n\n    store (str):\n        The store to get all the certificate serials from\n\n    Returns:\n        list: A list of serial numbers found, or an empty list if none found\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' certutil.get_stored_cert_serials <store>\n    "
    cmd = 'certutil.exe -store "{}"'.format(store)
    out = __salt__['cmd.run'](cmd)
    matches = re.findall('={16}\\r\\n.*:\\s*(\\w*)\\r\\n', out)
    return matches

def add_store(source, store, retcode=False, saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add the cert to the given Certificate Store\n\n    source (str):\n        The source certificate file. This is either the path to a local file or\n        a file from the file server in the form of ``salt://path/to/file``\n\n    store (str):\n        The certificate store to add the certificate to\n\n    retcode (bool):\n        If ``True``, return the retcode instead of stdout. Default is ``False``\n\n    saltenv (str):\n        The salt environment to use. This is ignored if the path is local\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' certutil.add_store salt://cert.cer TrustedPublisher\n        salt '*' certutil.add_store C:\\path\\to\\local.cer TrustedPublisher\n    "
    source = __salt__['cp.cache_file'](source, saltenv)
    if not os.path.exists(source):
        msg = 'cert_file not found: {}'.format(source)
        raise CommandExecutionError(msg)
    cmd = 'certutil.exe -addstore {} "{}"'.format(store, source)
    if retcode:
        return __salt__['cmd.retcode'](cmd)
    else:
        return __salt__['cmd.run'](cmd)

def del_store(source, store, retcode=False, saltenv='base'):
    if False:
        print('Hello World!')
    "\n    Delete the cert from the given Certificate Store\n\n    source (str):\n        The source certificate file. This is either the path to a local file or\n        a file from the file server in the form of ``salt://path/to/file``\n\n    store (str):\n        The certificate store to delete the certificate from\n\n    retcode (bool):\n        If ``True``, return the retcode instead of stdout. Default is ``False``\n\n    saltenv (str):\n        The salt environment to use. This is ignored if the path is local\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' certutil.del_store salt://cert.cer TrustedPublisher\n        salt '*' certutil.del_store C:\\path\\to\\local.cer TrustedPublisher\n    "
    source = __salt__['cp.cache_file'](source, saltenv)
    if not os.path.exists(source):
        msg = 'cert_file not found: {}'.format(source)
        raise CommandExecutionError(msg)
    serial = get_cert_serial(source)
    cmd = 'certutil.exe -delstore {} "{}"'.format(store, serial)
    if retcode:
        return __salt__['cmd.retcode'](cmd)
    else:
        return __salt__['cmd.run'](cmd)