"""
Install certificates into the keychain on Mac OS

.. versionadded:: 2016.3.0

"""
import logging
import re
import shlex
import salt.utils.platform
log = logging.getLogger(__name__)
__virtualname__ = 'keychain'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work on Mac OS\n    '
    if salt.utils.platform.is_darwin():
        return __virtualname__
    return (False, 'Only available on Mac OS systems with pipes')

def install(cert, password, keychain='/Library/Keychains/System.keychain', allow_any=False, keychain_password=None):
    if False:
        return 10
    "\n    Install a certificate\n\n    cert\n        The certificate to install\n\n    password\n        The password for the certificate being installed formatted in the way\n        described for openssl command in the PASS PHRASE ARGUMENTS section.\n\n        Note: The password given here will show up as plaintext in the job returned\n        info.\n\n    keychain\n        The keychain to install the certificate to, this defaults to\n        /Library/Keychains/System.keychain\n\n    allow_any\n        Allow any application to access the imported certificate without warning\n\n    keychain_password\n        If your keychain is likely to be locked pass the password and it will be unlocked\n        before running the import\n\n        Note: The password given here will show up as plaintext in the returned job\n        info.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.install test.p12 test123\n    "
    if keychain_password is not None:
        unlock_keychain(keychain, keychain_password)
    cmd = f'security import {cert} -P {password} -k {keychain}'
    if allow_any:
        cmd += ' -A'
    return __salt__['cmd.run'](cmd)

def uninstall(cert_name, keychain='/Library/Keychains/System.keychain', keychain_password=None):
    if False:
        return 10
    "\n    Uninstall a certificate from a keychain\n\n    cert_name\n        The name of the certificate to remove\n\n    keychain\n        The keychain to install the certificate to, this defaults to\n        /Library/Keychains/System.keychain\n\n    keychain_password\n        If your keychain is likely to be locked pass the password and it will be unlocked\n        before running the import\n\n        Note: The password given here will show up as plaintext in the returned job\n        info.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.install test.p12 test123\n    "
    if keychain_password is not None:
        unlock_keychain(keychain, keychain_password)
    cmd = f'security delete-certificate -c "{cert_name}" {keychain}'
    return __salt__['cmd.run'](cmd)

def list_certs(keychain='/Library/Keychains/System.keychain'):
    if False:
        while True:
            i = 10
    "\n    List all of the installed certificates\n\n    keychain\n        The keychain to install the certificate to, this defaults to\n        /Library/Keychains/System.keychain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.list_certs\n    "
    cmd = 'security find-certificate -a {} | grep -o "alis".*\\" | grep -o \'\\"[-A-Za-z0-9.:() ]*\\"\''.format(shlex.quote(keychain))
    out = __salt__['cmd.run'](cmd, python_shell=True)
    return out.replace('"', '').split('\n')

def get_friendly_name(cert, password):
    if False:
        while True:
            i = 10
    "\n    Get the friendly name of the given certificate\n\n    cert\n        The certificate to install\n\n    password\n        The password for the certificate being installed formatted in the way\n        described for openssl command in the PASS PHRASE ARGUMENTS section\n\n        Note: The password given here will show up as plaintext in the returned job\n        info.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.get_friendly_name /tmp/test.p12 test123\n    "
    cmd = 'openssl pkcs12 -in {} -passin pass:{} -info -nodes -nokeys 2> /dev/null | grep friendlyName:'.format(shlex.quote(cert), shlex.quote(password))
    out = __salt__['cmd.run'](cmd, python_shell=True)
    return out.replace('friendlyName: ', '').strip()

def get_default_keychain(user=None, domain='user'):
    if False:
        return 10
    "\n    Get the default keychain\n\n    user\n        The user to check the default keychain of\n\n    domain\n        The domain to use valid values are user|system|common|dynamic, the default is user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.get_default_keychain\n    "
    cmd = f'security default-keychain -d {domain}'
    return __salt__['cmd.run'](cmd, runas=user)

def set_default_keychain(keychain, domain='user', user=None):
    if False:
        while True:
            i = 10
    "\n    Set the default keychain\n\n    keychain\n        The location of the keychain to set as default\n\n    domain\n        The domain to use valid values are user|system|common|dynamic, the default is user\n\n    user\n        The user to set the default keychain as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.set_keychain /Users/fred/Library/Keychains/login.keychain\n    "
    cmd = f'security default-keychain -d {domain} -s {keychain}'
    return __salt__['cmd.run'](cmd, runas=user)

def unlock_keychain(keychain, password):
    if False:
        while True:
            i = 10
    "\n    Unlock the given keychain with the password\n\n    keychain\n        The keychain to unlock\n\n    password\n        The password to use to unlock the keychain.\n\n        Note: The password given here will show up as plaintext in the returned job\n        info.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.unlock_keychain /tmp/test.p12 test123\n    "
    cmd = f'security unlock-keychain -p {password} {keychain}'
    __salt__['cmd.run'](cmd)

def get_hash(name, password=None):
    if False:
        print('Hello World!')
    "\n    Returns the hash of a certificate in the keychain.\n\n    name\n        The name of the certificate (which you can get from keychain.get_friendly_name) or the\n        location of a p12 file.\n\n    password\n        The password that is used in the certificate. Only required if your passing a p12 file.\n        Note: This will be outputted to logs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' keychain.get_hash /tmp/test.p12 test123\n    "
    if '.p12' in name[-4:]:
        cmd = 'openssl pkcs12 -in {0} -passin pass:{1} -passout pass:{1}'.format(name, password)
    else:
        cmd = f'security find-certificate -c "{name}" -m -p'
    out = __salt__['cmd.run'](cmd)
    matches = re.search('-----BEGIN CERTIFICATE-----(.*)-----END CERTIFICATE-----', out, re.DOTALL | re.MULTILINE)
    if matches:
        return matches.group(1)
    else:
        return False