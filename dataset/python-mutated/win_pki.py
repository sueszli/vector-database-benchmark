"""
Microsoft certificate management via the PKI Client PowerShell module.
https://technet.microsoft.com/en-us/itpro/powershell/windows/pkiclient/pkiclient

The PKI Client PowerShell module is only available on Windows 8+ and Windows
Server 2012+.
https://technet.microsoft.com/en-us/library/hh848636(v=wps.620).aspx

:platform:      Windows

:depends:
    - PowerShell 4
    - PKI Client Module (Windows 8+ / Windows Server 2012+)

.. versionadded:: 2016.11.0
"""
import ast
import logging
import os
import salt.utils.json
import salt.utils.platform
import salt.utils.powershell
import salt.utils.versions
from salt.exceptions import SaltInvocationError
_DEFAULT_CONTEXT = 'LocalMachine'
_DEFAULT_FORMAT = 'cer'
_DEFAULT_STORE = 'My'
_LOG = logging.getLogger(__name__)
__virtualname__ = 'win_pki'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Requires Windows\n    Requires Windows 8+ / Windows Server 2012+\n    Requires PowerShell\n    Requires PKI Client PowerShell module installed.\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Only available on Windows Systems')
    if salt.utils.versions.version_cmp(__grains__['osversion'], '6.2.9200') == -1:
        return (False, 'Only available on Windows 8+ / Windows Server 2012 +')
    if not __salt__['cmd.shell_info']('powershell')['installed']:
        return (False, 'Powershell not available')
    if not salt.utils.powershell.module_exists('PKI'):
        return (False, 'PowerShell PKI module not available')
    return __virtualname__

def _cmd_run(cmd, as_json=False):
    if False:
        print('Hello World!')
    '\n    Ensure that the Pki module is loaded, and convert to and extract data from\n    Json as needed.\n    '
    cmd_full = ['Import-Module -Name PKI; ']
    if as_json:
        cmd_full.append('ConvertTo-Json -Compress -Depth 4 -InputObject @({})'.format(cmd))
    else:
        cmd_full.append(cmd)
    cmd_ret = __salt__['cmd.run_all'](''.join(cmd_full), shell='powershell', python_shell=True)
    if cmd_ret['retcode'] != 0:
        _LOG.error('Unable to execute command: %s\nError: %s', cmd, cmd_ret['stderr'])
    if as_json:
        try:
            items = salt.utils.json.loads(cmd_ret['stdout'], strict=False)
            return items
        except ValueError:
            _LOG.error('Unable to parse return data as Json.')
    return cmd_ret['stdout']

def _validate_cert_path(name):
    if False:
        return 10
    '\n    Ensure that the certificate path, as determind from user input, is valid.\n    '
    cmd = "Test-Path -Path '{}'".format(name)
    if not ast.literal_eval(_cmd_run(cmd=cmd)):
        raise SaltInvocationError('Invalid path specified: {}'.format(name))

def _validate_cert_format(name):
    if False:
        print('Hello World!')
    '\n    Ensure that the certificate format, as determind from user input, is valid.\n    '
    cert_formats = ['cer', 'pfx']
    if name not in cert_formats:
        raise SaltInvocationError("Invalid certificate format '{}' specified. Valid formats: {}".format(name, cert_formats))

def get_stores():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the certificate location contexts and their corresponding stores.\n\n    :return: A dictionary of the certificate location contexts and stores.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_pki.get_stores\n    "
    ret = dict()
    cmd = "Get-ChildItem -Path 'Cert:\\' | Select-Object LocationName, StoreNames"
    items = _cmd_run(cmd=cmd, as_json=True)
    for item in items:
        ret[item['LocationName']] = list()
        for store in item['StoreNames']:
            ret[item['LocationName']].append(store)
    return ret

def get_certs(context=_DEFAULT_CONTEXT, store=_DEFAULT_STORE):
    if False:
        print('Hello World!')
    "\n    Get the available certificates in the given store.\n\n    :param str context: The name of the certificate store location context.\n    :param str store: The name of the certificate store.\n\n    :return: A dictionary of the certificate thumbprints and properties.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_pki.get_certs\n    "
    ret = dict()
    cmd = list()
    blacklist_keys = ['DnsNameList']
    store_path = 'Cert:\\{}\\{}'.format(context, store)
    _validate_cert_path(name=store_path)
    cmd.append("Get-ChildItem -Path '{}' | Select-Object".format(store_path))
    cmd.append(' DnsNameList, SerialNumber, Subject, Thumbprint, Version')
    items = _cmd_run(cmd=''.join(cmd), as_json=True)
    for item in items:
        cert_info = dict()
        for key in item:
            if key not in blacklist_keys:
                cert_info[key.lower()] = item[key]
        names = item.get('DnsNameList', None)
        if isinstance(names, list):
            cert_info['dnsnames'] = [name.get('Unicode') for name in names]
        else:
            cert_info['dnsnames'] = []
        ret[item['Thumbprint']] = cert_info
    return ret

def get_cert_file(name, cert_format=_DEFAULT_FORMAT, password=''):
    if False:
        i = 10
        return i + 15
    "\n    Get the details of the certificate file.\n\n    :param str name: The filesystem path of the certificate file.\n    :param str cert_format: The certificate format. Specify 'cer' for X.509, or\n        'pfx' for PKCS #12.\n    :param str password: The password of the certificate. Only applicable to pfx\n        format. Note that if used interactively, the password will be seen by all minions.\n        To protect the password, use a state and get the password from pillar.\n\n    :return: A dictionary of the certificate thumbprints and properties.\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_pki.get_cert_file name='C:\\certs\\example.cer'\n    "
    ret = dict()
    cmd = list()
    blacklist_keys = ['DnsNameList']
    cert_format = cert_format.lower()
    _validate_cert_format(name=cert_format)
    if not name or not os.path.isfile(name):
        _LOG.error('Path is not present: %s', name)
        return ret
    if cert_format == 'pfx':
        if password:
            cmd.append('$CertObject = New-Object')
            cmd.append(' System.Security.Cryptography.X509Certificates.X509Certificate2;')
            cmd.append(" $CertObject.Import('{}'".format(name))
            cmd.append(",'{}'".format(password))
            cmd.append(",'DefaultKeySet') ; $CertObject")
            cmd.append(' | Select-Object DnsNameList, SerialNumber, Subject, Thumbprint, Version')
        else:
            cmd.append("Get-PfxCertificate -FilePath '{}'".format(name))
            cmd.append(' | Select-Object DnsNameList, SerialNumber, Subject, Thumbprint, Version')
    else:
        cmd.append('$CertObject = New-Object')
        cmd.append(' System.Security.Cryptography.X509Certificates.X509Certificate2;')
        cmd.append(" $CertObject.Import('{}'); $CertObject".format(name))
        cmd.append(' | Select-Object DnsNameList, SerialNumber, Subject, Thumbprint, Version')
    items = _cmd_run(cmd=''.join(cmd), as_json=True)
    for item in items:
        for key in item:
            if key not in blacklist_keys:
                ret[key.lower()] = item[key]
        ret['dnsnames'] = [name['Unicode'] for name in item['DnsNameList']]
    if ret:
        _LOG.debug('Certificate thumbprint obtained successfully: %s', name)
    else:
        _LOG.error('Unable to obtain certificate thumbprint: %s', name)
    return ret

def import_cert(name, cert_format=_DEFAULT_FORMAT, context=_DEFAULT_CONTEXT, store=_DEFAULT_STORE, exportable=True, password='', saltenv='base'):
    if False:
        return 10
    "\n    Import the certificate file into the given certificate store.\n\n    :param str name: The path of the certificate file to import.\n    :param str cert_format: The certificate format. Specify 'cer' for X.509, or\n        'pfx' for PKCS #12.\n    :param str context: The name of the certificate store location context.\n    :param str store: The name of the certificate store.\n    :param bool exportable: Mark the certificate as exportable. Only applicable\n        to pfx format.\n    :param str password: The password of the certificate. Only applicable to pfx\n        format. Note that if used interactively, the password will be seen by all minions.\n        To protect the password, use a state and get the password from pillar.\n    :param str saltenv: The environment the file resides in.\n\n    :return: A boolean representing whether all changes succeeded.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_pki.import_cert name='salt://cert.cer'\n    "
    cmd = list()
    thumbprint = None
    store_path = 'Cert:\\{}\\{}'.format(context, store)
    cert_format = cert_format.lower()
    _validate_cert_format(name=cert_format)
    cached_source_path = __salt__['cp.cache_file'](name, saltenv)
    if not cached_source_path:
        _LOG.error('Unable to get cached copy of file: %s', name)
        return False
    if password:
        cert_props = get_cert_file(name=cached_source_path, cert_format=cert_format, password=password)
    else:
        cert_props = get_cert_file(name=cached_source_path, cert_format=cert_format)
    current_certs = get_certs(context=context, store=store)
    if cert_props['thumbprint'] in current_certs:
        _LOG.debug("Certificate thumbprint '%s' already present in store: %s", cert_props['thumbprint'], store_path)
        return True
    if cert_format == 'pfx':
        if password:
            cmd.append("$Password = ConvertTo-SecureString -String '{}'".format(password))
            cmd.append(' -AsPlainText -Force; ')
        else:
            cmd.append('$Password = New-Object System.Security.SecureString; ')
        cmd.append("Import-PfxCertificate -FilePath '{}'".format(cached_source_path))
        cmd.append(" -CertStoreLocation '{}'".format(store_path))
        cmd.append(' -Password $Password')
        if exportable:
            cmd.append(' -Exportable')
    else:
        cmd.append("Import-Certificate -FilePath '{}'".format(cached_source_path))
        cmd.append(" -CertStoreLocation '{}'".format(store_path))
    _cmd_run(cmd=''.join(cmd))
    new_certs = get_certs(context=context, store=store)
    for new_cert in new_certs:
        if new_cert not in current_certs:
            thumbprint = new_cert
    if thumbprint:
        _LOG.debug('Certificate imported successfully: %s', name)
        return True
    _LOG.error('Unable to import certificate: %s', name)
    return False

def export_cert(name, thumbprint, cert_format=_DEFAULT_FORMAT, context=_DEFAULT_CONTEXT, store=_DEFAULT_STORE, password=''):
    if False:
        print('Hello World!')
    "\n    Export the certificate to a file from the given certificate store.\n\n    :param str name: The destination path for the exported certificate file.\n    :param str thumbprint: The thumbprint value of the target certificate.\n    :param str cert_format: The certificate format. Specify 'cer' for X.509, or\n        'pfx' for PKCS #12.\n    :param str context: The name of the certificate store location context.\n    :param str store: The name of the certificate store.\n    :param str password: The password of the certificate. Only applicable to pfx\n        format. Note that if used interactively, the password will be seen by all minions.\n        To protect the password, use a state and get the password from pillar.\n\n    :return: A boolean representing whether all changes succeeded.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_pki.export_cert name='C:\\certs\\example.cer' thumbprint='AAA000'\n    "
    cmd = list()
    thumbprint = thumbprint.upper()
    cert_path = 'Cert:\\{}\\{}\\{}'.format(context, store, thumbprint)
    cert_format = cert_format.lower()
    _validate_cert_path(name=cert_path)
    _validate_cert_format(name=cert_format)
    if cert_format == 'pfx':
        if password:
            cmd.append("$Password = ConvertTo-SecureString -String '{}'".format(password))
            cmd.append(' -AsPlainText -Force; ')
        else:
            cmd.append('$Password = New-Object System.Security.SecureString; ')
        cmd.append("Export-PfxCertificate -Cert '{}' -FilePath '{}'".format(cert_path, name))
        cmd.append(' -Password $Password')
    else:
        cmd.append("Export-Certificate -Cert '{}' -FilePath '{}'".format(cert_path, name))
    cmd.append(" | Out-Null; Test-Path -Path '{}'".format(name))
    ret = ast.literal_eval(_cmd_run(cmd=''.join(cmd)))
    if ret:
        _LOG.debug('Certificate exported successfully: %s', name)
    else:
        _LOG.error('Unable to export certificate: %s', name)
    return ret

def test_cert(thumbprint, context=_DEFAULT_CONTEXT, store=_DEFAULT_STORE, untrusted_root=False, dns_name='', eku=''):
    if False:
        return 10
    "\n    Check the certificate for validity.\n\n    :param str thumbprint: The thumbprint value of the target certificate.\n    :param str context: The name of the certificate store location context.\n    :param str store: The name of the certificate store.\n    :param bool untrusted_root: Whether the root certificate is required to be\n        trusted in chain building.\n    :param str dns_name: The DNS name to verify as valid for the certificate.\n    :param str eku: The enhanced key usage object identifiers to verify for the\n        certificate chain.\n\n    :return: A boolean representing whether the certificate was considered\n        valid.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_pki.test_cert thumbprint='AAA000' dns_name='example.test'\n    "
    cmd = list()
    thumbprint = thumbprint.upper()
    cert_path = 'Cert:\\{}\\{}\\{}'.format(context, store, thumbprint)
    cmd.append("Test-Certificate -Cert '{}'".format(cert_path))
    _validate_cert_path(name=cert_path)
    if untrusted_root:
        cmd.append(' -AllowUntrustedRoot')
    if dns_name:
        cmd.append(" -DnsName '{}'".format(dns_name))
    if eku:
        cmd.append(" -EKU '{}'".format(eku))
    cmd.append(' -ErrorAction SilentlyContinue')
    return ast.literal_eval(_cmd_run(cmd=''.join(cmd)))

def remove_cert(thumbprint, context=_DEFAULT_CONTEXT, store=_DEFAULT_STORE):
    if False:
        print('Hello World!')
    "\n    Remove the certificate from the given certificate store.\n\n    :param str thumbprint: The thumbprint value of the target certificate.\n    :param str context: The name of the certificate store location context.\n    :param str store: The name of the certificate store.\n\n    :return: A boolean representing whether all changes succeeded.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_pki.remove_cert thumbprint='AAA000'\n    "
    thumbprint = thumbprint.upper()
    store_path = 'Cert:\\{}\\{}'.format(context, store)
    cert_path = '{}\\{}'.format(store_path, thumbprint)
    cmd = "Remove-Item -Path '{}'".format(cert_path)
    current_certs = get_certs(context=context, store=store)
    if thumbprint not in current_certs:
        _LOG.debug("Certificate '%s' already absent in store: %s", thumbprint, store_path)
        return True
    _validate_cert_path(name=cert_path)
    _cmd_run(cmd=cmd)
    new_certs = get_certs(context=context, store=store)
    if thumbprint in new_certs:
        _LOG.error('Unable to remove certificate: %s', cert_path)
        return False
    _LOG.debug('Certificate removed successfully: %s', cert_path)
    return True