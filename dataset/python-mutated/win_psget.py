"""
Module for managing PowerShell through PowerShellGet (PSGet)

:depends:
    - PowerShell 5.0
    - PSGet

Support for PowerShell
"""
import logging
import xml.etree.ElementTree
import salt.utils.platform
import salt.utils.versions
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'psget'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the system module of the kernel is Windows\n    '
    if not salt.utils.platform.is_windows():
        log.debug('Module PSGet: Only available on Windows systems')
        return (False, 'Module PSGet: Only available on Windows systems')
    powershell_info = __salt__['cmd.shell_info']('powershell')
    if not powershell_info['installed']:
        log.debug('Module PSGet: Requires PowerShell')
        return (False, 'Module PSGet: Requires PowerShell')
    if salt.utils.versions.compare(powershell_info['version'], '<', '5.0'):
        log.debug('Module PSGet: Requires PowerShell 5 or newer')
        return (False, 'Module PSGet: Requires PowerShell 5 or newer.')
    return __virtualname__

def _ps_xml_to_dict(parent, dic=None):
    if False:
        while True:
            i = 10
    '\n    Formats powershell Xml to a dict.\n    Note: This _ps_xml_to_dict is not perfect with powershell Xml.\n    '
    if dic is None:
        dic = {}
    for child in parent:
        if list(child):
            new_dic = _ps_xml_to_dict(child, {})
            if 'Name' in new_dic:
                dic[new_dic['Name']] = new_dic
            else:
                try:
                    dic[[name for (ps_type, name) in child.items() if ps_type == 'Type'][0]] = new_dic
                except IndexError:
                    dic[child.text] = new_dic
        else:
            for (xml_type, name) in child.items():
                if xml_type == 'Name':
                    dic[name] = child.text
    return dic

def _pshell(cmd, cwd=None, depth=2):
    if False:
        print('Hello World!')
    '\n    Execute the desired powershell command and ensure that it returns data\n    in Xml format and load that into python\n    '
    cmd = '{} | ConvertTo-Xml -Depth {} -As "stream"'.format(cmd, depth)
    log.debug('DSC: %s', cmd)
    results = __salt__['cmd.run_all'](cmd, shell='powershell', cwd=cwd, python_shell=True)
    if 'pid' in results:
        del results['pid']
    if 'retcode' not in results or results['retcode'] != 0:
        raise CommandExecutionError('Issue executing powershell {}'.format(cmd), info=results)
    try:
        ret = _ps_xml_to_dict(xml.etree.ElementTree.fromstring(results['stdout'].encode('utf-8')))
    except xml.etree.ElementTree.ParseError:
        results['stdout'] = results['stdout'][:1000] + '. . .'
        raise CommandExecutionError('No XML results from powershell', info=results)
    return ret

def bootstrap():
    if False:
        i = 10
        return i + 15
    "\n    Make sure that nuget-anycpu.exe is installed.\n    This will download the official nuget-anycpu.exe from the internet.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.bootstrap\n    "
    cmd = 'Get-PackageProvider -Name NuGet -ForceBootstrap | Select Name, Version, ProviderPath'
    ret = _pshell(cmd, depth=1)
    return ret

def avail_modules(desc=False):
    if False:
        i = 10
        return i + 15
    "\n    List available modules in registered Powershell module repositories.\n\n    :param desc: If ``True``, the verbose description will be returned.\n    :type  desc: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.avail_modules\n        salt 'win01' psget.avail_modules desc=True\n    "
    cmd = 'Find-Module | Select Name, Description'
    modules = _pshell(cmd, depth=1)
    names = []
    if desc:
        names = {}
    for key in modules:
        module = modules[key]
        if desc:
            names[module['Name']] = module['Description']
            continue
        names.append(module['Name'])
    return names

def list_modules(desc=False):
    if False:
        while True:
            i = 10
    "\n    List currently installed PSGet Modules on the system.\n\n    :param desc: If ``True``, the verbose description will be returned.\n    :type  desc: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.list_modules\n        salt 'win01' psget.list_modules desc=True\n    "
    cmd = 'Get-InstalledModule'
    modules = _pshell(cmd)
    names = []
    if desc:
        names = {}
    for key in modules:
        module = modules[key]
        if desc:
            names[module['Name']] = module
            continue
        names.append(module['Name'])
    return names

def install(name, minimum_version=None, required_version=None, scope=None, repository=None):
    if False:
        return 10
    "\n    Install a Powershell module from powershell gallery on the system.\n\n    :param name: Name of a Powershell module\n    :type  name: ``str``\n\n    :param minimum_version: The maximum version to install, e.g. 1.23.2\n    :type  minimum_version: ``str``\n\n    :param required_version: Install a specific version\n    :type  required_version: ``str``\n\n    :param scope: The scope to install the module to, e.g. CurrentUser, Computer\n    :type  scope: ``str``\n\n    :param repository: The friendly name of a private repository, e.g. MyREpo\n    :type  repository: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.install PowerPlan\n    "
    flags = [('Name', name)]
    if minimum_version is not None:
        flags.append(('MinimumVersion', minimum_version))
    if required_version is not None:
        flags.append(('RequiredVersion', required_version))
    if scope is not None:
        flags.append(('Scope', scope))
    if repository is not None:
        flags.append(('Repository', repository))
    params = ''
    for (flag, value) in flags:
        params += '-{} {} '.format(flag, value)
    cmd = 'Install-Module {} -Force'.format(params)
    _pshell(cmd)
    return name in list_modules()

def update(name, maximum_version=None, required_version=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update a PowerShell module to a specific version, or the newest\n\n    :param name: Name of a Powershell module\n    :type  name: ``str``\n\n    :param maximum_version: The maximum version to install, e.g. 1.23.2\n    :type  maximum_version: ``str``\n\n    :param required_version: Install a specific version\n    :type  required_version: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.update PowerPlan\n    "
    flags = [('Name', name)]
    if maximum_version is not None:
        flags.append(('MaximumVersion', maximum_version))
    if required_version is not None:
        flags.append(('RequiredVersion', required_version))
    params = ''
    for (flag, value) in flags:
        params += '-{} {} '.format(flag, value)
    cmd = 'Update-Module {} -Force'.format(params)
    _pshell(cmd)
    return name in list_modules()

def remove(name):
    if False:
        i = 10
        return i + 15
    "\n    Remove a Powershell DSC module from the system.\n\n    :param  name: Name of a Powershell DSC module\n    :type   name: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.remove PowerPlan\n    "
    cmd = 'Uninstall-Module "{}"'.format(name)
    no_ret = _pshell(cmd)
    return name not in list_modules()

def register_repository(name, location, installation_policy=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Register a PSGet repository on the local machine\n\n    :param name: The name for the repository\n    :type  name: ``str``\n\n    :param location: The URI for the repository\n    :type  location: ``str``\n\n    :param installation_policy: The installation policy\n        for packages, e.g. Trusted, Untrusted\n    :type  installation_policy: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.register_repository MyRepo https://myrepo.mycompany.com/packages\n    "
    flags = [('Name', name)]
    flags.append(('SourceLocation', location))
    if installation_policy is not None:
        flags.append(('InstallationPolicy', installation_policy))
    params = ''
    for (flag, value) in flags:
        params += '-{} {} '.format(flag, value)
    cmd = 'Register-PSRepository {}'.format(params)
    no_ret = _pshell(cmd)
    return name not in list_modules()

def get_repository(name):
    if False:
        return 10
    "\n    Get the details of a local PSGet repository\n\n    :param  name: Name of the repository\n    :type   name: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'win01' psget.get_repository MyRepo\n    "
    cmd = 'Get-PSRepository "{}"'.format(name)
    no_ret = _pshell(cmd)
    return name not in list_modules()