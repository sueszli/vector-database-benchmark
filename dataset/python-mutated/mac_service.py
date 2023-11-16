"""
The service module for macOS

.. versionadded:: 2016.3.0

This module has support for services in the following locations.

.. code-block:: bash

    /System/Library/LaunchDaemons/
    /System/Library/LaunchAgents/
    /Library/LaunchDaemons/
    /Library/LaunchAgents/

    # As of version "2019.2.0" support for user-specific services were added.
    /Users/foo/Library/LaunchAgents/

.. note::
    As of the 2019.2.0 release, if a service is located in a ``LaunchAgent``
    path and a ``runas`` user is NOT specified, the current console user will
    be used to properly interact with the service.

.. note::
    As of the 3002 release, if a service name of ``salt-minion`` is passed this
    module will convert it over to it's macOS equivalent name, in this case
    to ``com.saltstack.salt.minion``. This is true for ``salt-master``
    ``salt-api``, and ``salt-syndic`` as well.

"""
import logging
import os
import salt.utils.files
import salt.utils.path
import salt.utils.platform
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
from salt.utils.versions import Version
__virtualname__ = 'service'
__func_alias__ = {'list_': 'list'}
log = logging.getLogger(__name__)
SALT_MAC_SERVICES = {'salt-minion': 'com.saltstack.salt.minion', 'salt-master': 'com.saltstack.salt.master', 'salt-api': 'com.saltstack.salt.api', 'salt-syndic': 'com.saltstack.salt.syndic'}

def __virtual__():
    if False:
        return 10
    '\n    Only for macOS with launchctl\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'Failed to load the mac_service module:\nOnly available on macOS systems.')
    if not salt.utils.path.which('launchctl'):
        return (False, 'Failed to load the mac_service module:\nRequired binary not found: "launchctl"')
    if not salt.utils.path.which('plutil'):
        return (False, 'Failed to load the mac_service module:\nRequired binary not found: "plutil"')
    if Version(__grains__['osrelease']) < Version('10.11'):
        return (False, 'Failed to load the mac_service module:\nRequires macOS 10.11 or newer')
    return __virtualname__

def _name_in_services(name, services):
    if False:
        while True:
            i = 10
    '\n    Checks to see if the given service is in the given services.\n\n    :param str name: Service label, file name, or full path\n\n    :param dict services: The currently available services.\n\n    :return: The service information for the service, otherwise\n    an empty dictionary\n\n    :rtype: dict\n    '
    if name in services:
        return services[name]
    for service in services.values():
        if service['file_path'].lower() == name:
            return service
        (basename, ext) = os.path.splitext(service['file_name'])
        if basename.lower() == name:
            return service
    return dict()

def _get_service(name):
    if False:
        print('Hello World!')
    '\n    Get information about a service.  If the service is not found, raise an\n    error\n\n    :param str name: Service label, file name, or full path\n\n    :return: The service information for the service, otherwise an Error\n    :rtype: dict\n    '
    services = __utils__['mac_utils.available_services']()
    name = SALT_MAC_SERVICES.get(name, name).lower()
    service = _name_in_services(name, services)
    if service:
        return service
    try:
        if not __context__['using_cached_services']:
            raise CommandExecutionError(f'Service not found: {name}')
    except KeyError:
        pass
    if __context__.get('service.state') == 'dead':
        raise CommandExecutionError(f'Service not found: {name}')
    services = __utils__['mac_utils.available_services'](refresh=True)
    service = _name_in_services(name, services)
    if not service:
        raise CommandExecutionError(f'Service not found: {name}')
    return service

def _always_running_service(name):
    if False:
        return 10
    '\n    Check if the service should always be running based on the KeepAlive Key\n    in the service plist.\n\n    :param str name: Service label, file name, or full path\n\n    :return: True if the KeepAlive key is set to True, False if set to False or\n        not set in the plist at all.\n\n    :rtype: bool\n\n    .. versionadded:: 2019.2.0\n    '
    service_info = show(name)
    try:
        keep_alive = service_info['plist']['KeepAlive']
    except KeyError:
        return False
    if isinstance(keep_alive, dict):
        for (_file, value) in keep_alive.get('PathState', {}).items():
            if value is True and os.path.exists(_file):
                return True
            elif value is False and (not os.path.exists(_file)):
                return True
    if keep_alive is True:
        return True
    return False

def _get_domain_target(name, service_target=False):
    if False:
        print('Hello World!')
    '\n    Returns the domain/service target and path for a service. This is used to\n    determine whether or not a service should be loaded in a user space or\n    system space.\n\n    :param str name: Service label, file name, or full path\n\n    :param bool service_target: Whether to return a full\n    service target. This is needed for the enable and disable\n    subcommands of /bin/launchctl. Defaults to False\n\n    :return: Tuple of the domain/service target and the path to the service.\n\n    :rtype: tuple\n\n    .. versionadded:: 2019.2.0\n    '
    service = _get_service(name)
    path = service['file_path']
    domain_target = 'system'
    if 'LaunchAgents' in path:
        uid = __utils__['mac_utils.console_user']()
        domain_target = f'gui/{uid}'
    if service_target is True:
        domain_target = '{}/{}'.format(domain_target, service['plist']['Label'])
    return (domain_target, path)

def _launch_agent(name):
    if False:
        return 10
    '\n    Checks to see if the provided service is a LaunchAgent\n\n    :param str name: Service label, file name, or full path\n\n    :return: True if a LaunchAgent, False if not.\n\n    :rtype: bool\n\n    .. versionadded:: 2019.2.0\n    '
    path = _get_service(name)['file_path']
    if 'LaunchAgents' not in path:
        return False
    return True

def show(name):
    if False:
        return 10
    "\n    Show properties of a launchctl service\n\n    :param str name: Service label, file name, or full path\n\n    :return: The service information if the service is found\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.show org.cups.cupsd  # service label\n        salt '*' service.show org.cups.cupsd.plist  # file name\n        salt '*' service.show /System/Library/LaunchDaemons/org.cups.cupsd.plist  # full path\n    "
    return _get_service(name)

def launchctl(sub_cmd, *args, **kwargs):
    if False:
        return 10
    "\n    Run a launchctl command and raise an error if it fails\n\n    :param str sub_cmd: Sub command supplied to launchctl\n\n    :param tuple args: Tuple containing additional arguments to pass to\n        launchctl\n\n    :param dict kwargs: Dictionary containing arguments to pass to\n        ``cmd.run_all``\n\n    :param bool return_stdout: A keyword argument.  If true return the stdout\n        of the launchctl command\n\n    :return: ``True`` if successful, raise ``CommandExecutionError`` if not, or\n        the stdout of the launchctl command if requested\n    :rtype: bool, str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.launchctl debug org.cups.cupsd\n    "
    return __utils__['mac_utils.launchctl'](sub_cmd, *args, **kwargs)

def list_(name=None, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run launchctl list and return the output\n\n    :param str name: The name of the service to list\n\n    :param str runas: User to run launchctl commands\n\n    :return: If a name is passed returns information about the named service,\n        otherwise returns a list of all services and pids\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.list\n        salt '*' service.list org.cups.cupsd\n    "
    if name:
        service = _get_service(name)
        label = service['plist']['Label']
        if not runas and _launch_agent(name):
            runas = __utils__['mac_utils.console_user'](username=True)
        return launchctl('list', label, return_stdout=True, runas=runas)
    return launchctl('list', return_stdout=True, runas=runas)

def enable(name, runas=None):
    if False:
        print('Hello World!')
    "\n    Enable a launchd service. Raises an error if the service fails to be enabled\n\n    :param str name: Service label, file name, or full path\n\n    :param str runas: User to run launchctl commands\n\n    :return: ``True`` if successful or if the service is already enabled\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable org.cups.cupsd\n    "
    service_target = _get_domain_target(name, service_target=True)[0]
    return launchctl('enable', service_target, runas=runas)

def disable(name, runas=None):
    if False:
        while True:
            i = 10
    "\n    Disable a launchd service. Raises an error if the service fails to be\n    disabled\n\n    :param str name: Service label, file name, or full path\n\n    :param str runas: User to run launchctl commands\n\n    :return: ``True`` if successful or if the service is already disabled\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable org.cups.cupsd\n    "
    service_target = _get_domain_target(name, service_target=True)[0]
    return launchctl('disable', service_target, runas=runas)

def start(name, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Start a launchd service.  Raises an error if the service fails to start\n\n    .. note::\n        To start a service in macOS the service must be enabled first. Use\n        ``service.enable`` to enable the service.\n\n    :param str name: Service label, file name, or full path\n\n    :param str runas: User to run launchctl commands\n\n    :return: ``True`` if successful or if the service is already running\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start org.cups.cupsd\n    "
    (domain_target, path) = _get_domain_target(name)
    return launchctl('bootstrap', domain_target, path, runas=runas)

def stop(name, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop a launchd service.  Raises an error if the service fails to stop\n\n    .. note::\n        Though ``service.stop`` will unload a service in macOS, the service\n        will start on next boot unless it is disabled. Use ``service.disable``\n        to disable the service\n\n    :param str name: Service label, file name, or full path\n\n    :param str runas: User to run launchctl commands\n\n    :return: ``True`` if successful or if the service is already stopped\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop org.cups.cupsd\n    "
    (domain_target, path) = _get_domain_target(name)
    return launchctl('bootout', domain_target, path, runas=runas)

def restart(name, runas=None):
    if False:
        return 10
    "\n    Unloads and reloads a launchd service.  Raises an error if the service\n    fails to reload\n\n    :param str name: Service label, file name, or full path\n\n    :param str runas: User to run launchctl commands\n\n    :return: ``True`` if successful\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart org.cups.cupsd\n    "
    if __salt__['service.loaded'](name, runas=runas):
        __salt__['service.stop'](name, runas=runas)
    return __salt__['service.start'](name, runas=runas)

def status(name, sig=None, runas=None):
    if False:
        print('Hello World!')
    "\n    Return the status for a service.\n\n    .. note::\n        Previously this function would return a PID for a running service with\n        a PID or 'loaded' for a loaded service without a PID. This was changed\n        to have better parity with other service modules that return True/False.\n\n    :param str name: Used to find the service from launchctl.  Can be the\n        service Label, file name, or path to the service file. (normally a plist)\n\n    :param str sig: Find the service with status.pid instead.  Note that\n        ``name`` must still be provided.\n\n    :param str runas: User to run launchctl commands.\n\n    :return: True if running, otherwise False.\n\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status cups\n    "
    if sig:
        return __salt__['status.pid'](sig)
    try:
        _get_service(name)
    except CommandExecutionError as msg:
        log.error(msg)
        return False
    if not runas and _launch_agent(name):
        runas = __utils__['mac_utils.console_user'](username=True)
    try:
        output = __salt__['service.list'](name, runas=runas)
    except CommandExecutionError:
        return False
    if _always_running_service(name):
        return True if '"PID" =' in output else False
    return True

def available(name):
    if False:
        while True:
            i = 10
    "\n    Check that the given service is available.\n\n    :param str name: The name of the service\n\n    :return: True if the service is available, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available com.openssh.sshd\n    "
    try:
        _get_service(name)
        return True
    except CommandExecutionError:
        return False

def missing(name):
    if False:
        while True:
            i = 10
    "\n    The inverse of service.available\n    Check that the given service is not available.\n\n    :param str name: The name of the service\n\n    :return: True if the service is not available, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing com.openssh.sshd\n    "
    return not available(name)

def enabled(name, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Check if the specified service is enabled (not disabled, capable of being\n    loaded/bootstrapped).\n\n    .. note::\n        Previously this function would see if the service is loaded via\n        ``launchctl list`` to determine if the service is enabled. This was not\n        an accurate way to do so. The new behavior checks to make sure its not\n        disabled to determine the status. Please use ``service.loaded`` for the\n        previous behavior.\n\n    :param str name: The name of the service to look up.\n\n    :param str runas: User to run launchctl commands.\n\n    :return: True if the specified service enabled, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled org.cups.cupsd\n    "
    return not __salt__['service.disabled'](name, runas)

def disabled(name, runas=None, domain='system'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check if the specified service is not enabled. This is the opposite of\n    ``service.enabled``\n\n    :param str name: The name to look up\n\n    :param str runas: User to run launchctl commands\n\n    :param str domain: domain to check for disabled services. Default is system.\n\n    :return: True if the specified service is NOT enabled, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled org.cups.cupsd\n    "
    domain = _get_domain_target(name, service_target=True)[0]
    disabled = launchctl('print-disabled', domain, return_stdout=True, runas=runas)
    for service in disabled.split('\n'):
        if name in service:
            srv_name = service.split('=>')[0].split('"')[1]
            status = service.split('=>')[1]
            if name != srv_name:
                pass
            else:
                matches = ['true', 'disabled']
                return True if any([x in status.lower() for x in matches]) else False
    return False

def get_all(runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of services that are enabled or available. Can be used to\n    find the name of a service.\n\n    :param str runas: User to run launchctl commands\n\n    :return: A list of all the services available or enabled\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    enabled = get_enabled(runas=runas)
    available = list(__utils__['mac_utils.available_services']().keys())
    return sorted(set(enabled + available))

def get_enabled(runas=None):
    if False:
        return 10
    "\n    Return a list of all services that are enabled. Can be used to find the\n    name of a service.\n\n    :param str runas: User to run launchctl commands\n\n    :return: A list of all the services enabled on the system\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    stdout = list_(runas=runas)
    service_lines = [line for line in stdout.splitlines()]
    enabled = []
    for line in service_lines:
        if line.startswith('PID'):
            continue
        (pid, status, label) = line.split('\t')
        enabled.append(label)
    return sorted(set(enabled))

def loaded(name, runas=None):
    if False:
        while True:
            i = 10
    "\n    Check if the specified service is loaded.\n\n    :param str name: The name of the service to look up\n\n    :param str runas: User to run launchctl commands\n\n    :return: ``True`` if the specified service is loaded, otherwise ``False``\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.loaded org.cups.cupsd\n    "
    try:
        __salt__['service.list'](name=name, runas=runas)
        return True
    except CommandExecutionError:
        return False