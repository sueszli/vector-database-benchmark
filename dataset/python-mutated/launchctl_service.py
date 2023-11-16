"""
Module for the management of MacOS systems that use launchd/launchctl

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.

:depends:   - plistlib Python module
"""
import fnmatch
import logging
import os
import plistlib
import re
import salt.utils.data
import salt.utils.decorators as decorators
import salt.utils.files
import salt.utils.path
import salt.utils.platform
import salt.utils.stringutils
from salt.utils.versions import Version
log = logging.getLogger(__name__)
__virtualname__ = 'service'
BEFORE_YOSEMITE = True

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on MacOS\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'Failed to load the mac_service module:\nOnly available on macOS systems.')
    if not os.path.exists('/bin/launchctl'):
        return (False, 'Failed to load the mac_service module:\nRequired binary not found: "/bin/launchctl"')
    if Version(__grains__['osrelease']) >= Version('10.11'):
        return (False, 'Failed to load the mac_service module:\nNot available on El Capitan, uses mac_service.py')
    if Version(__grains__['osrelease']) >= Version('10.10'):
        global BEFORE_YOSEMITE
        BEFORE_YOSEMITE = False
    return __virtualname__

def _launchd_paths():
    if False:
        return 10
    '\n    Paths where launchd services can be found\n    '
    return ['/Library/LaunchAgents', '/Library/LaunchDaemons', '/System/Library/LaunchAgents', '/System/Library/LaunchDaemons']

@decorators.memoize
def _available_services():
    if False:
        print('Hello World!')
    '\n    Return a dictionary of all available services on the system\n    '
    available_services = dict()
    for launch_dir in _launchd_paths():
        for (root, dirs, files) in salt.utils.path.os_walk(launch_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                true_path = os.path.realpath(file_path)
                if not os.path.exists(true_path):
                    continue
                try:
                    with salt.utils.files.fopen(file_path):
                        plist = plistlib.readPlist(salt.utils.data.decode(true_path))
                except Exception:
                    cmd = '/usr/bin/plutil -convert xml1 -o - -- "{}"'.format(true_path)
                    plist_xml = __salt__['cmd.run_all'](cmd, python_shell=False)['stdout']
                    plist = plistlib.readPlistFromBytes(salt.utils.stringutils.to_bytes(plist_xml))
                try:
                    available_services[plist.Label.lower()] = {'filename': filename, 'file_path': true_path, 'plist': plist}
                except AttributeError:
                    pass
    return available_services

def _service_by_name(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the service info for a service by label, filename or path\n    '
    services = _available_services()
    name = name.lower()
    if name in services:
        return services[name]
    for service in services.values():
        if service['file_path'].lower() == name:
            return service
        (basename, ext) = os.path.splitext(service['filename'])
        if basename.lower() == name:
            return service
    return False

def get_all():
    if False:
        return 10
    "\n    Return all installed services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    cmd = 'launchctl list'
    service_lines = [line for line in __salt__['cmd.run'](cmd).splitlines() if not line.startswith('PID')]
    service_labels_from_list = [line.split('\t')[2] for line in service_lines]
    service_labels_from_services = list(_available_services().keys())
    return sorted(set(service_labels_from_list + service_labels_from_services))

def _get_launchctl_data(job_label, runas=None):
    if False:
        print('Hello World!')
    if BEFORE_YOSEMITE:
        cmd = 'launchctl list -x {}'.format(job_label)
    else:
        cmd = 'launchctl list {}'.format(job_label)
    launchctl_data = __salt__['cmd.run_all'](cmd, python_shell=False, runas=runas)
    if launchctl_data['stderr']:
        return None
    return launchctl_data['stdout']

def available(job_label):
    if False:
        while True:
            i = 10
    "\n    Check that the given service is available.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available com.openssh.sshd\n    "
    return True if _service_by_name(job_label) else False

def missing(job_label):
    if False:
        return 10
    "\n    The inverse of service.available\n    Check that the given service is not available.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing com.openssh.sshd\n    "
    return False if _service_by_name(job_label) else True

def status(name, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Return the status for a service via systemd.\n    If the name contains globbing, a dict mapping service name to True/False\n    values is returned.\n\n    .. versionchanged:: 2018.3.0\n        The service name can now be a glob (e.g. ``salt*``)\n\n    Args:\n        name (str): The name of the service to check\n        runas (str): User to run launchctl commands\n\n    Returns:\n        bool: True if running, False otherwise\n        dict: Maps service name to True if running, False otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name>\n    "
    contains_globbing = bool(re.search('\\*|\\?|\\[.+\\]', name))
    if contains_globbing:
        services = fnmatch.filter(get_all(), name)
    else:
        services = [name]
    results = {}
    for service in services:
        service_info = _service_by_name(service)
        lookup_name = service_info['plist']['Label'] if service_info else service
        launchctl_data = _get_launchctl_data(lookup_name, runas=runas)
        if launchctl_data:
            if BEFORE_YOSEMITE:
                results[service] = 'PID' in plistlib.loads(launchctl_data)
            else:
                pattern = '"PID" = [0-9]+;'
                results[service] = True if re.search(pattern, launchctl_data) else False
        else:
            results[service] = False
    if contains_globbing:
        return results
    return results[name]

def stop(job_label, runas=None):
    if False:
        print('Hello World!')
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service label>\n        salt '*' service.stop org.ntp.ntpd\n        salt '*' service.stop /System/Library/LaunchDaemons/org.ntp.ntpd.plist\n    "
    service = _service_by_name(job_label)
    if service:
        cmd = 'launchctl unload -w {}'.format(service['file_path'], runas=runas)
        return not __salt__['cmd.retcode'](cmd, runas=runas, python_shell=False)
    return False

def start(job_label, runas=None):
    if False:
        return 10
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service label>\n        salt '*' service.start org.ntp.ntpd\n        salt '*' service.start /System/Library/LaunchDaemons/org.ntp.ntpd.plist\n    "
    service = _service_by_name(job_label)
    if service:
        cmd = 'launchctl load -w {}'.format(service['file_path'], runas=runas)
        return not __salt__['cmd.retcode'](cmd, runas=runas, python_shell=False)
    return False

def restart(job_label, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service label>\n    "
    stop(job_label, runas=runas)
    return start(job_label, runas=runas)

def enabled(job_label, runas=None):
    if False:
        while True:
            i = 10
    "\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service label>\n    "
    overrides_data = dict(plistlib.readPlist('/var/db/launchd.db/com.apple.launchd/overrides.plist'))
    if overrides_data.get(job_label, False):
        if overrides_data[job_label]['Disabled']:
            return False
        else:
            return True
    else:
        return False

def disabled(job_label, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return True if the named service is disabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service label>\n    "
    overrides_data = dict(plistlib.readPlist('/var/db/launchd.db/com.apple.launchd/overrides.plist'))
    if overrides_data.get(job_label, False):
        if overrides_data[job_label]['Disabled']:
            return True
        else:
            return False
    else:
        return True