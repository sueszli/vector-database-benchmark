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

# Define the module's virtual name
__virtualname__ = "service"

__func_alias__ = {
    "list_": "list",
}

log = logging.getLogger(__name__)

SALT_MAC_SERVICES = {
    "salt-minion": "com.saltstack.salt.minion",
    "salt-master": "com.saltstack.salt.master",
    "salt-api": "com.saltstack.salt.api",
    "salt-syndic": "com.saltstack.salt.syndic",
}


def __virtual__():
    """
    Only for macOS with launchctl
    """
    if not salt.utils.platform.is_darwin():
        return (
            False,
            "Failed to load the mac_service module:\nOnly available on macOS systems.",
        )

    if not salt.utils.path.which("launchctl"):
        return (
            False,
            "Failed to load the mac_service module:\n"
            'Required binary not found: "launchctl"',
        )

    if not salt.utils.path.which("plutil"):
        return (
            False,
            "Failed to load the mac_service module:\n"
            'Required binary not found: "plutil"',
        )

    if Version(__grains__["osrelease"]) < Version("10.11"):
        return (
            False,
            "Failed to load the mac_service module:\nRequires macOS 10.11 or newer",
        )

    return __virtualname__


def _name_in_services(name, services):
    """
    Checks to see if the given service is in the given services.

    :param str name: Service label, file name, or full path

    :param dict services: The currently available services.

    :return: The service information for the service, otherwise
    an empty dictionary

    :rtype: dict
    """
    if name in services:
        # Match on label
        return services[name]

    for service in services.values():
        if service["file_path"].lower() == name:
            # Match on full path
            return service
        basename, ext = os.path.splitext(service["file_name"])
        if basename.lower() == name:
            # Match on basename
            return service

    return dict()


def _get_service(name):
    """
    Get information about a service.  If the service is not found, raise an
    error

    :param str name: Service label, file name, or full path

    :return: The service information for the service, otherwise an Error
    :rtype: dict
    """
    services = __utils__["mac_utils.available_services"]()
    # fix the name differences between platforms
    # salt-minion becomes com.saltstack.salt.minion
    name = SALT_MAC_SERVICES.get(name, name).lower()

    service = _name_in_services(name, services)

    # if we would the service we can return it
    if service:
        return service

    # if we got here our service is not available, now we can check to see if
    # we received a cached batch of services, if not we did a fresh check
    # so we need to raise that the service could not be found.
    try:
        if not __context__["using_cached_services"]:
            raise CommandExecutionError(f"Service not found: {name}")
    except KeyError:
        pass

    # if we can't find a service and we are being run from a service.dead
    # state then there is no reason to check again.
    # fixes https://github.com/saltstack/salt/issues/57907
    if __context__.get("service.state") == "dead":
        raise CommandExecutionError(f"Service not found: {name}")

    # we used a cached version to check, a service could have been made
    # between now and then, we should refresh our available services.
    services = __utils__["mac_utils.available_services"](refresh=True)

    # check to see if we found the service we are looking for.
    service = _name_in_services(name, services)

    if not service:
        # Could not find the service after refresh raise.
        raise CommandExecutionError(f"Service not found: {name}")

    # found it :)
    return service


def _always_running_service(name):
    """
    Check if the service should always be running based on the KeepAlive Key
    in the service plist.

    :param str name: Service label, file name, or full path

    :return: True if the KeepAlive key is set to True, False if set to False or
        not set in the plist at all.

    :rtype: bool

    .. versionadded:: 2019.2.0
    """

    # get all the info from the launchctl service
    service_info = show(name)

    # get the value for the KeepAlive key in service plist
    try:
        keep_alive = service_info["plist"]["KeepAlive"]
    except KeyError:
        return False

    # check if KeepAlive is True and not just set.

    if isinstance(keep_alive, dict):
        # check for pathstate
        for _file, value in keep_alive.get("PathState", {}).items():
            if value is True and os.path.exists(_file):
                return True
            elif value is False and not os.path.exists(_file):
                return True

    if keep_alive is True:
        return True

    return False


def _get_domain_target(name, service_target=False):
    """
    Returns the domain/service target and path for a service. This is used to
    determine whether or not a service should be loaded in a user space or
    system space.

    :param str name: Service label, file name, or full path

    :param bool service_target: Whether to return a full
    service target. This is needed for the enable and disable
    subcommands of /bin/launchctl. Defaults to False

    :return: Tuple of the domain/service target and the path to the service.

    :rtype: tuple

    .. versionadded:: 2019.2.0
    """

    # Get service information
    service = _get_service(name)

    # get the path to the service
    path = service["file_path"]

    # most of the time we'll be at the system level.
    domain_target = "system"

    # check if a LaunchAgent as we should treat these differently.
    if "LaunchAgents" in path:
        # Get the console user so we can service in the correct session
        uid = __utils__["mac_utils.console_user"]()
        domain_target = f"gui/{uid}"

    # check to see if we need to make it a full service target.
    if service_target is True:
        domain_target = "{}/{}".format(domain_target, service["plist"]["Label"])

    return (domain_target, path)


def _launch_agent(name):
    """
    Checks to see if the provided service is a LaunchAgent

    :param str name: Service label, file name, or full path

    :return: True if a LaunchAgent, False if not.

    :rtype: bool

    .. versionadded:: 2019.2.0
    """

    # Get the path to the service.
    path = _get_service(name)["file_path"]

    if "LaunchAgents" not in path:
        return False
    return True


def show(name):
    """
    Show properties of a launchctl service

    :param str name: Service label, file name, or full path

    :return: The service information if the service is found
    :rtype: dict

    CLI Example:

    .. code-block:: bash

        salt '*' service.show org.cups.cupsd  # service label
        salt '*' service.show org.cups.cupsd.plist  # file name
        salt '*' service.show /System/Library/LaunchDaemons/org.cups.cupsd.plist  # full path
    """
    return _get_service(name)


def launchctl(sub_cmd, *args, **kwargs):
    """
    Run a launchctl command and raise an error if it fails

    :param str sub_cmd: Sub command supplied to launchctl

    :param tuple args: Tuple containing additional arguments to pass to
        launchctl

    :param dict kwargs: Dictionary containing arguments to pass to
        ``cmd.run_all``

    :param bool return_stdout: A keyword argument.  If true return the stdout
        of the launchctl command

    :return: ``True`` if successful, raise ``CommandExecutionError`` if not, or
        the stdout of the launchctl command if requested
    :rtype: bool, str

    CLI Example:

    .. code-block:: bash

        salt '*' service.launchctl debug org.cups.cupsd
    """
    return __utils__["mac_utils.launchctl"](sub_cmd, *args, **kwargs)


def list_(name=None, runas=None):
    """
    Run launchctl list and return the output

    :param str name: The name of the service to list

    :param str runas: User to run launchctl commands

    :return: If a name is passed returns information about the named service,
        otherwise returns a list of all services and pids
    :rtype: str

    CLI Example:

    .. code-block:: bash

        salt '*' service.list
        salt '*' service.list org.cups.cupsd
    """
    if name:
        # Get service information and label
        service = _get_service(name)
        label = service["plist"]["Label"]

        # we can assume if we are trying to list a LaunchAgent we need
        # to run as a user, if not provided, we'll use the console user.
        if not runas and _launch_agent(name):
            runas = __utils__["mac_utils.console_user"](username=True)

        # Collect information on service: will raise an error if it fails
        return launchctl("list", label, return_stdout=True, runas=runas)

    # Collect information on all services: will raise an error if it fails
    return launchctl("list", return_stdout=True, runas=runas)


def enable(name, runas=None):
    """
    Enable a launchd service. Raises an error if the service fails to be enabled

    :param str name: Service label, file name, or full path

    :param str runas: User to run launchctl commands

    :return: ``True`` if successful or if the service is already enabled
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.enable org.cups.cupsd
    """
    # Get the domain target. enable requires a full <service-target>
    service_target = _get_domain_target(name, service_target=True)[0]

    # Enable the service: will raise an error if it fails
    return launchctl("enable", service_target, runas=runas)


def disable(name, runas=None):
    """
    Disable a launchd service. Raises an error if the service fails to be
    disabled

    :param str name: Service label, file name, or full path

    :param str runas: User to run launchctl commands

    :return: ``True`` if successful or if the service is already disabled
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.disable org.cups.cupsd
    """
    # Get the service target. enable requires a full <service-target>
    service_target = _get_domain_target(name, service_target=True)[0]

    # disable the service: will raise an error if it fails
    return launchctl("disable", service_target, runas=runas)


def start(name, runas=None):
    """
    Start a launchd service.  Raises an error if the service fails to start

    .. note::
        To start a service in macOS the service must be enabled first. Use
        ``service.enable`` to enable the service.

    :param str name: Service label, file name, or full path

    :param str runas: User to run launchctl commands

    :return: ``True`` if successful or if the service is already running
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.start org.cups.cupsd
    """
    # Get the domain target.
    domain_target, path = _get_domain_target(name)

    # Load (bootstrap) the service: will raise an error if it fails
    return launchctl("bootstrap", domain_target, path, runas=runas)


def stop(name, runas=None):
    """
    Stop a launchd service.  Raises an error if the service fails to stop

    .. note::
        Though ``service.stop`` will unload a service in macOS, the service
        will start on next boot unless it is disabled. Use ``service.disable``
        to disable the service

    :param str name: Service label, file name, or full path

    :param str runas: User to run launchctl commands

    :return: ``True`` if successful or if the service is already stopped
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.stop org.cups.cupsd
    """
    # Get the domain target.
    domain_target, path = _get_domain_target(name)

    # Stop (bootout) the service: will raise an error if it fails
    return launchctl("bootout", domain_target, path, runas=runas)


def restart(name, runas=None):
    """
    Unloads and reloads a launchd service.  Raises an error if the service
    fails to reload

    :param str name: Service label, file name, or full path

    :param str runas: User to run launchctl commands

    :return: ``True`` if successful
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.restart org.cups.cupsd
    """
    # Restart the service: will raise an error if it fails
    if __salt__["service.loaded"](name, runas=runas):
        __salt__["service.stop"](name, runas=runas)
    return __salt__["service.start"](name, runas=runas)


def status(name, sig=None, runas=None):
    """
    Return the status for a service.

    .. note::
        Previously this function would return a PID for a running service with
        a PID or 'loaded' for a loaded service without a PID. This was changed
        to have better parity with other service modules that return True/False.

    :param str name: Used to find the service from launchctl.  Can be the
        service Label, file name, or path to the service file. (normally a plist)

    :param str sig: Find the service with status.pid instead.  Note that
        ``name`` must still be provided.

    :param str runas: User to run launchctl commands.

    :return: True if running, otherwise False.

    :rtype: str

    CLI Example:

    .. code-block:: bash

        salt '*' service.status cups
    """
    # Find service with ps
    if sig:
        return __salt__["status.pid"](sig)

    try:
        _get_service(name)
    except CommandExecutionError as msg:
        log.error(msg)
        return False

    if not runas and _launch_agent(name):
        runas = __utils__["mac_utils.console_user"](username=True)

    try:
        output = __salt__["service.list"](name, runas=runas)
    except CommandExecutionError:
        return False

    # we should only check for a PID if it's supposed to have one.
    # If we can't find a PID then something is wrong with the service.
    if _always_running_service(name):
        return True if '"PID" =' in output else False

    return True


def available(name):
    """
    Check that the given service is available.

    :param str name: The name of the service

    :return: True if the service is available, otherwise False
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.available com.openssh.sshd
    """
    try:
        _get_service(name)
        return True
    except CommandExecutionError:
        return False


def missing(name):
    """
    The inverse of service.available
    Check that the given service is not available.

    :param str name: The name of the service

    :return: True if the service is not available, otherwise False
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.missing com.openssh.sshd
    """
    return not available(name)


def enabled(name, runas=None):
    """
    Check if the specified service is enabled (not disabled, capable of being
    loaded/bootstrapped).

    .. note::
        Previously this function would see if the service is loaded via
        ``launchctl list`` to determine if the service is enabled. This was not
        an accurate way to do so. The new behavior checks to make sure its not
        disabled to determine the status. Please use ``service.loaded`` for the
        previous behavior.

    :param str name: The name of the service to look up.

    :param str runas: User to run launchctl commands.

    :return: True if the specified service enabled, otherwise False
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.enabled org.cups.cupsd
    """
    # There isn't a direct way to get enabled, but if its not disabled
    # then its enabled.
    return not __salt__["service.disabled"](name, runas)


def disabled(name, runas=None, domain="system"):
    """
    Check if the specified service is not enabled. This is the opposite of
    ``service.enabled``

    :param str name: The name to look up

    :param str runas: User to run launchctl commands

    :param str domain: domain to check for disabled services. Default is system.

    :return: True if the specified service is NOT enabled, otherwise False
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.disabled org.cups.cupsd
    """
    domain = _get_domain_target(name, service_target=True)[0]

    disabled = launchctl("print-disabled", domain, return_stdout=True, runas=runas)
    for service in disabled.split("\n"):
        if name in service:
            srv_name = service.split("=>")[0].split('"')[1]
            status = service.split("=>")[1]
            if name != srv_name:
                pass
            else:
                matches = ["true", "disabled"]
                return True if any([x in status.lower() for x in matches]) else False

    return False


def get_all(runas=None):
    """
    Return a list of services that are enabled or available. Can be used to
    find the name of a service.

    :param str runas: User to run launchctl commands

    :return: A list of all the services available or enabled
    :rtype: list

    CLI Example:

    .. code-block:: bash

        salt '*' service.get_all
    """
    # Get list of enabled services
    enabled = get_enabled(runas=runas)

    # Get list of all services
    available = list(__utils__["mac_utils.available_services"]().keys())

    # Return composite list
    return sorted(set(enabled + available))


def get_enabled(runas=None):
    """
    Return a list of all services that are enabled. Can be used to find the
    name of a service.

    :param str runas: User to run launchctl commands

    :return: A list of all the services enabled on the system
    :rtype: list

    CLI Example:

    .. code-block:: bash

        salt '*' service.get_enabled
    """
    # Collect list of enabled services
    stdout = list_(runas=runas)
    service_lines = [line for line in stdout.splitlines()]

    # Construct list of enabled services
    enabled = []
    for line in service_lines:
        # Skip header line
        if line.startswith("PID"):
            continue

        pid, status, label = line.split("\t")
        enabled.append(label)

    return sorted(set(enabled))


def loaded(name, runas=None):
    """
    Check if the specified service is loaded.

    :param str name: The name of the service to look up

    :param str runas: User to run launchctl commands

    :return: ``True`` if the specified service is loaded, otherwise ``False``
    :rtype: bool

    CLI Example:

    .. code-block:: bash

        salt '*' service.loaded org.cups.cupsd
    """
    # Try to list the service.  If it can't be listed, it's not enabled
    try:
        __salt__["service.list"](name=name, runas=runas)
        return True
    except CommandExecutionError:
        return False
