"""
Manage VMware vCenter servers and ESXi hosts.

.. versionadded:: 2015.8.4

:codeauthor: Alexandru Bleotu <alexandru.bleotu@morganstaley.com>

Dependencies
============

- pyVmomi Python Module
- ESXCLI

pyVmomi
-------

PyVmomi can be installed via pip:

.. code-block:: bash

    pip install pyVmomi

.. note::

    Version 6.0 of pyVmomi has some problems with SSL error handling on certain
    versions of Python. If using version 6.0 of pyVmomi, Python 2.7.9,
    or newer must be present. This is due to an upstream dependency
    in pyVmomi 6.0 that is not supported in Python versions 2.7 to 2.7.8. If the
    version of Python is not in the supported range, you will need to install an
    earlier version of pyVmomi. See `Issue #29537`_ for more information.

.. _Issue #29537: https://github.com/saltstack/salt/issues/29537

Based on the note above, to install an earlier version of pyVmomi than the
version currently listed in PyPi, run the following:

.. code-block:: bash

    pip install pyVmomi==5.5.0.2014.1.1

The 5.5.0.2014.1.1 is a known stable version that this original vSphere Execution
Module was developed against.

vSphere Automation SDK
----------------------

vSphere Automation SDK can be installed via pip:

.. code-block:: bash

    pip install --upgrade pip setuptools
    pip install --upgrade git+https://github.com/vmware/vsphere-automation-sdk-python.git

.. note::

    The SDK also requires OpenSSL 1.0.1+ if you want to connect to vSphere 6.5+ in order to support
    TLS1.1 & 1.2.

    In order to use the tagging functions in this module, vSphere Automation SDK is necessary to
    install.

The module is currently in version 1.0.3
(as of 8/26/2019)

ESXCLI
------

Currently, about a third of the functions used in the vSphere Execution Module require
the ESXCLI package be installed on the machine running the Proxy Minion process.

The ESXCLI package is also referred to as the VMware vSphere CLI, or vCLI. VMware
provides vCLI package installation instructions for `vSphere 5.5`_ and
`vSphere 6.0`_.

.. _vSphere 5.5: http://pubs.vmware.com/vsphere-55/index.jsp#com.vmware.vcli.getstart.doc/cli_install.4.2.html
.. _vSphere 6.0: http://pubs.vmware.com/vsphere-60/index.jsp#com.vmware.vcli.getstart.doc/cli_install.4.2.html

Once all of the required dependencies are in place and the vCLI package is
installed, you can check to see if you can connect to your ESXi host or vCenter
server by running the following command:

.. code-block:: bash

    esxcli -s <host-location> -u <username> -p <password> system syslog config get

If the connection was successful, ESXCLI was successfully installed on your system.
You should see output related to the ESXi host's syslog configuration.

.. note::

    Be aware that some functionality in this execution module may depend on the
    type of license attached to a vCenter Server or ESXi host(s).

    For example, certain services are only available to manipulate service state
    or policies with a VMware vSphere Enterprise or Enterprise Plus license, while
    others are available with a Standard license. The ``ntpd`` service is restricted
    to an Enterprise Plus license, while ``ssh`` is available via the Standard
    license.

    Please see the `vSphere Comparison`_ page for more information.

.. _vSphere Comparison: https://www.vmware.com/products/vsphere/compare


About
=====

This execution module was designed to be able to handle connections both to a
vCenter Server, as well as to an ESXi host. It utilizes the pyVmomi Python
library and the ESXCLI package to run remote execution functions against either
the defined vCenter server or the ESXi host.

Whether or not the function runs against a vCenter Server or an ESXi host depends
entirely upon the arguments passed into the function. Each function requires a
``host`` location, ``username``, and ``password``. If the credentials provided
apply to a vCenter Server, then the function will be run against the vCenter
Server. For example, when listing hosts using vCenter credentials, you'll get a
list of hosts associated with that vCenter Server:

.. code-block:: bash

    # salt my-minion vsphere.list_hosts <vcenter-ip> <vcenter-user> <vcenter-password>
    my-minion:
    - esxi-1.example.com
    - esxi-2.example.com

However, some functions should be used against ESXi hosts, not vCenter Servers.
Functionality such as getting a host's coredump network configuration should be
performed against a host and not a vCenter server. If the authentication
information you're using is against a vCenter server and not an ESXi host, you
can provide the host name that is associated with the vCenter server in the
command, as a list, using the ``host_names`` or ``esxi_host`` kwarg. For
example:

.. code-block:: bash

    # salt my-minion vsphere.get_coredump_network_config <vcenter-ip> <vcenter-user>         <vcenter-password> esxi_hosts='[esxi-1.example.com, esxi-2.example.com]'
    my-minion:
    ----------
        esxi-1.example.com:
            ----------
            Coredump Config:
                ----------
                enabled:
                    False
        esxi-2.example.com:
            ----------
            Coredump Config:
                ----------
                enabled:
                    True
                host_vnic:
                    vmk0
                ip:
                    coredump-location.example.com
                port:
                    6500

You can also use these functions against an ESXi host directly by establishing a
connection to an ESXi host using the host's location, username, and password. If ESXi
connection credentials are used instead of vCenter credentials, the ``host_names`` and
``esxi_hosts`` arguments are not needed.

.. code-block:: bash

    # salt my-minion vsphere.get_coredump_network_config esxi-1.example.com root <host-password>
    local:
    ----------
        10.4.28.150:
            ----------
            Coredump Config:
                ----------
                enabled:
                    True
                host_vnic:
                    vmk0
                ip:
                    coredump-location.example.com
                port:
                    6500
"""
import datetime
import logging
import sys
from functools import wraps
import salt.utils.args
import salt.utils.dictupdate as dictupdate
import salt.utils.http
import salt.utils.path
import salt.utils.pbm
import salt.utils.vmware
import salt.utils.vsan
from salt.config.schemas.esxcluster import ESXClusterConfigSchema, ESXClusterEntitySchema
from salt.config.schemas.esxi import DiskGroupsDiskIdSchema, SimpleHostCacheSchema, VmfsDatastoreSchema
from salt.config.schemas.esxvm import ESXVirtualMachineDeleteSchema, ESXVirtualMachineUnregisterSchema
from salt.config.schemas.vcenter import VCenterEntitySchema
from salt.exceptions import ArgumentValueError, CommandExecutionError, InvalidConfigError, InvalidEntityError, VMwareApiError, VMwareObjectExistsError, VMwareObjectRetrievalError, VMwareSaltError
from salt.utils.decorators import depends, ignores_kwargs
from salt.utils.dictdiffer import recursive_diff
from salt.utils.listdiffer import list_diff
log = logging.getLogger(__name__)
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
try:
    from pyVmomi import VmomiSupport, pbm, vim, vmodl
    if 'vim25/6.0' in VmomiSupport.versionMap and sys.version_info > (2, 7) and (sys.version_info < (2, 7, 9)):
        log.debug('pyVmomi not loaded: Incompatible versions of Python. See Issue #29537.')
        raise ImportError()
    HAS_PYVMOMI = True
except ImportError:
    HAS_PYVMOMI = False
try:
    from com.vmware.cis.tagging_client import Category, CategoryModel, Tag, TagAssociation, TagModel
    from com.vmware.vapi.std.errors_client import AlreadyExists, InvalidArgument, NotFound, Unauthenticated, Unauthorized
    from com.vmware.vapi.std_client import DynamicID
    from com.vmware.vcenter_client import Cluster
    vsphere_errors = (AlreadyExists, InvalidArgument, NotFound, Unauthenticated, Unauthorized)
    HAS_VSPHERE_SDK = True
except ImportError:
    HAS_VSPHERE_SDK = False
esx_cli = salt.utils.path.which('esxcli')
if esx_cli:
    HAS_ESX_CLI = True
else:
    HAS_ESX_CLI = False
__virtualname__ = 'vsphere'
__proxyenabled__ = ['esxi', 'esxcluster', 'esxdatacenter', 'vcenter', 'esxvm']

def __virtual__():
    if False:
        return 10
    return __virtualname__

def _deprecation_message(function):
    if False:
        while True:
            i = 10
    '\n    Decorator wrapper to warn about azurearm deprecation\n    '

    @wraps(function)
    def wrapped(*args, **kwargs):
        if False:
            while True:
                i = 10
        salt.utils.versions.warn_until(3008, "The 'vsphere' functionality in Salt has been deprecated and its functionality will be removed in version 3008 in favor of the saltext.vmware Salt Extension. (https://github.com/saltstack/salt-ext-modules-vmware)", category=FutureWarning)
        ret = function(*args, **salt.utils.args.clean_kwargs(**kwargs))
        return ret
    return wrapped

@_deprecation_message
def get_proxy_type():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the proxy type retrieved either from the pillar of from the proxy\n    minion's config.  Returns ``<undefined>`` otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.get_proxy_type\n    "
    if __pillar__.get('proxy', {}).get('proxytype'):
        return __pillar__['proxy']['proxytype']
    if __opts__.get('proxy', {}).get('proxytype'):
        return __opts__['proxy']['proxytype']
    return '<undefined>'

def _get_proxy_connection_details():
    if False:
        print('Hello World!')
    '\n    Returns the connection details of the following proxies: esxi\n    '
    proxytype = get_proxy_type()
    if proxytype == 'esxi':
        details = __salt__['esxi.get_details']()
    elif proxytype == 'esxcluster':
        details = __salt__['esxcluster.get_details']()
    elif proxytype == 'esxdatacenter':
        details = __salt__['esxdatacenter.get_details']()
    elif proxytype == 'vcenter':
        details = __salt__['vcenter.get_details']()
    elif proxytype == 'esxvm':
        details = __salt__['esxvm.get_details']()
    else:
        raise CommandExecutionError(f"'{proxytype}' proxy is not supported")
    proxy_details = [details.get('vcenter') if 'vcenter' in details else details.get('host'), details.get('username'), details.get('password'), details.get('protocol'), details.get('port'), details.get('mechanism'), details.get('principal'), details.get('domain')]
    if 'verify_ssl' in details:
        proxy_details.append(details.get('verify_ssl'))
    return tuple(proxy_details)

def _supports_proxies(*proxy_types):
    if False:
        return 10
    '\n    Decorator to specify which proxy types are supported by a function\n\n    proxy_types:\n        Arbitrary list of strings with the supported types of proxies\n    '

    def _supports_proxies_(fn):
        if False:
            for i in range(10):
                print('nop')

        @wraps(fn)
        def __supports_proxies_(*args, **kwargs):
            if False:
                return 10
            proxy_type = get_proxy_type()
            if proxy_type not in proxy_types:
                raise CommandExecutionError("'{}' proxy is not supported by function {}".format(proxy_type, fn.__name__))
            return fn(*args, **salt.utils.args.clean_kwargs(**kwargs))
        return __supports_proxies_
    return _supports_proxies_

def _gets_service_instance_via_proxy(fn):
    if False:
        print('Hello World!')
    '\n    Decorator that connects to a target system (vCenter or ESXi host) using the\n    proxy details and passes the connection (vim.ServiceInstance) to\n    the decorated function.\n\n    Supported proxies: esxi, esxcluster, esxdatacenter.\n\n    Notes:\n        1. The decorated function must have a ``service_instance`` parameter\n        or a ``**kwarg`` type argument (name of argument is not important);\n        2. If the ``service_instance`` parameter is already defined, the value\n        is passed through to the decorated function;\n        3. If the ``service_instance`` parameter in not defined, the\n        connection is created using the proxy details and the service instance\n        is returned.\n    '
    fn_name = fn.__name__
    (arg_names, args_name, kwargs_name, default_values) = salt.utils.args.get_function_argspec(fn)
    default_values = default_values if default_values is not None else []

    @wraps(fn)
    def _gets_service_instance_via_proxy_(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'service_instance' not in arg_names and (not kwargs_name):
            raise CommandExecutionError("Function {} must have either a 'service_instance', or a '**kwargs' type parameter".format(fn_name))
        connection_details = _get_proxy_connection_details()
        local_service_instance = None
        if 'service_instance' in arg_names:
            idx = arg_names.index('service_instance')
            if idx >= len(arg_names) - len(default_values):
                if len(args) > idx:
                    if not args[idx]:
                        local_service_instance = salt.utils.vmware.get_service_instance(*connection_details)
                        args = list(args)
                        args[idx] = local_service_instance
                elif not kwargs.get('service_instance'):
                    local_service_instance = salt.utils.vmware.get_service_instance(*connection_details)
                    kwargs['service_instance'] = local_service_instance
        elif not kwargs.get('service_instance'):
            local_service_instance = salt.utils.vmware.get_service_instance(*connection_details)
            kwargs['service_instance'] = local_service_instance
        try:
            ret = fn(*args, **salt.utils.args.clean_kwargs(**kwargs))
            if local_service_instance:
                salt.utils.vmware.disconnect(local_service_instance)
            return ret
        except Exception as e:
            if local_service_instance:
                salt.utils.vmware.disconnect(local_service_instance)
            raise
    return _gets_service_instance_via_proxy_

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter', 'vcenter', 'esxvm')
@_deprecation_message
def get_service_instance_via_proxy(service_instance=None):
    if False:
        return 10
    '\n    Returns a service instance to the proxied endpoint (vCenter/ESXi host).\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    Note:\n        Should be used by state functions not invoked directly.\n\n    CLI Example:\n\n        See note above\n    '
    connection_details = _get_proxy_connection_details()
    return salt.utils.vmware.get_service_instance(*connection_details)

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter', 'vcenter', 'esxvm')
@_deprecation_message
def disconnect(service_instance):
    if False:
        print('Hello World!')
    '\n    Disconnects from a vCenter or ESXi host\n\n    Note:\n        Should be used by state functions, not invoked directly.\n\n    service_instance\n        Service instance (vim.ServiceInstance)\n\n    CLI Example:\n\n        See note above.\n    '
    salt.utils.vmware.disconnect(service_instance)
    return True

@depends(HAS_ESX_CLI)
@_deprecation_message
def esxcli_cmd(cmd_str, host=None, username=None, password=None, protocol=None, port=None, esxi_hosts=None, credstore=None):
    if False:
        i = 10
        return i + 15
    "\n    Run an ESXCLI command directly on the host or list of hosts.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    cmd_str\n        The ESXCLI command to run. Note: This should not include the ``-s``, ``-u``,\n        ``-p``, ``-h``, ``--protocol``, or ``--portnumber`` arguments that are\n        frequently passed when using a bare ESXCLI command from the command line.\n        Those arguments are handled by this function via the other args and kwargs.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.esxcli_cmd my.esxi.host root bad-password             'system coredump network get'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.esxcli_cmd my.vcenter.location root bad-password             'system coredump network get' esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd_str, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            if response['retcode'] != 0:
                ret.update({esxi_host: {'Error': response.get('stdout')}})
            else:
                ret.update({esxi_host: response})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd_str, protocol=protocol, port=port, credstore=credstore)
        if response['retcode'] != 0:
            ret.update({host: {'Error': response.get('stdout')}})
        else:
            ret.update({host: response})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def get_coredump_network_config(host, username, password, protocol=None, port=None, esxi_hosts=None, credstore=None):
    if False:
        return 10
    "\n    Retrieve information on ESXi or vCenter network dump collection and\n    format it into a dictionary.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: A dictionary with the network configuration, or, if getting\n             the network config failed, a an error message retrieved from the\n             standard cmd.run_all dictionary, per host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.get_coredump_network_config my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_coredump_network_config my.vcenter.location root bad-password             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n\n    "
    cmd = 'system coredump network get'
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            if response['retcode'] != 0:
                ret.update({esxi_host: {'Error': response.get('stdout')}})
            else:
                ret.update({esxi_host: {'Coredump Config': _format_coredump_stdout(response)}})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, credstore=credstore)
        if response['retcode'] != 0:
            ret.update({host: {'Error': response.get('stdout')}})
        else:
            stdout = _format_coredump_stdout(response)
            ret.update({host: {'Coredump Config': stdout}})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def coredump_network_enable(host, username, password, enabled, protocol=None, port=None, esxi_hosts=None, credstore=None):
    if False:
        i = 10
        return i + 15
    "\n    Enable or disable ESXi core dump collection. Returns ``True`` if coredump is enabled\n    and returns ``False`` if core dump is not enabled. If there was an error, the error\n    will be the value printed in the ``Error`` key dictionary for the given host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    enabled\n        Python True or False to enable or disable coredumps.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.coredump_network_enable my.esxi.host root bad-password True\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.coredump_network_enable my.vcenter.location root bad-password True             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    if enabled:
        enable_it = 1
    else:
        enable_it = 0
    cmd = f'system coredump network set -e {enable_it}'
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            if response['retcode'] != 0:
                ret.update({esxi_host: {'Error': response.get('stdout')}})
            else:
                ret.update({esxi_host: {'Coredump Enabled': enabled}})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, credstore=credstore)
        if response['retcode'] != 0:
            ret.update({host: {'Error': response.get('stdout')}})
        else:
            ret.update({host: {'Coredump Enabled': enabled}})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def set_coredump_network_config(host, username, password, dump_ip, protocol=None, port=None, host_vnic='vmk0', dump_port=6500, esxi_hosts=None, credstore=None):
    if False:
        while True:
            i = 10
    "\n\n    Set the network parameters for a network coredump collection.\n    Note that ESXi requires that the dumps first be enabled (see\n    `coredump_network_enable`) before these parameters may be set.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    dump_ip\n        IP address of host that will accept the dump.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    host_vnic\n        Host VNic port through which to communicate. Defaults to ``vmk0``.\n\n    dump_port\n        TCP port to use for the dump, defaults to ``6500``.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: A standard cmd.run_all dictionary with a `success` key added, per host.\n             `success` will be True if the set succeeded, False otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.set_coredump_network_config my.esxi.host root bad-password 'dump_ip.host.com'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.set_coredump_network_config my.vcenter.location root bad-password 'dump_ip.host.com'             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    cmd = 'system coredump network set -v {} -i {} -o {}'.format(host_vnic, dump_ip, dump_port)
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            if response['retcode'] != 0:
                response['success'] = False
            else:
                response['success'] = True
            ret.update({esxi_host: response})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, credstore=credstore)
        if response['retcode'] != 0:
            response['success'] = False
        else:
            response['success'] = True
        ret.update({host: response})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def get_firewall_status(host, username, password, protocol=None, port=None, esxi_hosts=None, credstore=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Show status of all firewall rule sets.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: Nested dictionary with two toplevel keys ``rulesets`` and ``success``\n             ``success`` will be True or False depending on query success\n             ``rulesets`` will list the rulesets and their statuses if ``success``\n             was true, per host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.get_firewall_status my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_firewall_status my.vcenter.location root bad-password             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    cmd = 'network firewall ruleset list'
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            if response['retcode'] != 0:
                ret.update({esxi_host: {'Error': response['stdout'], 'success': False, 'rulesets': None}})
            else:
                ret.update({esxi_host: _format_firewall_stdout(response)})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, credstore=credstore)
        if response['retcode'] != 0:
            ret.update({host: {'Error': response['stdout'], 'success': False, 'rulesets': None}})
        else:
            ret.update({host: _format_firewall_stdout(response)})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def enable_firewall_ruleset(host, username, password, ruleset_enable, ruleset_name, protocol=None, port=None, esxi_hosts=None, credstore=None):
    if False:
        print('Hello World!')
    "\n    Enable or disable an ESXi firewall rule set.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    ruleset_enable\n        True to enable the ruleset, false to disable.\n\n    ruleset_name\n        Name of ruleset to target.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: A standard cmd.run_all dictionary, per host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.enable_firewall_ruleset my.esxi.host root bad-password True 'syslog'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.enable_firewall_ruleset my.vcenter.location root bad-password True 'syslog'             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    cmd = 'network firewall ruleset set --enabled {} --ruleset-id={}'.format(ruleset_enable, ruleset_name)
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            ret.update({esxi_host: response})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, credstore=credstore)
        ret.update({host: response})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def syslog_service_reload(host, username, password, protocol=None, port=None, esxi_hosts=None, credstore=None):
    if False:
        print('Hello World!')
    "\n    Reload the syslog service so it will pick up any changes.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: A standard cmd.run_all dictionary.  This dictionary will at least\n             have a `retcode` key.  If `retcode` is 0 the command was successful.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.syslog_service_reload my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.syslog_service_reload my.vcenter.location root bad-password             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    cmd = 'system syslog reload'
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            ret.update({esxi_host: response})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, credstore=credstore)
        ret.update({host: response})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def set_syslog_config(host, username, password, syslog_config, config_value, protocol=None, port=None, firewall=True, reset_service=True, esxi_hosts=None, credstore=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the specified syslog configuration parameter. By default, this function will\n    reset the syslog service after the configuration is set.\n\n    host\n        ESXi or vCenter host to connect to.\n\n    username\n        User to connect as, usually root.\n\n    password\n        Password to connect with.\n\n    syslog_config\n        Name of parameter to set (corresponds to the command line switch for\n        esxcli without the double dashes (--))\n\n        Valid syslog_config values are ``logdir``, ``loghost``, ``default-rotate`,\n        ``default-size``, ``default-timeout``, and ``logdir-unique``.\n\n    config_value\n        Value for the above parameter. For ``loghost``, URLs or IP addresses to\n        use for logging. Multiple log servers can be specified by listing them,\n        comma-separated, but without spaces before or after commas.\n\n        (reference: https://blogs.vmware.com/vsphere/2012/04/configuring-multiple-syslog-servers-for-esxi-5.html)\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    firewall\n        Enable the firewall rule set for syslog. Defaults to ``True``.\n\n    reset_service\n        After a successful parameter set, reset the service. Defaults to ``True``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: Dictionary with a top-level key of 'success' which indicates\n             if all the parameters were reset, and individual keys\n             for each parameter indicating which succeeded or failed, per host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.set_syslog_config my.esxi.host root bad-password             loghost ssl://localhost:5432,tcp://10.1.0.1:1514\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.set_syslog_config my.vcenter.location root bad-password             loghost ssl://localhost:5432,tcp://10.1.0.1:1514             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n\n    "
    ret = {}
    if firewall and syslog_config == 'loghost':
        if esxi_hosts:
            if not isinstance(esxi_hosts, list):
                raise CommandExecutionError("'esxi_hosts' must be a list.")
            for esxi_host in esxi_hosts:
                response = enable_firewall_ruleset(host, username, password, ruleset_enable=True, ruleset_name='syslog', protocol=protocol, port=port, esxi_hosts=[esxi_host], credstore=credstore).get(esxi_host)
                if response['retcode'] != 0:
                    ret.update({esxi_host: {'enable_firewall': {'message': response['stdout'], 'success': False}}})
                else:
                    ret.update({esxi_host: {'enable_firewall': {'success': True}}})
        else:
            response = enable_firewall_ruleset(host, username, password, ruleset_enable=True, ruleset_name='syslog', protocol=protocol, port=port, credstore=credstore).get(host)
            if response['retcode'] != 0:
                ret.update({host: {'enable_firewall': {'message': response['stdout'], 'success': False}}})
            else:
                ret.update({host: {'enable_firewall': {'success': True}}})
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = _set_syslog_config_helper(host, username, password, syslog_config, config_value, protocol=protocol, port=port, reset_service=reset_service, esxi_host=esxi_host, credstore=credstore)
            if ret.get(esxi_host) is None:
                ret.update({esxi_host: {}})
            ret[esxi_host].update(response)
    else:
        response = _set_syslog_config_helper(host, username, password, syslog_config, config_value, protocol=protocol, port=port, reset_service=reset_service, credstore=credstore)
        if ret.get(host) is None:
            ret.update({host: {}})
        ret[host].update(response)
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def get_syslog_config(host, username, password, protocol=None, port=None, esxi_hosts=None, credstore=None):
    if False:
        i = 10
        return i + 15
    "\n    Retrieve the syslog configuration.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: Dictionary with keys and values corresponding to the\n             syslog configuration, per host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.get_syslog_config my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_syslog_config my.vcenter.location root bad-password             esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    cmd = 'system syslog config get'
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            ret.update({esxi_host: _format_syslog_config(response)})
    else:
        response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, credstore=credstore)
        ret.update({host: _format_syslog_config(response)})
    return ret

@depends(HAS_ESX_CLI)
@_deprecation_message
def reset_syslog_config(host, username, password, protocol=None, port=None, syslog_config=None, esxi_hosts=None, credstore=None):
    if False:
        print('Hello World!')
    "\n    Reset the syslog service to its default settings.\n\n    Valid syslog_config values are ``logdir``, ``loghost``, ``logdir-unique``,\n    ``default-rotate``, ``default-size``, ``default-timeout``,\n    or ``all`` for all of these.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    syslog_config\n        List of parameters to reset, provided as a comma-delimited string, or 'all' to\n        reset all syslog configuration parameters. Required.\n\n    esxi_hosts\n        If ``host`` is a vCenter host, then use esxi_hosts to execute this function\n        on a list of one or more ESXi machines.\n\n    credstore\n        Optionally set to path to the credential store file.\n\n    :return: Dictionary with a top-level key of 'success' which indicates\n             if all the parameters were reset, and individual keys\n             for each parameter indicating which succeeded or failed, per host.\n\n    .. note::\n\n        ``syslog_config`` can be passed as a quoted, comma-separated string. See CLI Example for details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for ESXi host connection information\n        salt '*' vsphere.reset_syslog_config my.esxi.host root bad-password             syslog_config='logdir,loghost'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.reset_syslog_config my.vcenter.location root bad-password             syslog_config='logdir,loghost' esxi_hosts='[esxi-1.host.com, esxi-2.host.com]'\n    "
    if not syslog_config:
        raise CommandExecutionError("The 'reset_syslog_config' function requires a 'syslog_config' setting.")
    valid_resets = ['logdir', 'loghost', 'default-rotate', 'default-size', 'default-timeout', 'logdir-unique']
    cmd = 'system syslog config set --reset='
    if ',' in syslog_config:
        resets = [ind_reset.strip() for ind_reset in syslog_config.split(',')]
    elif syslog_config == 'all':
        resets = valid_resets
    else:
        resets = [syslog_config]
    ret = {}
    if esxi_hosts:
        if not isinstance(esxi_hosts, list):
            raise CommandExecutionError("'esxi_hosts' must be a list.")
        for esxi_host in esxi_hosts:
            response_dict = _reset_syslog_config_params(host, username, password, cmd, resets, valid_resets, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            ret.update({esxi_host: response_dict})
    else:
        response_dict = _reset_syslog_config_params(host, username, password, cmd, resets, valid_resets, protocol=protocol, port=port, credstore=credstore)
        ret.update({host: response_dict})
    return ret

@ignores_kwargs('credstore')
@_deprecation_message
def upload_ssh_key(host, username, password, ssh_key=None, ssh_key_file=None, protocol=None, port=None, certificate_verify=None):
    if False:
        print('Hello World!')
    "\n    Upload an ssh key for root to an ESXi host via http PUT.\n    This function only works for ESXi, not vCenter.\n    Only one ssh key can be uploaded for root.  Uploading a second key will\n    replace any existing key.\n\n    :param host: The location of the ESXi Host\n    :param username: Username to connect as\n    :param password: Password for the ESXi web endpoint\n    :param ssh_key: Public SSH key, will be added to authorized_keys on ESXi\n    :param ssh_key_file: File containing the SSH key.  Use 'ssh_key' or\n                         ssh_key_file, but not both.\n    :param protocol: defaults to https, can be http if ssl is disabled on ESXi\n    :param port: defaults to 443 for https\n    :param certificate_verify: If true require that the SSL connection present\n                               a valid certificate. Default: True\n    :return: Dictionary with a 'status' key, True if upload is successful.\n             If upload is unsuccessful, 'status' key will be False and\n             an 'Error' key will have an informative message.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.upload_ssh_key my.esxi.host root bad-password ssh_key_file='/etc/salt/my_keys/my_key.pub'\n\n    "
    if protocol is None:
        protocol = 'https'
    if port is None:
        port = 443
    if certificate_verify is None:
        certificate_verify = True
    url = f'{protocol}://{host}:{port}/host/ssh_root_authorized_keys'
    ret = {}
    result = None
    try:
        if ssh_key:
            result = salt.utils.http.query(url, status=True, text=True, method='PUT', username=username, password=password, data=ssh_key, verify_ssl=certificate_verify)
        elif ssh_key_file:
            result = salt.utils.http.query(url, status=True, text=True, method='PUT', username=username, password=password, data_file=ssh_key_file, data_render=False, verify_ssl=certificate_verify)
        if result.get('status') == 200:
            ret['status'] = True
        else:
            ret['status'] = False
            ret['Error'] = result['error']
    except Exception as msg:
        ret['status'] = False
        ret['Error'] = msg
    return ret

@ignores_kwargs('credstore')
@_deprecation_message
def get_ssh_key(host, username, password, protocol=None, port=None, certificate_verify=None):
    if False:
        while True:
            i = 10
    "\n    Retrieve the authorized_keys entry for root.\n    This function only works for ESXi, not vCenter.\n\n    :param host: The location of the ESXi Host\n    :param username: Username to connect as\n    :param password: Password for the ESXi web endpoint\n    :param protocol: defaults to https, can be http if ssl is disabled on ESXi\n    :param port: defaults to 443 for https\n    :param certificate_verify: If true require that the SSL connection present\n                               a valid certificate. Default: True\n    :return: True if upload is successful\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.get_ssh_key my.esxi.host root bad-password certificate_verify=True\n\n    "
    if protocol is None:
        protocol = 'https'
    if port is None:
        port = 443
    if certificate_verify is None:
        certificate_verify = True
    url = f'{protocol}://{host}:{port}/host/ssh_root_authorized_keys'
    ret = {}
    try:
        result = salt.utils.http.query(url, status=True, text=True, method='GET', username=username, password=password, verify_ssl=certificate_verify)
        if result.get('status') == 200:
            ret['status'] = True
            ret['key'] = result['text']
        else:
            ret['status'] = False
            ret['Error'] = result['error']
    except Exception as msg:
        ret['status'] = False
        ret['Error'] = msg
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def get_host_datetime(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        print('Hello World!')
    "\n    Get the date/time information for a given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to get date/time information.\n\n        If host_names is not provided, the date/time information will be retrieved for the\n        ``host`` location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.get_host_datetime my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_host_datetime my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        date_time_manager = _get_date_time_mgr(host_ref)
        date_time = date_time_manager.QueryDateTime()
        ret.update({host_name: date_time})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def get_ntp_config(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        return 10
    "\n    Get the NTP configuration information for a given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to get ntp configuration information.\n\n        If host_names is not provided, the NTP configuration will be retrieved for the\n        ``host`` location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.get_ntp_config my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_ntp_config my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        ntp_config = host_ref.configManager.dateTimeSystem.dateTimeInfo.ntpConfig.server
        ret.update({host_name: ntp_config})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def get_service_policy(host, username, password, service_name, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        print('Hello World!')
    "\n    Get the service name's policy for a given host or list of hosts.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    service_name\n        The name of the service for which to retrieve the policy. Supported service names are:\n          - DCUI\n          - TSM\n          - SSH\n          - lbtd\n          - lsassd\n          - lwiod\n          - netlogond\n          - ntpd\n          - sfcbd-watchdog\n          - snmpd\n          - vprobed\n          - vpxa\n          - xorg\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to get service policy information.\n\n        If host_names is not provided, the service policy information will be retrieved\n        for the ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.get_service_policy my.esxi.host root bad-password 'ssh'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_service_policy my.vcenter.location root bad-password 'ntpd'         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    valid_services = ['DCUI', 'TSM', 'SSH', 'ssh', 'lbtd', 'lsassd', 'lwiod', 'netlogond', 'ntpd', 'sfcbd-watchdog', 'snmpd', 'vprobed', 'vpxa', 'xorg']
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        if service_name not in valid_services:
            ret.update({host_name: {'Error': f'{service_name} is not a valid service name.'}})
            return ret
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        services = host_ref.configManager.serviceSystem.serviceInfo.service
        if service_name == 'SSH' or service_name == 'ssh':
            temp_service_name = 'TSM-SSH'
        else:
            temp_service_name = service_name
        for service in services:
            if service.key == temp_service_name:
                ret.update({host_name: {service_name: service.policy}})
                break
            else:
                msg = "Could not find service '{}' for host '{}'.".format(service_name, host_name)
                ret.update({host_name: {'Error': msg}})
        if ret.get(host_name) is None:
            msg = f"'vsphere.get_service_policy' failed for host {host_name}."
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def get_service_running(host, username, password, service_name, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        print('Hello World!')
    "\n    Get the service name's running state for a given host or list of hosts.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    service_name\n        The name of the service for which to retrieve the policy. Supported service names are:\n          - DCUI\n          - TSM\n          - SSH\n          - lbtd\n          - lsassd\n          - lwiod\n          - netlogond\n          - ntpd\n          - sfcbd-watchdog\n          - snmpd\n          - vprobed\n          - vpxa\n          - xorg\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to get the service's running state.\n\n        If host_names is not provided, the service's running state will be retrieved\n        for the ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.get_service_running my.esxi.host root bad-password 'ssh'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_service_running my.vcenter.location root bad-password 'ntpd'         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    valid_services = ['DCUI', 'TSM', 'SSH', 'ssh', 'lbtd', 'lsassd', 'lwiod', 'netlogond', 'ntpd', 'sfcbd-watchdog', 'snmpd', 'vprobed', 'vpxa', 'xorg']
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        if service_name not in valid_services:
            ret.update({host_name: {'Error': f'{service_name} is not a valid service name.'}})
            return ret
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        services = host_ref.configManager.serviceSystem.serviceInfo.service
        if service_name == 'SSH' or service_name == 'ssh':
            temp_service_name = 'TSM-SSH'
        else:
            temp_service_name = service_name
        for service in services:
            if service.key == temp_service_name:
                ret.update({host_name: {service_name: service.running}})
                break
            else:
                msg = "Could not find service '{}' for host '{}'.".format(service_name, host_name)
                ret.update({host_name: {'Error': msg}})
        if ret.get(host_name) is None:
            msg = f"'vsphere.get_service_running' failed for host {host_name}."
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def get_vmotion_enabled(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the VMotion enabled status for a given host or a list of host_names. Returns ``True``\n    if VMotion is enabled, ``False`` if it is not enabled.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts to check if VMotion is enabled.\n\n        If host_names is not provided, the VMotion status will be retrieved for the\n        ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.get_vmotion_enabled my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_vmotion_enabled my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vmotion_vnic = host_ref.configManager.vmotionSystem.netConfig.selectedVnic
        if vmotion_vnic:
            ret.update({host_name: {'VMotion Enabled': True}})
        else:
            ret.update({host_name: {'VMotion Enabled': False}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def get_vsan_enabled(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the VSAN enabled status for a given host or a list of host_names. Returns ``True``\n    if VSAN is enabled, ``False`` if it is not enabled, and ``None`` if a VSAN Host Config\n    is unset, per host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts to check if VSAN enabled.\n\n        If host_names is not provided, the VSAN status will be retrieved for the\n        ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.get_vsan_enabled my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_vsan_enabled my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vsan_config = host_ref.config.vsanHostConfig
        if vsan_config is None:
            msg = f"VSAN System Config Manager is unset for host '{host_name}'."
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
        else:
            ret.update({host_name: {'VSAN Enabled': vsan_config.enabled}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def get_vsan_eligible_disks(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of VSAN-eligible disks for a given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts to check if any VSAN-eligible disks are available.\n\n        If host_names is not provided, the VSAN-eligible disks will be retrieved\n        for the ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.get_vsan_eligible_disks my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.get_vsan_eligible_disks my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    response = _get_vsan_eligible_disks(service_instance, host, host_names)
    ret = {}
    for (host_name, value) in response.items():
        error = value.get('Error')
        if error:
            ret.update({host_name: {'Error': error}})
            continue
        disks = value.get('Eligible')
        if disks and isinstance(disks, list):
            disk_names = []
            for disk in disks:
                disk_names.append(disk.canonicalName)
            ret.update({host_name: {'Eligible': disk_names}})
        else:
            ret.update({host_name: {'Eligible': disks}})
    return ret

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter', 'vcenter', 'esxvm')
@_gets_service_instance_via_proxy
@_deprecation_message
def test_vcenter_connection(service_instance=None):
    if False:
        print('Hello World!')
    "\n    Checks if a connection is to a vCenter\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.test_vcenter_connection\n    "
    try:
        if salt.utils.vmware.is_connection_to_a_vcenter(service_instance):
            return True
    except VMwareSaltError:
        return False
    return False

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def system_info(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        return 10
    "\n    Return system information about a VMware environment.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.system_info 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    ret = salt.utils.vmware.get_inventory(service_instance).about.__dict__
    if 'apiType' in ret:
        if ret['apiType'] == 'HostAgent':
            ret = dictupdate.update(ret, salt.utils.vmware.get_hardware_grains(service_instance))
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_datacenters(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of datacenters for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_datacenters 1.2.3.4 root bad-password\n\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_datacenters(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_clusters(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of clusters for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_clusters 1.2.3.4 root bad-password\n\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_clusters(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_datastore_clusters(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Returns a list of datastore clusters for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_datastore_clusters 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_datastore_clusters(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_datastores(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of datastores for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_datastores 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_datastores(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_hosts(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Returns a list of hosts for the specified VMware environment.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_hosts 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_hosts(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_resourcepools(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        print('Hello World!')
    "\n    Returns a list of resource pools for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_resourcepools 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_resourcepools(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_networks(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of networks for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_networks 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_networks(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_vms(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        print('Hello World!')
    "\n    Returns a list of VMs for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_vms 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_vms(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_folders(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of folders for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_folders 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_folders(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_dvs(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Returns a list of distributed virtual switches for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_dvs 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_dvs(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_vapps(host, username, password, protocol=None, port=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Returns a list of vApps for the specified host.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # List vapps from all minions\n        salt '*' vsphere.list_vapps 1.2.3.4 root bad-password\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    return salt.utils.vmware.list_vapps(service_instance)

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_ssds(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Returns a list of SSDs for the given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter the hosts for which to retrieve SSDs.\n\n        If host_names is not provided, SSDs will be retrieved for the\n        ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.list_ssds my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.list_ssds my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    names = []
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        disks = _get_host_ssds(host_ref)
        for disk in disks:
            names.append(disk.canonicalName)
        ret.update({host_name: names})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def list_non_ssds(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of Non-SSD disks for the given host or list of host_names.\n\n    .. note::\n\n        In the pyVmomi StorageSystem, ScsiDisks may, or may not have an ``ssd`` attribute.\n        This attribute indicates if the ScsiDisk is SSD backed. As this option is optional,\n        if a relevant disk in the StorageSystem does not have ``ssd = true``, it will end\n        up in the ``non_ssds`` list here.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter the hosts for which to retrieve Non-SSD disks.\n\n        If host_names is not provided, Non-SSD disks will be retrieved for the\n        ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.list_non_ssds my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.list_non_ssds my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    names = []
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        disks = _get_host_non_ssds(host_ref)
        for disk in disks:
            names.append(disk.canonicalName)
        ret.update({host_name: names})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def set_ntp_config(host, username, password, ntp_servers, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set NTP configuration for a given host of list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    ntp_servers\n        A list of servers that should be added to and configured for the specified\n        host's NTP configuration.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter which hosts to configure ntp servers.\n\n        If host_names is not provided, the NTP servers will be configured for the\n        ``host`` location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.ntp_configure my.esxi.host root bad-password '[192.174.1.100, 192.174.1.200]'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.ntp_configure my.vcenter.location root bad-password '[192.174.1.100, 192.174.1.200]'         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    if not isinstance(ntp_servers, list):
        raise CommandExecutionError("'ntp_servers' must be a list.")
    ntp_config = vim.HostNtpConfig(server=ntp_servers)
    date_config = vim.HostDateTimeConfig(ntpConfig=ntp_config)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        date_time_manager = _get_date_time_mgr(host_ref)
        log.debug("Configuring NTP Servers '%s' for host '%s'.", ntp_servers, host_name)
        try:
            date_time_manager.UpdateDateTimeConfig(config=date_config)
        except vim.fault.HostConfigFault as err:
            msg = f'vsphere.ntp_configure_servers failed: {err}'
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
            continue
        ret.update({host_name: {'NTP Servers': ntp_config}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def service_start(host, username, password, service_name, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Start the named service for the given host or list of hosts.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    service_name\n        The name of the service for which to set the policy. Supported service names are:\n          - DCUI\n          - TSM\n          - SSH\n          - lbtd\n          - lsassd\n          - lwiod\n          - netlogond\n          - ntpd\n          - sfcbd-watchdog\n          - snmpd\n          - vprobed\n          - vpxa\n          - xorg\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to start the service.\n\n        If host_names is not provided, the service will be started for the ``host``\n        location instead. This is useful for when service instance connection information\n        is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.service_start my.esxi.host root bad-password 'ntpd'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.service_start my.vcenter.location root bad-password 'ntpd'         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    valid_services = ['DCUI', 'TSM', 'SSH', 'ssh', 'lbtd', 'lsassd', 'lwiod', 'netlogond', 'ntpd', 'sfcbd-watchdog', 'snmpd', 'vprobed', 'vpxa', 'xorg']
    ret = {}
    if service_name == 'SSH' or service_name == 'ssh':
        temp_service_name = 'TSM-SSH'
    else:
        temp_service_name = service_name
    for host_name in host_names:
        if service_name not in valid_services:
            ret.update({host_name: {'Error': f'{service_name} is not a valid service name.'}})
            return ret
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        service_manager = _get_service_manager(host_ref)
        log.debug("Starting the '%s' service on %s.", service_name, host_name)
        try:
            service_manager.StartService(id=temp_service_name)
        except vim.fault.HostConfigFault as err:
            msg = "'vsphere.service_start' failed for host {}: {}".format(host_name, err)
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
            continue
        except vim.fault.RestrictedVersion as err:
            log.debug(err)
            ret.update({host_name: {'Error': err}})
            continue
        ret.update({host_name: {'Service Started': True}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def service_stop(host, username, password, service_name, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        return 10
    "\n    Stop the named service for the given host or list of hosts.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    service_name\n        The name of the service for which to set the policy. Supported service names are:\n          - DCUI\n          - TSM\n          - SSH\n          - lbtd\n          - lsassd\n          - lwiod\n          - netlogond\n          - ntpd\n          - sfcbd-watchdog\n          - snmpd\n          - vprobed\n          - vpxa\n          - xorg\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to stop the service.\n\n        If host_names is not provided, the service will be stopped for the ``host``\n        location instead. This is useful for when service instance connection information\n        is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.service_stop my.esxi.host root bad-password 'ssh'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.service_stop my.vcenter.location root bad-password 'ssh'         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    valid_services = ['DCUI', 'TSM', 'SSH', 'ssh', 'lbtd', 'lsassd', 'lwiod', 'netlogond', 'ntpd', 'sfcbd-watchdog', 'snmpd', 'vprobed', 'vpxa', 'xorg']
    ret = {}
    if service_name == 'SSH' or service_name == 'ssh':
        temp_service_name = 'TSM-SSH'
    else:
        temp_service_name = service_name
    for host_name in host_names:
        if service_name not in valid_services:
            ret.update({host_name: {'Error': f'{service_name} is not a valid service name.'}})
            return ret
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        service_manager = _get_service_manager(host_ref)
        log.debug("Stopping the '%s' service on %s.", service_name, host_name)
        try:
            service_manager.StopService(id=temp_service_name)
        except vim.fault.HostConfigFault as err:
            msg = f"'vsphere.service_stop' failed for host {host_name}: {err}"
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
            continue
        except vim.fault.RestrictedVersion as err:
            log.debug(err)
            ret.update({host_name: {'Error': err}})
            continue
        ret.update({host_name: {'Service Stopped': True}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def service_restart(host, username, password, service_name, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Restart the named service for the given host or list of hosts.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    service_name\n        The name of the service for which to set the policy. Supported service names are:\n          - DCUI\n          - TSM\n          - SSH\n          - lbtd\n          - lsassd\n          - lwiod\n          - netlogond\n          - ntpd\n          - sfcbd-watchdog\n          - snmpd\n          - vprobed\n          - vpxa\n          - xorg\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to restart the service.\n\n        If host_names is not provided, the service will be restarted for the ``host``\n        location instead. This is useful for when service instance connection information\n        is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.service_restart my.esxi.host root bad-password 'ntpd'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.service_restart my.vcenter.location root bad-password 'ntpd'         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    valid_services = ['DCUI', 'TSM', 'SSH', 'ssh', 'lbtd', 'lsassd', 'lwiod', 'netlogond', 'ntpd', 'sfcbd-watchdog', 'snmpd', 'vprobed', 'vpxa', 'xorg']
    ret = {}
    if service_name == 'SSH' or service_name == 'ssh':
        temp_service_name = 'TSM-SSH'
    else:
        temp_service_name = service_name
    for host_name in host_names:
        if service_name not in valid_services:
            ret.update({host_name: {'Error': f'{service_name} is not a valid service name.'}})
            return ret
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        service_manager = _get_service_manager(host_ref)
        log.debug("Restarting the '%s' service on %s.", service_name, host_name)
        try:
            service_manager.RestartService(id=temp_service_name)
        except vim.fault.HostConfigFault as err:
            msg = "'vsphere.service_restart' failed for host {}: {}".format(host_name, err)
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
            continue
        except vim.fault.RestrictedVersion as err:
            log.debug(err)
            ret.update({host_name: {'Error': err}})
            continue
        ret.update({host_name: {'Service Restarted': True}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def set_service_policy(host, username, password, service_name, service_policy, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the service name's policy for a given host or list of hosts.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    service_name\n        The name of the service for which to set the policy. Supported service names are:\n          - DCUI\n          - TSM\n          - SSH\n          - lbtd\n          - lsassd\n          - lwiod\n          - netlogond\n          - ntpd\n          - sfcbd-watchdog\n          - snmpd\n          - vprobed\n          - vpxa\n          - xorg\n\n    service_policy\n        The policy to set for the service. For example, 'automatic'.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to tell\n        vCenter the hosts for which to set the service policy.\n\n        If host_names is not provided, the service policy information will be retrieved\n        for the ``host`` location instead. This is useful for when service instance\n        connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.set_service_policy my.esxi.host root bad-password 'ntpd' 'automatic'\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.set_service_policy my.vcenter.location root bad-password 'ntpd' 'automatic'         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    valid_services = ['DCUI', 'TSM', 'SSH', 'ssh', 'lbtd', 'lsassd', 'lwiod', 'netlogond', 'ntpd', 'sfcbd-watchdog', 'snmpd', 'vprobed', 'vpxa', 'xorg']
    ret = {}
    for host_name in host_names:
        if service_name not in valid_services:
            ret.update({host_name: {'Error': f'{service_name} is not a valid service name.'}})
            return ret
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        service_manager = _get_service_manager(host_ref)
        services = host_ref.configManager.serviceSystem.serviceInfo.service
        for service in services:
            service_key = None
            if service.key == service_name:
                service_key = service.key
            elif service_name == 'ssh' or service_name == 'SSH':
                if service.key == 'TSM-SSH':
                    service_key = 'TSM-SSH'
            if service_key:
                try:
                    service_manager.UpdateServicePolicy(id=service_key, policy=service_policy)
                except vim.fault.NotFound:
                    msg = f"The service name '{service_name}' was not found."
                    log.debug(msg)
                    ret.update({host_name: {'Error': msg}})
                    continue
                except vim.fault.HostConfigFault as err:
                    msg = "'vsphere.set_service_policy' failed for host {}: {}".format(host_name, err)
                    log.debug(msg)
                    ret.update({host_name: {'Error': msg}})
                    continue
                ret.update({host_name: True})
            if ret.get(host_name) is None:
                msg = "Could not find service '{}' for host '{}'.".format(service_name, host_name)
                log.debug(msg)
                ret.update({host_name: {'Error': msg}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def update_host_datetime(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        return 10
    "\n    Update the date/time on the given host or list of host_names. This function should be\n    used with caution since network delays and execution delays can result in time skews.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts should update their date/time.\n\n        If host_names is not provided, the date/time will be updated for the ``host``\n        location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.update_date_time my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.update_date_time my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        date_time_manager = _get_date_time_mgr(host_ref)
        try:
            date_time_manager.UpdateDateTime(datetime.datetime.utcnow())
        except vim.fault.HostConfigFault as err:
            msg = "'vsphere.update_date_time' failed for host {}: {}".format(host_name, err)
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
            continue
        ret.update({host_name: {'Datetime Updated': True}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def update_host_password(host, username, password, new_password, protocol=None, port=None, verify_ssl=True):
    if False:
        print('Hello World!')
    "\n    Update the password for a given host.\n\n    .. note:: Currently only works with connections to ESXi hosts. Does not work with vCenter servers.\n\n    host\n        The location of the ESXi host.\n\n    username\n        The username used to login to the ESXi host, such as ``root``.\n\n    password\n        The password used to login to the ESXi host.\n\n    new_password\n        The new password that will be updated for the provided username on the ESXi host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.update_host_password my.esxi.host root original-bad-password new-bad-password\n\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    account_manager = salt.utils.vmware.get_inventory(service_instance).accountManager
    user_account = vim.host.LocalAccountManager.AccountSpecification()
    user_account.id = username
    user_account.password = new_password
    try:
        account_manager.UpdateUser(user_account)
    except vmodl.fault.SystemError as err:
        raise CommandExecutionError(err.msg)
    except vim.fault.UserNotFound:
        raise CommandExecutionError("'vsphere.update_host_password' failed for host {}: User was not found.".format(host))
    except vim.fault.AlreadyExists:
        pass
    return True

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def vmotion_disable(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Disable vMotion for a given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts should disable VMotion.\n\n        If host_names is not provided, VMotion will be disabled for the ``host``\n        location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.vmotion_disable my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.vmotion_disable my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vmotion_system = host_ref.configManager.vmotionSystem
        try:
            vmotion_system.DeselectVnic()
        except vim.fault.HostConfigFault as err:
            msg = f'vsphere.vmotion_disable failed: {err}'
            log.debug(msg)
            ret.update({host_name: {'Error': msg, 'VMotion Disabled': False}})
            continue
        ret.update({host_name: {'VMotion Disabled': True}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def vmotion_enable(host, username, password, protocol=None, port=None, host_names=None, device='vmk0', verify_ssl=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Enable vMotion for a given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts should enable VMotion.\n\n        If host_names is not provided, VMotion will be enabled for the ``host``\n        location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    device\n        The device that uniquely identifies the VirtualNic that will be used for\n        VMotion for each host. Defaults to ``vmk0``.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.vmotion_enable my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.vmotion_enable my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vmotion_system = host_ref.configManager.vmotionSystem
        try:
            vmotion_system.SelectVnic(device)
        except vim.fault.HostConfigFault as err:
            msg = f'vsphere.vmotion_disable failed: {err}'
            log.debug(msg)
            ret.update({host_name: {'Error': msg, 'VMotion Enabled': False}})
            continue
        ret.update({host_name: {'VMotion Enabled': True}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def vsan_add_disks(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add any VSAN-eligible disks to the VSAN System for the given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts need to add any VSAN-eligible disks to the host's\n        VSAN system.\n\n        If host_names is not provided, VSAN-eligible disks will be added to the hosts's\n        VSAN system for the ``host`` location instead. This is useful for when service\n        instance connection information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.vsan_add_disks my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.vsan_add_disks my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    host_names = _check_hosts(service_instance, host, host_names)
    response = _get_vsan_eligible_disks(service_instance, host, host_names)
    ret = {}
    for (host_name, value) in response.items():
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vsan_system = host_ref.configManager.vsanSystem
        if vsan_system is None:
            msg = "VSAN System Config Manager is unset for host '{}'. VSAN configuration cannot be changed without a configured VSAN System.".format(host_name)
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
        else:
            eligible = value.get('Eligible')
            error = value.get('Error')
            if eligible and isinstance(eligible, list):
                try:
                    task = vsan_system.AddDisks(eligible)
                    salt.utils.vmware.wait_for_task(task, host_name, 'Adding disks to VSAN', sleep_seconds=3)
                except vim.fault.InsufficientDisks as err:
                    log.debug(err.msg)
                    ret.update({host_name: {'Error': err.msg}})
                    continue
                except Exception as err:
                    msg = "'vsphere.vsan_add_disks' failed for host {}: {}".format(host_name, err)
                    log.debug(msg)
                    ret.update({host_name: {'Error': msg}})
                    continue
                log.debug("Successfully added disks to the VSAN system for host '%s'.", host_name)
                disk_names = []
                for disk in eligible:
                    disk_names.append(disk.canonicalName)
                ret.update({host_name: {'Disks Added': disk_names}})
            elif eligible and isinstance(eligible, str):
                ret.update({host_name: {'Disks Added': eligible}})
            elif error:
                ret.update({host_name: {'Error': error}})
            else:
                ret.update({host_name: {'Disks Added': 'No new VSAN-eligible disks were found to add.'}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def vsan_disable(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        while True:
            i = 10
    "\n    Disable VSAN for a given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts should disable VSAN.\n\n        If host_names is not provided, VSAN will be disabled for the ``host``\n        location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.vsan_disable my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.vsan_disable my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    vsan_config = vim.vsan.host.ConfigInfo()
    vsan_config.enabled = False
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vsan_system = host_ref.configManager.vsanSystem
        if vsan_system is None:
            msg = "VSAN System Config Manager is unset for host '{}'. VSAN configuration cannot be changed without a configured VSAN System.".format(host_name)
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
        else:
            try:
                task = vsan_system.UpdateVsan_Task(vsan_config)
                salt.utils.vmware.wait_for_task(task, host_name, 'Disabling VSAN', sleep_seconds=3)
            except vmodl.fault.SystemError as err:
                log.debug(err.msg)
                ret.update({host_name: {'Error': err.msg}})
                continue
            except Exception as err:
                msg = "'vsphere.vsan_disable' failed for host {}: {}".format(host_name, err)
                log.debug(msg)
                ret.update({host_name: {'Error': msg}})
                continue
            ret.update({host_name: {'VSAN Disabled': True}})
    return ret

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def vsan_enable(host, username, password, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        return 10
    "\n    Enable VSAN for a given host or list of host_names.\n\n    host\n        The location of the host.\n\n    username\n        The username used to login to the host, such as ``root``.\n\n    password\n        The password used to login to the host.\n\n    protocol\n        Optionally set to alternate protocol if the host is not using the default\n        protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the host is not using the default\n        port. Default port is ``443``.\n\n    host_names\n        List of ESXi host names. When the host, username, and password credentials\n        are provided for a vCenter Server, the host_names argument is required to\n        tell vCenter which hosts should enable VSAN.\n\n        If host_names is not provided, VSAN will be enabled for the ``host``\n        location instead. This is useful for when service instance connection\n        information is used for a single ESXi host.\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Used for single ESXi host connection information\n        salt '*' vsphere.vsan_enable my.esxi.host root bad-password\n\n        # Used for connecting to a vCenter Server\n        salt '*' vsphere.vsan_enable my.vcenter.location root bad-password         host_names='[esxi-1.host.com, esxi-2.host.com]'\n    "
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    vsan_config = vim.vsan.host.ConfigInfo()
    vsan_config.enabled = True
    host_names = _check_hosts(service_instance, host, host_names)
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vsan_system = host_ref.configManager.vsanSystem
        if vsan_system is None:
            msg = "VSAN System Config Manager is unset for host '{}'. VSAN configuration cannot be changed without a configured VSAN System.".format(host_name)
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
        else:
            try:
                task = vsan_system.UpdateVsan_Task(vsan_config)
                salt.utils.vmware.wait_for_task(task, host_name, 'Enabling VSAN', sleep_seconds=3)
            except vmodl.fault.SystemError as err:
                log.debug(err.msg)
                ret.update({host_name: {'Error': err.msg}})
                continue
            except vim.fault.VsanFault as err:
                msg = "'vsphere.vsan_enable' failed for host {}: {}".format(host_name, err)
                log.debug(msg)
                ret.update({host_name: {'Error': msg}})
                continue
            ret.update({host_name: {'VSAN Enabled': True}})
    return ret

def _get_dvs_config_dict(dvs_name, dvs_config):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the dict representation of the DVS config\n\n    dvs_name\n        The name of the DVS\n\n    dvs_config\n        The DVS config\n    '
    log.trace("Building the dict of the DVS '%s' config", dvs_name)
    conf_dict = {'name': dvs_name, 'contact_email': dvs_config.contact.contact, 'contact_name': dvs_config.contact.name, 'description': dvs_config.description, 'lacp_api_version': dvs_config.lacpApiVersion, 'network_resource_control_version': dvs_config.networkResourceControlVersion, 'network_resource_management_enabled': dvs_config.networkResourceManagementEnabled, 'max_mtu': dvs_config.maxMtu}
    if isinstance(dvs_config.uplinkPortPolicy, vim.DVSNameArrayUplinkPortPolicy):
        conf_dict.update({'uplink_names': dvs_config.uplinkPortPolicy.uplinkPortName})
    return conf_dict

def _get_dvs_link_discovery_protocol(dvs_name, dvs_link_disc_protocol):
    if False:
        return 10
    '\n    Returns the dict representation of the DVS link discovery protocol\n\n    dvs_name\n        The name of the DVS\n\n    dvs_link_disc_protocl\n        The DVS link discovery protocol\n    '
    log.trace("Building the dict of the DVS '%s' link discovery protocol", dvs_name)
    return {'operation': dvs_link_disc_protocol.operation, 'protocol': dvs_link_disc_protocol.protocol}

def _get_dvs_product_info(dvs_name, dvs_product_info):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the dict representation of the DVS product_info\n\n    dvs_name\n        The name of the DVS\n\n    dvs_product_info\n        The DVS product info\n    '
    log.trace("Building the dict of the DVS '%s' product info", dvs_name)
    return {'name': dvs_product_info.name, 'vendor': dvs_product_info.vendor, 'version': dvs_product_info.version}

def _get_dvs_capability(dvs_name, dvs_capability):
    if False:
        print('Hello World!')
    '\n    Returns the dict representation of the DVS product_info\n\n    dvs_name\n        The name of the DVS\n\n    dvs_capability\n        The DVS capability\n    '
    log.trace("Building the dict of the DVS '%s' capability", dvs_name)
    return {'operation_supported': dvs_capability.dvsOperationSupported, 'portgroup_operation_supported': dvs_capability.dvPortGroupOperationSupported, 'port_operation_supported': dvs_capability.dvPortOperationSupported}

def _get_dvs_infrastructure_traffic_resources(dvs_name, dvs_infra_traffic_ress):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of dict representations of the DVS infrastructure traffic\n    resource\n\n    dvs_name\n        The name of the DVS\n\n    dvs_infra_traffic_ress\n        The DVS infrastructure traffic resources\n    '
    log.trace("Building the dicts of the DVS '%s' infrastructure traffic resources", dvs_name)
    res_dicts = []
    for res in dvs_infra_traffic_ress:
        res_dict = {'key': res.key, 'limit': res.allocationInfo.limit, 'reservation': res.allocationInfo.reservation}
        if res.allocationInfo.shares:
            res_dict.update({'num_shares': res.allocationInfo.shares.shares, 'share_level': res.allocationInfo.shares.level})
        res_dicts.append(res_dict)
    return res_dicts

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_dvss(datacenter=None, dvs_names=None, service_instance=None):
    if False:
        print('Hello World!')
    "\n    Returns a list of distributed virtual switches (DVSs).\n    The list can be filtered by the datacenter or DVS names.\n\n    datacenter\n        The datacenter to look for DVSs in.\n        Default value is None.\n\n    dvs_names\n        List of DVS names to look for. If None, all DVSs are returned.\n        Default value is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_dvss\n\n        salt '*' vsphere.list_dvss dvs_names=[dvs1,dvs2]\n    "
    ret_list = []
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    for dvs in salt.utils.vmware.get_dvss(dc_ref, dvs_names, not dvs_names):
        dvs_dict = {}
        props = salt.utils.vmware.get_properties_of_managed_object(dvs, ['name', 'config', 'capability', 'networkResourcePool'])
        dvs_dict = _get_dvs_config_dict(props['name'], props['config'])
        dvs_dict.update({'product_info': _get_dvs_product_info(props['name'], props['config'].productInfo)})
        if props['config'].linkDiscoveryProtocolConfig:
            dvs_dict.update({'link_discovery_protocol': _get_dvs_link_discovery_protocol(props['name'], props['config'].linkDiscoveryProtocolConfig)})
        dvs_dict.update({'capability': _get_dvs_capability(props['name'], props['capability'])})
        if hasattr(props['config'], 'infrastructureTrafficResourceConfig'):
            dvs_dict.update({'infrastructure_traffic_resource_pools': _get_dvs_infrastructure_traffic_resources(props['name'], props['config'].infrastructureTrafficResourceConfig)})
        ret_list.append(dvs_dict)
    return ret_list

def _apply_dvs_config(config_spec, config_dict):
    if False:
        i = 10
        return i + 15
    '\n    Applies the values of the config dict dictionary to a config spec\n    (vim.VMwareDVSConfigSpec)\n    '
    if config_dict.get('name'):
        config_spec.name = config_dict['name']
    if config_dict.get('contact_email') or config_dict.get('contact_name'):
        if not config_spec.contact:
            config_spec.contact = vim.DVSContactInfo()
        config_spec.contact.contact = config_dict.get('contact_email')
        config_spec.contact.name = config_dict.get('contact_name')
    if config_dict.get('description'):
        config_spec.description = config_dict.get('description')
    if config_dict.get('max_mtu'):
        config_spec.maxMtu = config_dict.get('max_mtu')
    if config_dict.get('lacp_api_version'):
        config_spec.lacpApiVersion = config_dict.get('lacp_api_version')
    if config_dict.get('network_resource_control_version'):
        config_spec.networkResourceControlVersion = config_dict.get('network_resource_control_version')
    if config_dict.get('uplink_names'):
        if not config_spec.uplinkPortPolicy or not isinstance(config_spec.uplinkPortPolicy, vim.DVSNameArrayUplinkPortPolicy):
            config_spec.uplinkPortPolicy = vim.DVSNameArrayUplinkPortPolicy()
        config_spec.uplinkPortPolicy.uplinkPortName = config_dict['uplink_names']

def _apply_dvs_link_discovery_protocol(disc_prot_config, disc_prot_dict):
    if False:
        return 10
    '\n    Applies the values of the disc_prot_dict dictionary to a link discovery\n    protocol config object (vim.LinkDiscoveryProtocolConfig)\n    '
    disc_prot_config.operation = disc_prot_dict['operation']
    disc_prot_config.protocol = disc_prot_dict['protocol']

def _apply_dvs_product_info(product_info_spec, product_info_dict):
    if False:
        i = 10
        return i + 15
    '\n    Applies the values of the product_info_dict dictionary to a product info\n    spec (vim.DistributedVirtualSwitchProductSpec)\n    '
    if product_info_dict.get('name'):
        product_info_spec.name = product_info_dict['name']
    if product_info_dict.get('vendor'):
        product_info_spec.vendor = product_info_dict['vendor']
    if product_info_dict.get('version'):
        product_info_spec.version = product_info_dict['version']

def _apply_dvs_capability(capability_spec, capability_dict):
    if False:
        return 10
    '\n    Applies the values of the capability_dict dictionary to a DVS capability\n    object (vim.vim.DVSCapability)\n    '
    if 'operation_supported' in capability_dict:
        capability_spec.dvsOperationSupported = capability_dict['operation_supported']
    if 'port_operation_supported' in capability_dict:
        capability_spec.dvPortOperationSupported = capability_dict['port_operation_supported']
    if 'portgroup_operation_supported' in capability_dict:
        capability_spec.dvPortGroupOperationSupported = capability_dict['portgroup_operation_supported']

def _apply_dvs_infrastructure_traffic_resources(infra_traffic_resources, resource_dicts):
    if False:
        while True:
            i = 10
    '\n    Applies the values of the resource dictionaries to infra traffic resources,\n    creating the infra traffic resource if required\n    (vim.DistributedVirtualSwitchProductSpec)\n    '
    for res_dict in resource_dicts:
        filtered_traffic_resources = [r for r in infra_traffic_resources if r.key == res_dict['key']]
        if filtered_traffic_resources:
            traffic_res = filtered_traffic_resources[0]
        else:
            traffic_res = vim.DvsHostInfrastructureTrafficResource()
            traffic_res.key = res_dict['key']
            traffic_res.allocationInfo = vim.DvsHostInfrastructureTrafficResourceAllocation()
            infra_traffic_resources.append(traffic_res)
        if res_dict.get('limit'):
            traffic_res.allocationInfo.limit = res_dict['limit']
        if res_dict.get('reservation'):
            traffic_res.allocationInfo.reservation = res_dict['reservation']
        if res_dict.get('num_shares') or res_dict.get('share_level'):
            if not traffic_res.allocationInfo.shares:
                traffic_res.allocationInfo.shares = vim.SharesInfo()
        if res_dict.get('share_level'):
            traffic_res.allocationInfo.shares.level = vim.SharesLevel(res_dict['share_level'])
        if res_dict.get('num_shares'):
            traffic_res.allocationInfo.shares.shares = res_dict['num_shares']

def _apply_dvs_network_resource_pools(network_resource_pools, resource_dicts):
    if False:
        while True:
            i = 10
    '\n    Applies the values of the resource dictionaries to network resource pools,\n    creating the resource pools if required\n    (vim.DVSNetworkResourcePoolConfigSpec)\n    '
    for res_dict in resource_dicts:
        ress = [r for r in network_resource_pools if r.key == res_dict['key']]
        if ress:
            res = ress[0]
        else:
            res = vim.DVSNetworkResourcePoolConfigSpec()
            res.key = res_dict['key']
            res.allocationInfo = vim.DVSNetworkResourcePoolAllocationInfo()
            network_resource_pools.append(res)
        if res_dict.get('limit'):
            res.allocationInfo.limit = res_dict['limit']
        if res_dict.get('num_shares') and res_dict.get('share_level'):
            if not res.allocationInfo.shares:
                res.allocationInfo.shares = vim.SharesInfo()
            res.allocationInfo.shares.shares = res_dict['num_shares']
            res.allocationInfo.shares.level = vim.SharesLevel(res_dict['share_level'])

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_dvs(dvs_dict, dvs_name, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Creates a distributed virtual switch (DVS).\n\n    Note: The ``dvs_name`` param will override any name set in ``dvs_dict``.\n\n    dvs_dict\n        Dict representation of the new DVS (example in salt.states.dvs)\n\n    dvs_name\n        Name of the DVS to be created.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.create_dvs dvs dict=$dvs_dict dvs_name=dvs_name\n    "
    log.trace("Creating dvs '%s' with dict = %s", dvs_name, dvs_dict)
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    dvs_dict['name'] = dvs_name
    dvs_create_spec = vim.DVSCreateSpec()
    dvs_create_spec.configSpec = vim.VMwareDVSConfigSpec()
    _apply_dvs_config(dvs_create_spec.configSpec, dvs_dict)
    if dvs_dict.get('product_info'):
        dvs_create_spec.productInfo = vim.DistributedVirtualSwitchProductSpec()
        _apply_dvs_product_info(dvs_create_spec.productInfo, dvs_dict['product_info'])
    if dvs_dict.get('capability'):
        dvs_create_spec.capability = vim.DVSCapability()
        _apply_dvs_capability(dvs_create_spec.capability, dvs_dict['capability'])
    if dvs_dict.get('link_discovery_protocol'):
        dvs_create_spec.configSpec.linkDiscoveryProtocolConfig = vim.LinkDiscoveryProtocolConfig()
        _apply_dvs_link_discovery_protocol(dvs_create_spec.configSpec.linkDiscoveryProtocolConfig, dvs_dict['link_discovery_protocol'])
    if dvs_dict.get('infrastructure_traffic_resource_pools'):
        dvs_create_spec.configSpec.infrastructureTrafficResourceConfig = []
        _apply_dvs_infrastructure_traffic_resources(dvs_create_spec.configSpec.infrastructureTrafficResourceConfig, dvs_dict['infrastructure_traffic_resource_pools'])
    log.trace('dvs_create_spec = %s', dvs_create_spec)
    salt.utils.vmware.create_dvs(dc_ref, dvs_name, dvs_create_spec)
    if 'network_resource_management_enabled' in dvs_dict:
        dvs_refs = salt.utils.vmware.get_dvss(dc_ref, dvs_names=[dvs_name])
        if not dvs_refs:
            raise VMwareObjectRetrievalError(f"DVS '{dvs_name}' wasn't found in datacenter '{datacenter}'")
        dvs_ref = dvs_refs[0]
        salt.utils.vmware.set_dvs_network_resource_management_enabled(dvs_ref, dvs_dict['network_resource_management_enabled'])
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def update_dvs(dvs_dict, dvs, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Updates a distributed virtual switch (DVS).\n\n    Note: Updating the product info, capability, uplinks of a DVS is not\n          supported so the corresponding entries in ``dvs_dict`` will be\n          ignored.\n\n    dvs_dict\n        Dictionary with the values the DVS should be update with\n        (example in salt.states.dvs)\n\n    dvs\n        Name of the DVS to be updated.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.update_dvs dvs_dict=$dvs_dict dvs=dvs1\n    "
    log.trace("Updating dvs '%s' with dict = %s", dvs, dvs_dict)
    for prop in ['product_info', 'capability', 'uplink_names', 'name']:
        if prop in dvs_dict:
            del dvs_dict[prop]
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    dvs_refs = salt.utils.vmware.get_dvss(dc_ref, dvs_names=[dvs])
    if not dvs_refs:
        raise VMwareObjectRetrievalError(f"DVS '{dvs}' wasn't found in datacenter '{datacenter}'")
    dvs_ref = dvs_refs[0]
    dvs_props = salt.utils.vmware.get_properties_of_managed_object(dvs_ref, ['config', 'capability'])
    dvs_config = vim.VMwareDVSConfigSpec()
    skipped_properties = ['host']
    for prop in dvs_config.__dict__.keys():
        if prop in skipped_properties:
            continue
        if hasattr(dvs_props['config'], prop):
            setattr(dvs_config, prop, getattr(dvs_props['config'], prop))
    _apply_dvs_config(dvs_config, dvs_dict)
    if dvs_dict.get('link_discovery_protocol'):
        if not dvs_config.linkDiscoveryProtocolConfig:
            dvs_config.linkDiscoveryProtocolConfig = vim.LinkDiscoveryProtocolConfig()
        _apply_dvs_link_discovery_protocol(dvs_config.linkDiscoveryProtocolConfig, dvs_dict['link_discovery_protocol'])
    if dvs_dict.get('infrastructure_traffic_resource_pools'):
        if not dvs_config.infrastructureTrafficResourceConfig:
            dvs_config.infrastructureTrafficResourceConfig = []
        _apply_dvs_infrastructure_traffic_resources(dvs_config.infrastructureTrafficResourceConfig, dvs_dict['infrastructure_traffic_resource_pools'])
    log.trace('dvs_config = %s', dvs_config)
    salt.utils.vmware.update_dvs(dvs_ref, dvs_config_spec=dvs_config)
    if 'network_resource_management_enabled' in dvs_dict:
        salt.utils.vmware.set_dvs_network_resource_management_enabled(dvs_ref, dvs_dict['network_resource_management_enabled'])
    return True

def _get_dvportgroup_out_shaping(pg_name, pg_default_port_config):
    if False:
        while True:
            i = 10
    '\n    Returns the out shaping policy of a distributed virtual portgroup\n\n    pg_name\n        The name of the portgroup\n\n    pg_default_port_config\n        The dafault port config of the portgroup\n    '
    log.trace("Retrieving portgroup's '%s' out shaping config", pg_name)
    out_shaping_policy = pg_default_port_config.outShapingPolicy
    if not out_shaping_policy:
        return {}
    return {'average_bandwidth': out_shaping_policy.averageBandwidth.value, 'burst_size': out_shaping_policy.burstSize.value, 'enabled': out_shaping_policy.enabled.value, 'peak_bandwidth': out_shaping_policy.peakBandwidth.value}

def _get_dvportgroup_security_policy(pg_name, pg_default_port_config):
    if False:
        i = 10
        return i + 15
    '\n    Returns the security policy of a distributed virtual portgroup\n\n    pg_name\n        The name of the portgroup\n\n    pg_default_port_config\n        The dafault port config of the portgroup\n    '
    log.trace("Retrieving portgroup's '%s' security policy config", pg_name)
    sec_policy = pg_default_port_config.securityPolicy
    if not sec_policy:
        return {}
    return {'allow_promiscuous': sec_policy.allowPromiscuous.value, 'forged_transmits': sec_policy.forgedTransmits.value, 'mac_changes': sec_policy.macChanges.value}

def _get_dvportgroup_teaming(pg_name, pg_default_port_config):
    if False:
        print('Hello World!')
    '\n    Returns the teaming of a distributed virtual portgroup\n\n    pg_name\n        The name of the portgroup\n\n    pg_default_port_config\n        The dafault port config of the portgroup\n    '
    log.trace("Retrieving portgroup's '%s' teaming config", pg_name)
    teaming_policy = pg_default_port_config.uplinkTeamingPolicy
    if not teaming_policy:
        return {}
    ret_dict = {'notify_switches': teaming_policy.notifySwitches.value, 'policy': teaming_policy.policy.value, 'reverse_policy': teaming_policy.reversePolicy.value, 'rolling_order': teaming_policy.rollingOrder.value}
    if teaming_policy.failureCriteria:
        failure_criteria = teaming_policy.failureCriteria
        ret_dict.update({'failure_criteria': {'check_beacon': failure_criteria.checkBeacon.value, 'check_duplex': failure_criteria.checkDuplex.value, 'check_error_percent': failure_criteria.checkErrorPercent.value, 'check_speed': failure_criteria.checkSpeed.value, 'full_duplex': failure_criteria.fullDuplex.value, 'percentage': failure_criteria.percentage.value, 'speed': failure_criteria.speed.value}})
    if teaming_policy.uplinkPortOrder:
        uplink_order = teaming_policy.uplinkPortOrder
        ret_dict.update({'port_order': {'active': uplink_order.activeUplinkPort, 'standby': uplink_order.standbyUplinkPort}})
    return ret_dict

def _get_dvportgroup_dict(pg_ref):
    if False:
        return 10
    '\n    Returns a dictionary with a distributed virtual portgroup data\n\n\n    pg_ref\n        Portgroup reference\n    '
    props = salt.utils.vmware.get_properties_of_managed_object(pg_ref, ['name', 'config.description', 'config.numPorts', 'config.type', 'config.defaultPortConfig'])
    pg_dict = {'name': props['name'], 'description': props.get('config.description'), 'num_ports': props['config.numPorts'], 'type': props['config.type']}
    if props['config.defaultPortConfig']:
        dpg = props['config.defaultPortConfig']
        if dpg.vlan and isinstance(dpg.vlan, vim.VmwareDistributedVirtualSwitchVlanIdSpec):
            pg_dict.update({'vlan_id': dpg.vlan.vlanId})
        pg_dict.update({'out_shaping': _get_dvportgroup_out_shaping(props['name'], props['config.defaultPortConfig'])})
        pg_dict.update({'security_policy': _get_dvportgroup_security_policy(props['name'], props['config.defaultPortConfig'])})
        pg_dict.update({'teaming': _get_dvportgroup_teaming(props['name'], props['config.defaultPortConfig'])})
    return pg_dict

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_dvportgroups(dvs=None, portgroup_names=None, service_instance=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of distributed virtual switch portgroups.\n    The list can be filtered by the portgroup names or by the DVS.\n\n    dvs\n        Name of the DVS containing the portgroups.\n        Default value is None.\n\n    portgroup_names\n        List of portgroup names to look for. If None, all portgroups are\n        returned.\n        Default value is None\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_dvportgroups\n\n        salt '*' vsphere.list_dvportgroups dvs=dvs1\n\n        salt '*' vsphere.list_dvportgroups portgroup_names=[pg1]\n\n        salt '*' vsphere.list_dvportgroups dvs=dvs1 portgroup_names=[pg1]\n    "
    ret_dict = []
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    if dvs:
        dvs_refs = salt.utils.vmware.get_dvss(dc_ref, dvs_names=[dvs])
        if not dvs_refs:
            raise VMwareObjectRetrievalError(f"DVS '{dvs}' was not retrieved")
        dvs_ref = dvs_refs[0]
    get_all_portgroups = True if not portgroup_names else False
    for pg_ref in salt.utils.vmware.get_dvportgroups(parent_ref=dvs_ref if dvs else dc_ref, portgroup_names=portgroup_names, get_all_portgroups=get_all_portgroups):
        ret_dict.append(_get_dvportgroup_dict(pg_ref))
    return ret_dict

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_uplink_dvportgroup(dvs, service_instance=None):
    if False:
        print('Hello World!')
    "\n    Returns the uplink portgroup of a distributed virtual switch.\n\n    dvs\n        Name of the DVS containing the portgroup.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_uplink_dvportgroup dvs=dvs_name\n    "
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    dvs_refs = salt.utils.vmware.get_dvss(dc_ref, dvs_names=[dvs])
    if not dvs_refs:
        raise VMwareObjectRetrievalError(f"DVS '{dvs}' was not retrieved")
    uplink_pg_ref = salt.utils.vmware.get_uplink_dvportgroup(dvs_refs[0])
    return _get_dvportgroup_dict(uplink_pg_ref)

def _apply_dvportgroup_out_shaping(pg_name, out_shaping, out_shaping_conf):
    if False:
        print('Hello World!')
    '\n    Applies the values in out_shaping_conf to an out_shaping object\n\n    pg_name\n        The name of the portgroup\n\n    out_shaping\n        The vim.DVSTrafficShapingPolicy to apply the config to\n\n    out_shaping_conf\n        The out shaping config\n    '
    log.trace("Building portgroup's '%s' out shaping policy", pg_name)
    if out_shaping_conf.get('average_bandwidth'):
        out_shaping.averageBandwidth = vim.LongPolicy()
        out_shaping.averageBandwidth.value = out_shaping_conf['average_bandwidth']
    if out_shaping_conf.get('burst_size'):
        out_shaping.burstSize = vim.LongPolicy()
        out_shaping.burstSize.value = out_shaping_conf['burst_size']
    if 'enabled' in out_shaping_conf:
        out_shaping.enabled = vim.BoolPolicy()
        out_shaping.enabled.value = out_shaping_conf['enabled']
    if out_shaping_conf.get('peak_bandwidth'):
        out_shaping.peakBandwidth = vim.LongPolicy()
        out_shaping.peakBandwidth.value = out_shaping_conf['peak_bandwidth']

def _apply_dvportgroup_security_policy(pg_name, sec_policy, sec_policy_conf):
    if False:
        for i in range(10):
            print('nop')
    '\n    Applies the values in sec_policy_conf to a security policy object\n\n    pg_name\n        The name of the portgroup\n\n    sec_policy\n        The vim.DVSTrafficShapingPolicy to apply the config to\n\n    sec_policy_conf\n        The out shaping config\n    '
    log.trace("Building portgroup's '%s' security policy", pg_name)
    if 'allow_promiscuous' in sec_policy_conf:
        sec_policy.allowPromiscuous = vim.BoolPolicy()
        sec_policy.allowPromiscuous.value = sec_policy_conf['allow_promiscuous']
    if 'forged_transmits' in sec_policy_conf:
        sec_policy.forgedTransmits = vim.BoolPolicy()
        sec_policy.forgedTransmits.value = sec_policy_conf['forged_transmits']
    if 'mac_changes' in sec_policy_conf:
        sec_policy.macChanges = vim.BoolPolicy()
        sec_policy.macChanges.value = sec_policy_conf['mac_changes']

def _apply_dvportgroup_teaming(pg_name, teaming, teaming_conf):
    if False:
        return 10
    '\n    Applies the values in teaming_conf to a teaming policy object\n\n    pg_name\n        The name of the portgroup\n\n    teaming\n        The vim.VmwareUplinkPortTeamingPolicy to apply the config to\n\n    teaming_conf\n        The teaming config\n    '
    log.trace("Building portgroup's '%s' teaming", pg_name)
    if 'notify_switches' in teaming_conf:
        teaming.notifySwitches = vim.BoolPolicy()
        teaming.notifySwitches.value = teaming_conf['notify_switches']
    if 'policy' in teaming_conf:
        teaming.policy = vim.StringPolicy()
        teaming.policy.value = teaming_conf['policy']
    if 'reverse_policy' in teaming_conf:
        teaming.reversePolicy = vim.BoolPolicy()
        teaming.reversePolicy.value = teaming_conf['reverse_policy']
    if 'rolling_order' in teaming_conf:
        teaming.rollingOrder = vim.BoolPolicy()
        teaming.rollingOrder.value = teaming_conf['rolling_order']
    if 'failure_criteria' in teaming_conf:
        if not teaming.failureCriteria:
            teaming.failureCriteria = vim.DVSFailureCriteria()
        failure_criteria_conf = teaming_conf['failure_criteria']
        if 'check_beacon' in failure_criteria_conf:
            teaming.failureCriteria.checkBeacon = vim.BoolPolicy()
            teaming.failureCriteria.checkBeacon.value = failure_criteria_conf['check_beacon']
        if 'check_duplex' in failure_criteria_conf:
            teaming.failureCriteria.checkDuplex = vim.BoolPolicy()
            teaming.failureCriteria.checkDuplex.value = failure_criteria_conf['check_duplex']
        if 'check_error_percent' in failure_criteria_conf:
            teaming.failureCriteria.checkErrorPercent = vim.BoolPolicy()
            teaming.failureCriteria.checkErrorPercent.value = failure_criteria_conf['check_error_percent']
        if 'check_speed' in failure_criteria_conf:
            teaming.failureCriteria.checkSpeed = vim.StringPolicy()
            teaming.failureCriteria.checkSpeed.value = failure_criteria_conf['check_speed']
        if 'full_duplex' in failure_criteria_conf:
            teaming.failureCriteria.fullDuplex = vim.BoolPolicy()
            teaming.failureCriteria.fullDuplex.value = failure_criteria_conf['full_duplex']
        if 'percentage' in failure_criteria_conf:
            teaming.failureCriteria.percentage = vim.IntPolicy()
            teaming.failureCriteria.percentage.value = failure_criteria_conf['percentage']
        if 'speed' in failure_criteria_conf:
            teaming.failureCriteria.speed = vim.IntPolicy()
            teaming.failureCriteria.speed.value = failure_criteria_conf['speed']
    if 'port_order' in teaming_conf:
        if not teaming.uplinkPortOrder:
            teaming.uplinkPortOrder = vim.VMwareUplinkPortOrderPolicy()
        if 'active' in teaming_conf['port_order']:
            teaming.uplinkPortOrder.activeUplinkPort = teaming_conf['port_order']['active']
        if 'standby' in teaming_conf['port_order']:
            teaming.uplinkPortOrder.standbyUplinkPort = teaming_conf['port_order']['standby']

def _apply_dvportgroup_config(pg_name, pg_spec, pg_conf):
    if False:
        for i in range(10):
            print('nop')
    '\n    Applies the values in conf to a distributed portgroup spec\n\n    pg_name\n        The name of the portgroup\n\n    pg_spec\n        The vim.DVPortgroupConfigSpec to apply the config to\n\n    pg_conf\n        The portgroup config\n    '
    log.trace("Building portgroup's '%s' spec", pg_name)
    if 'name' in pg_conf:
        pg_spec.name = pg_conf['name']
    if 'description' in pg_conf:
        pg_spec.description = pg_conf['description']
    if 'num_ports' in pg_conf:
        pg_spec.numPorts = pg_conf['num_ports']
    if 'type' in pg_conf:
        pg_spec.type = pg_conf['type']
    if not pg_spec.defaultPortConfig:
        for prop in ['vlan_id', 'out_shaping', 'security_policy', 'teaming']:
            if prop in pg_conf:
                pg_spec.defaultPortConfig = vim.VMwareDVSPortSetting()
    if 'vlan_id' in pg_conf:
        pg_spec.defaultPortConfig.vlan = vim.VmwareDistributedVirtualSwitchVlanIdSpec()
        pg_spec.defaultPortConfig.vlan.vlanId = pg_conf['vlan_id']
    if 'out_shaping' in pg_conf:
        if not pg_spec.defaultPortConfig.outShapingPolicy:
            pg_spec.defaultPortConfig.outShapingPolicy = vim.DVSTrafficShapingPolicy()
        _apply_dvportgroup_out_shaping(pg_name, pg_spec.defaultPortConfig.outShapingPolicy, pg_conf['out_shaping'])
    if 'security_policy' in pg_conf:
        if not pg_spec.defaultPortConfig.securityPolicy:
            pg_spec.defaultPortConfig.securityPolicy = vim.DVSSecurityPolicy()
        _apply_dvportgroup_security_policy(pg_name, pg_spec.defaultPortConfig.securityPolicy, pg_conf['security_policy'])
    if 'teaming' in pg_conf:
        if not pg_spec.defaultPortConfig.uplinkTeamingPolicy:
            pg_spec.defaultPortConfig.uplinkTeamingPolicy = vim.VmwareUplinkPortTeamingPolicy()
        _apply_dvportgroup_teaming(pg_name, pg_spec.defaultPortConfig.uplinkTeamingPolicy, pg_conf['teaming'])

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_dvportgroup(portgroup_dict, portgroup_name, dvs, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Creates a distributed virtual portgroup.\n\n    Note: The ``portgroup_name`` param will override any name already set\n    in ``portgroup_dict``.\n\n    portgroup_dict\n        Dictionary with the config values the portgroup should be created with\n        (example in salt.states.dvs).\n\n    portgroup_name\n        Name of the portgroup to be created.\n\n    dvs\n        Name of the DVS that will contain the portgroup.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.create_dvportgroup portgroup_dict=<dict>\n            portgroup_name=pg1 dvs=dvs1\n    "
    log.trace("Creating portgroup '%s' in dvs '%s' with dict = %s", portgroup_name, dvs, portgroup_dict)
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    dvs_refs = salt.utils.vmware.get_dvss(dc_ref, dvs_names=[dvs])
    if not dvs_refs:
        raise VMwareObjectRetrievalError(f"DVS '{dvs}' was not retrieved")
    portgroup_dict['name'] = portgroup_name
    spec = vim.DVPortgroupConfigSpec()
    _apply_dvportgroup_config(portgroup_name, spec, portgroup_dict)
    salt.utils.vmware.create_dvportgroup(dvs_refs[0], spec)
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def update_dvportgroup(portgroup_dict, portgroup, dvs, service_instance=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Updates a distributed virtual portgroup.\n\n    portgroup_dict\n        Dictionary with the values the portgroup should be update with\n        (example in salt.states.dvs).\n\n    portgroup\n        Name of the portgroup to be updated.\n\n    dvs\n        Name of the DVS containing the portgroups.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.update_dvportgroup portgroup_dict=<dict>\n            portgroup=pg1\n\n        salt '*' vsphere.update_dvportgroup portgroup_dict=<dict>\n            portgroup=pg1 dvs=dvs1\n    "
    log.trace("Updating portgroup '%s' in dvs '%s' with dict = %s", portgroup, dvs, portgroup_dict)
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    dvs_refs = salt.utils.vmware.get_dvss(dc_ref, dvs_names=[dvs])
    if not dvs_refs:
        raise VMwareObjectRetrievalError(f"DVS '{dvs}' was not retrieved")
    pg_refs = salt.utils.vmware.get_dvportgroups(dvs_refs[0], portgroup_names=[portgroup])
    if not pg_refs:
        raise VMwareObjectRetrievalError(f"Portgroup '{portgroup}' was not retrieved")
    pg_props = salt.utils.vmware.get_properties_of_managed_object(pg_refs[0], ['config'])
    spec = vim.DVPortgroupConfigSpec()
    for prop in ['autoExpand', 'configVersion', 'defaultPortConfig', 'description', 'name', 'numPorts', 'policy', 'portNameFormat', 'scope', 'type', 'vendorSpecificConfig']:
        setattr(spec, prop, getattr(pg_props['config'], prop))
    _apply_dvportgroup_config(portgroup, spec, portgroup_dict)
    salt.utils.vmware.update_dvportgroup(pg_refs[0], spec)
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster')
@_gets_service_instance_via_proxy
@_deprecation_message
def remove_dvportgroup(portgroup, dvs, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Removes a distributed virtual portgroup.\n\n    portgroup\n        Name of the portgroup to be removed.\n\n    dvs\n        Name of the DVS containing the portgroups.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.remove_dvportgroup portgroup=pg1 dvs=dvs1\n    "
    log.trace("Removing portgroup '%s' in dvs '%s'", portgroup, dvs)
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    dvs_refs = salt.utils.vmware.get_dvss(dc_ref, dvs_names=[dvs])
    if not dvs_refs:
        raise VMwareObjectRetrievalError(f"DVS '{dvs}' was not retrieved")
    pg_refs = salt.utils.vmware.get_dvportgroups(dvs_refs[0], portgroup_names=[portgroup])
    if not pg_refs:
        raise VMwareObjectRetrievalError(f"Portgroup '{portgroup}' was not retrieved")
    salt.utils.vmware.remove_dvportgroup(pg_refs[0])
    return True

def _get_policy_dict(policy):
    if False:
        return 10
    'Returns a dictionary representation of a policy'
    profile_dict = {'name': policy.name, 'description': policy.description, 'resource_type': policy.resourceType.resourceType}
    subprofile_dicts = []
    if isinstance(policy, pbm.profile.CapabilityBasedProfile) and isinstance(policy.constraints, pbm.profile.SubProfileCapabilityConstraints):
        for subprofile in policy.constraints.subProfiles:
            subprofile_dict = {'name': subprofile.name, 'force_provision': subprofile.forceProvision}
            cap_dicts = []
            for cap in subprofile.capability:
                cap_dict = {'namespace': cap.id.namespace, 'id': cap.id.id}
                val = cap.constraint[0].propertyInstance[0].value
                if isinstance(val, pbm.capability.types.Range):
                    val_dict = {'type': 'range', 'min': val.min, 'max': val.max}
                elif isinstance(val, pbm.capability.types.DiscreteSet):
                    val_dict = {'type': 'set', 'values': val.values}
                else:
                    val_dict = {'type': 'scalar', 'value': val}
                cap_dict['setting'] = val_dict
                cap_dicts.append(cap_dict)
            subprofile_dict['capabilities'] = cap_dicts
            subprofile_dicts.append(subprofile_dict)
    profile_dict['subprofiles'] = subprofile_dicts
    return profile_dict

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_storage_policies(policy_names=None, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of storage policies.\n\n    policy_names\n        Names of policies to list. If None, all policies are listed.\n        Default is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_storage_policies\n\n        salt '*' vsphere.list_storage_policies policy_names=[policy_name]\n    "
    profile_manager = salt.utils.pbm.get_profile_manager(service_instance)
    if not policy_names:
        policies = salt.utils.pbm.get_storage_policies(profile_manager, get_all_policies=True)
    else:
        policies = salt.utils.pbm.get_storage_policies(profile_manager, policy_names)
    return [_get_policy_dict(p) for p in policies]

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_default_vsan_policy(service_instance=None):
    if False:
        return 10
    "\n    Returns the default vsan storage policy.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_default_vsan_policy\n    "
    profile_manager = salt.utils.pbm.get_profile_manager(service_instance)
    policies = salt.utils.pbm.get_storage_policies(profile_manager, get_all_policies=True)
    def_policies = [p for p in policies if p.systemCreatedProfileType == 'VsanDefaultProfile']
    if not def_policies:
        raise VMwareObjectRetrievalError('Default VSAN policy was not retrieved')
    return _get_policy_dict(def_policies[0])

def _get_capability_definition_dict(cap_metadata):
    if False:
        for i in range(10):
            print('nop')
    return {'namespace': cap_metadata.id.namespace, 'id': cap_metadata.id.id, 'mandatory': cap_metadata.mandatory, 'description': cap_metadata.summary.summary, 'type': cap_metadata.propertyMetadata[0].type.typeName}

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_capability_definitions(service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of the metadata of all capabilities in the vCenter.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_capabilities\n    "
    profile_manager = salt.utils.pbm.get_profile_manager(service_instance)
    ret_list = [_get_capability_definition_dict(c) for c in salt.utils.pbm.get_capability_definitions(profile_manager)]
    return ret_list

def _apply_policy_config(policy_spec, policy_dict):
    if False:
        print('Hello World!')
    'Applies a policy dictionary to a policy spec'
    log.trace('policy_dict = %s', policy_dict)
    if policy_dict.get('name'):
        policy_spec.name = policy_dict['name']
    if policy_dict.get('description'):
        policy_spec.description = policy_dict['description']
    if policy_dict.get('subprofiles'):
        policy_spec.constraints = pbm.profile.SubProfileCapabilityConstraints()
        subprofiles = []
        for subprofile_dict in policy_dict['subprofiles']:
            subprofile_spec = pbm.profile.SubProfileCapabilityConstraints.SubProfile(name=subprofile_dict['name'])
            cap_specs = []
            if subprofile_dict.get('force_provision'):
                subprofile_spec.forceProvision = subprofile_dict['force_provision']
            for cap_dict in subprofile_dict['capabilities']:
                prop_inst_spec = pbm.capability.PropertyInstance(id=cap_dict['id'])
                setting_type = cap_dict['setting']['type']
                if setting_type == 'set':
                    prop_inst_spec.value = pbm.capability.types.DiscreteSet()
                    prop_inst_spec.value.values = cap_dict['setting']['values']
                elif setting_type == 'range':
                    prop_inst_spec.value = pbm.capability.types.Range()
                    prop_inst_spec.value.max = cap_dict['setting']['max']
                    prop_inst_spec.value.min = cap_dict['setting']['min']
                elif setting_type == 'scalar':
                    prop_inst_spec.value = cap_dict['setting']['value']
                cap_spec = pbm.capability.CapabilityInstance(id=pbm.capability.CapabilityMetadata.UniqueId(id=cap_dict['id'], namespace=cap_dict['namespace']), constraint=[pbm.capability.ConstraintInstance(propertyInstance=[prop_inst_spec])])
                cap_specs.append(cap_spec)
            subprofile_spec.capability = cap_specs
            subprofiles.append(subprofile_spec)
        policy_spec.constraints.subProfiles = subprofiles
    log.trace('updated policy_spec = %s', policy_spec)
    return policy_spec

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_storage_policy(policy_name, policy_dict, service_instance=None):
    if False:
        return 10
    '\n    Creates a storage policy.\n\n    Supported capability types: scalar, set, range.\n\n    policy_name\n        Name of the policy to create.\n        The value of the argument will override any existing name in\n        ``policy_dict``.\n\n    policy_dict\n        Dictionary containing the changes to apply to the policy.\n        (example in salt.states.pbm)\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' vsphere.create_storage_policy policy_name=\'policy name\'\n            policy_dict="$policy_dict"\n    '
    log.trace("create storage policy '%s', dict = %s", policy_name, policy_dict)
    profile_manager = salt.utils.pbm.get_profile_manager(service_instance)
    policy_create_spec = pbm.profile.CapabilityBasedProfileCreateSpec()
    policy_create_spec.resourceType = pbm.profile.ResourceType(resourceType=pbm.profile.ResourceTypeEnum.STORAGE)
    policy_dict['name'] = policy_name
    log.trace('Setting policy values in policy_update_spec')
    _apply_policy_config(policy_create_spec, policy_dict)
    salt.utils.pbm.create_storage_policy(profile_manager, policy_create_spec)
    return {'create_storage_policy': True}

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def update_storage_policy(policy, policy_dict, service_instance=None):
    if False:
        i = 10
        return i + 15
    '\n    Updates a storage policy.\n\n    Supported capability types: scalar, set, range.\n\n    policy\n        Name of the policy to update.\n\n    policy_dict\n        Dictionary containing the changes to apply to the policy.\n        (example in salt.states.pbm)\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' vsphere.update_storage_policy policy=\'policy name\'\n            policy_dict="$policy_dict"\n    '
    log.trace('updating storage policy, dict = %s', policy_dict)
    profile_manager = salt.utils.pbm.get_profile_manager(service_instance)
    policies = salt.utils.pbm.get_storage_policies(profile_manager, [policy])
    if not policies:
        raise VMwareObjectRetrievalError(f"Policy '{policy}' was not found")
    policy_ref = policies[0]
    policy_update_spec = pbm.profile.CapabilityBasedProfileUpdateSpec()
    log.trace('Setting policy values in policy_update_spec')
    for prop in ['description', 'constraints']:
        setattr(policy_update_spec, prop, getattr(policy_ref, prop))
    _apply_policy_config(policy_update_spec, policy_dict)
    salt.utils.pbm.update_storage_policy(profile_manager, policy_ref, policy_update_spec)
    return {'update_storage_policy': True}

@depends(HAS_PYVMOMI)
@_supports_proxies('esxcluster', 'esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_default_storage_policy_of_datastore(datastore, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of datastores assign the storage policies.\n\n    datastore\n        Name of the datastore to assign.\n        The datastore needs to be visible to the VMware entity the proxy\n        points to.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_default_storage_policy_of_datastore datastore=ds1\n    "
    log.trace("Listing the default storage policy of datastore '%s'", datastore)
    target_ref = _get_proxy_target(service_instance)
    ds_refs = salt.utils.vmware.get_datastores(service_instance, target_ref, datastore_names=[datastore])
    if not ds_refs:
        raise VMwareObjectRetrievalError(f"Datastore '{datastore}' was not found")
    profile_manager = salt.utils.pbm.get_profile_manager(service_instance)
    policy = salt.utils.pbm.get_default_storage_policy_of_datastore(profile_manager, ds_refs[0])
    return _get_policy_dict(policy)

@depends(HAS_PYVMOMI)
@_supports_proxies('esxcluster', 'esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def assign_default_storage_policy_to_datastore(policy, datastore, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Assigns a storage policy as the default policy to a datastore.\n\n    policy\n        Name of the policy to assign.\n\n    datastore\n        Name of the datastore to assign.\n        The datastore needs to be visible to the VMware entity the proxy\n        points to.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.assign_storage_policy_to_datastore\n            policy='policy name' datastore=ds1\n    "
    log.trace('Assigning policy %s to datastore %s', policy, datastore)
    profile_manager = salt.utils.pbm.get_profile_manager(service_instance)
    policies = salt.utils.pbm.get_storage_policies(profile_manager, [policy])
    if not policies:
        raise VMwareObjectRetrievalError(f"Policy '{policy}' was not found")
    policy_ref = policies[0]
    target_ref = _get_proxy_target(service_instance)
    ds_refs = salt.utils.vmware.get_datastores(service_instance, target_ref, datastore_names=[datastore])
    if not ds_refs:
        raise VMwareObjectRetrievalError(f"Datastore '{datastore}' was not found")
    ds_ref = ds_refs[0]
    salt.utils.pbm.assign_default_storage_policy_to_datastore(profile_manager, policy_ref, ds_ref)
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'esxcluster', 'vcenter', 'esxvm')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_datacenters_via_proxy(datacenter_names=None, service_instance=None):
    if False:
        return 10
    "\n    Returns a list of dict representations of VMware datacenters.\n    Connection is done via the proxy details.\n\n    Supported proxies: esxdatacenter\n\n    datacenter_names\n        List of datacenter names.\n        Default is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_datacenters_via_proxy\n\n        salt '*' vsphere.list_datacenters_via_proxy dc1\n\n        salt '*' vsphere.list_datacenters_via_proxy dc1,dc2\n\n        salt '*' vsphere.list_datacenters_via_proxy datacenter_names=[dc1, dc2]\n    "
    if not datacenter_names:
        dc_refs = salt.utils.vmware.get_datacenters(service_instance, get_all_datacenters=True)
    else:
        dc_refs = salt.utils.vmware.get_datacenters(service_instance, datacenter_names)
    return [{'name': salt.utils.vmware.get_managed_object_name(dc_ref)} for dc_ref in dc_refs]

@depends(HAS_PYVMOMI)
@_supports_proxies('esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_datacenter(datacenter_name, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a datacenter.\n\n    Supported proxies: esxdatacenter\n\n    datacenter_name\n        The datacenter name\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.create_datacenter dc1\n    "
    salt.utils.vmware.create_datacenter(service_instance, datacenter_name)
    return {'create_datacenter': True}

def _get_cluster_dict(cluster_name, cluster_ref):
    if False:
        return 10
    '\n    Returns a cluster dict representation from\n    a vim.ClusterComputeResource object.\n\n    cluster_name\n        Name of the cluster\n\n    cluster_ref\n        Reference to the cluster\n    '
    log.trace("Building a dictionary representation of cluster '%s'", cluster_name)
    props = salt.utils.vmware.get_properties_of_managed_object(cluster_ref, properties=['configurationEx'])
    res = {'ha': {'enabled': props['configurationEx'].dasConfig.enabled}, 'drs': {'enabled': props['configurationEx'].drsConfig.enabled}}
    ha_conf = props['configurationEx'].dasConfig
    log.trace('ha_conf = %s', ha_conf)
    res['ha']['admission_control_enabled'] = ha_conf.admissionControlEnabled
    if ha_conf.admissionControlPolicy and isinstance(ha_conf.admissionControlPolicy, vim.ClusterFailoverResourcesAdmissionControlPolicy):
        pol = ha_conf.admissionControlPolicy
        res['ha']['admission_control_policy'] = {'cpu_failover_percent': pol.cpuFailoverResourcesPercent, 'memory_failover_percent': pol.memoryFailoverResourcesPercent}
    if ha_conf.defaultVmSettings:
        def_vm_set = ha_conf.defaultVmSettings
        res['ha']['default_vm_settings'] = {'isolation_response': def_vm_set.isolationResponse, 'restart_priority': def_vm_set.restartPriority}
    res['ha']['hb_ds_candidate_policy'] = ha_conf.hBDatastoreCandidatePolicy
    if ha_conf.hostMonitoring:
        res['ha']['host_monitoring'] = ha_conf.hostMonitoring
    if ha_conf.option:
        res['ha']['options'] = [{'key': o.key, 'value': o.value} for o in ha_conf.option]
    res['ha']['vm_monitoring'] = ha_conf.vmMonitoring
    drs_conf = props['configurationEx'].drsConfig
    log.trace('drs_conf = %s', drs_conf)
    res['drs']['vmotion_rate'] = 6 - drs_conf.vmotionRate
    res['drs']['default_vm_behavior'] = drs_conf.defaultVmBehavior
    res['vm_swap_placement'] = props['configurationEx'].vmSwapPlacement
    si = salt.utils.vmware.get_service_instance_from_managed_object(cluster_ref)
    if salt.utils.vsan.vsan_supported(si):
        vcenter_info = salt.utils.vmware.get_service_info(si)
        if int(vcenter_info.build) >= 3634794:
            vsan_conf = salt.utils.vsan.get_cluster_vsan_info(cluster_ref)
            log.trace('vsan_conf = %s', vsan_conf)
            res['vsan'] = {'enabled': vsan_conf.enabled, 'auto_claim_storage': vsan_conf.defaultConfig.autoClaimStorage}
            if vsan_conf.dataEfficiencyConfig:
                data_eff = vsan_conf.dataEfficiencyConfig
                res['vsan'].update({'compression_enabled': data_eff.compressionEnabled or False, 'dedup_enabled': data_eff.dedupEnabled})
        elif props['configurationEx'].vsanConfigInfo:
            default_config = props['configurationEx'].vsanConfigInfo.defaultConfig
            res['vsan'] = {'enabled': props['configurationEx'].vsanConfigInfo.enabled, 'auto_claim_storage': default_config.autoClaimStorage}
    return res

@depends(HAS_PYVMOMI)
@_supports_proxies('esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_cluster(datacenter=None, cluster=None, service_instance=None):
    if False:
        return 10
    "\n    Returns a dict representation of an ESX cluster.\n\n    datacenter\n        Name of datacenter containing the cluster.\n        Ignored if already contained by proxy details.\n        Default value is None.\n\n    cluster\n        Name of cluster.\n        Ignored if already contained by proxy details.\n        Default value is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # vcenter proxy\n        salt '*' vsphere.list_cluster datacenter=dc1 cluster=cl1\n\n        # esxdatacenter proxy\n        salt '*' vsphere.list_cluster cluster=cl1\n\n        # esxcluster proxy\n        salt '*' vsphere.list_cluster\n    "
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        dc_ref = _get_proxy_target(service_instance)
        if not cluster:
            raise ArgumentValueError("'cluster' needs to be specified")
        cluster_ref = salt.utils.vmware.get_cluster(dc_ref, cluster)
    elif proxy_type == 'esxcluster':
        cluster_ref = _get_proxy_target(service_instance)
        cluster = __salt__['esxcluster.get_details']()['cluster']
    log.trace("Retrieving representation of cluster '%s' in a %s proxy", cluster, proxy_type)
    return _get_cluster_dict(cluster, cluster_ref)

def _apply_cluster_dict(cluster_spec, cluster_dict, vsan_spec=None, vsan_61=True):
    if False:
        return 10
    "\n    Applies the values of cluster_dict dictionary to a cluster spec\n    (vim.ClusterConfigSpecEx).\n\n    All vsan values (cluster_dict['vsan']) will be applied to\n    vsan_spec (vim.vsan.cluster.ConfigInfoEx). Can be not omitted\n    if not required.\n\n    VSAN 6.1 config needs to be applied differently than the post VSAN 6.1 way.\n    The type of configuration desired is dictated by the flag vsan_61.\n    "
    log.trace('Applying cluster dict %s', cluster_dict)
    if cluster_dict.get('ha'):
        ha_dict = cluster_dict['ha']
        if not cluster_spec.dasConfig:
            cluster_spec.dasConfig = vim.ClusterDasConfigInfo()
        das_config = cluster_spec.dasConfig
        if 'enabled' in ha_dict:
            das_config.enabled = ha_dict['enabled']
            if ha_dict['enabled']:
                das_config.failoverLevel = 1
        if 'admission_control_enabled' in ha_dict:
            das_config.admissionControlEnabled = ha_dict['admission_control_enabled']
        if 'admission_control_policy' in ha_dict:
            adm_pol_dict = ha_dict['admission_control_policy']
            if not das_config.admissionControlPolicy or not isinstance(das_config.admissionControlPolicy, vim.ClusterFailoverResourcesAdmissionControlPolicy):
                das_config.admissionControlPolicy = vim.ClusterFailoverResourcesAdmissionControlPolicy(cpuFailoverResourcesPercent=adm_pol_dict['cpu_failover_percent'], memoryFailoverResourcesPercent=adm_pol_dict['memory_failover_percent'])
        if 'default_vm_settings' in ha_dict:
            vm_set_dict = ha_dict['default_vm_settings']
            if not das_config.defaultVmSettings:
                das_config.defaultVmSettings = vim.ClusterDasVmSettings()
            if 'isolation_response' in vm_set_dict:
                das_config.defaultVmSettings.isolationResponse = vm_set_dict['isolation_response']
            if 'restart_priority' in vm_set_dict:
                das_config.defaultVmSettings.restartPriority = vm_set_dict['restart_priority']
        if 'hb_ds_candidate_policy' in ha_dict:
            das_config.hBDatastoreCandidatePolicy = ha_dict['hb_ds_candidate_policy']
        if 'host_monitoring' in ha_dict:
            das_config.hostMonitoring = ha_dict['host_monitoring']
        if 'options' in ha_dict:
            das_config.option = []
            for opt_dict in ha_dict['options']:
                das_config.option.append(vim.OptionValue(key=opt_dict['key']))
                if 'value' in opt_dict:
                    das_config.option[-1].value = opt_dict['value']
        if 'vm_monitoring' in ha_dict:
            das_config.vmMonitoring = ha_dict['vm_monitoring']
        cluster_spec.dasConfig = das_config
    if cluster_dict.get('drs'):
        drs_dict = cluster_dict['drs']
        drs_config = vim.ClusterDrsConfigInfo()
        if 'enabled' in drs_dict:
            drs_config.enabled = drs_dict['enabled']
        if 'vmotion_rate' in drs_dict:
            drs_config.vmotionRate = 6 - drs_dict['vmotion_rate']
        if 'default_vm_behavior' in drs_dict:
            drs_config.defaultVmBehavior = vim.DrsBehavior(drs_dict['default_vm_behavior'])
        cluster_spec.drsConfig = drs_config
    if cluster_dict.get('vm_swap_placement'):
        cluster_spec.vmSwapPlacement = cluster_dict['vm_swap_placement']
    if cluster_dict.get('vsan'):
        vsan_dict = cluster_dict['vsan']
        if not vsan_61:
            if 'enabled' in vsan_dict:
                if not vsan_spec.vsanClusterConfig:
                    vsan_spec.vsanClusterConfig = vim.vsan.cluster.ConfigInfo()
                vsan_spec.vsanClusterConfig.enabled = vsan_dict['enabled']
            if 'auto_claim_storage' in vsan_dict:
                if not vsan_spec.vsanClusterConfig:
                    vsan_spec.vsanClusterConfig = vim.vsan.cluster.ConfigInfo()
                if not vsan_spec.vsanClusterConfig.defaultConfig:
                    vsan_spec.vsanClusterConfig.defaultConfig = vim.VsanClusterConfigInfoHostDefaultInfo()
                elif vsan_spec.vsanClusterConfig.defaultConfig.uuid:
                    vsan_spec.vsanClusterConfig.defaultConfig.uuid = None
                vsan_spec.vsanClusterConfig.defaultConfig.autoClaimStorage = vsan_dict['auto_claim_storage']
            if 'compression_enabled' in vsan_dict:
                if not vsan_spec.dataEfficiencyConfig:
                    vsan_spec.dataEfficiencyConfig = vim.vsan.DataEfficiencyConfig()
                vsan_spec.dataEfficiencyConfig.compressionEnabled = vsan_dict['compression_enabled']
            if 'dedup_enabled' in vsan_dict:
                if not vsan_spec.dataEfficiencyConfig:
                    vsan_spec.dataEfficiencyConfig = vim.vsan.DataEfficiencyConfig()
                vsan_spec.dataEfficiencyConfig.dedupEnabled = vsan_dict['dedup_enabled']
        if not cluster_spec.vsanConfig:
            cluster_spec.vsanConfig = vim.VsanClusterConfigInfo()
        vsan_config = cluster_spec.vsanConfig
        if 'enabled' in vsan_dict:
            vsan_config.enabled = vsan_dict['enabled']
        if 'auto_claim_storage' in vsan_dict:
            if not vsan_config.defaultConfig:
                vsan_config.defaultConfig = vim.VsanClusterConfigInfoHostDefaultInfo()
            elif vsan_config.defaultConfig.uuid:
                vsan_config.defaultConfig.uuid = None
            vsan_config.defaultConfig.autoClaimStorage = vsan_dict['auto_claim_storage']
    log.trace('cluster_spec = %s', cluster_spec)

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_cluster(cluster_dict, datacenter=None, cluster=None, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Creates a cluster.\n\n    Note: cluster_dict['name'] will be overridden by the cluster param value\n\n    config_dict\n        Dictionary with the config values of the new cluster.\n\n    datacenter\n        Name of datacenter containing the cluster.\n        Ignored if already contained by proxy details.\n        Default value is None.\n\n    cluster\n        Name of cluster.\n        Ignored if already contained by proxy details.\n        Default value is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # esxdatacenter proxy\n        salt '*' vsphere.create_cluster cluster_dict=$cluster_dict cluster=cl1\n\n        # esxcluster proxy\n        salt '*' vsphere.create_cluster cluster_dict=$cluster_dict\n    "
    schema = ESXClusterConfigSchema.serialize()
    try:
        jsonschema.validate(cluster_dict, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise InvalidConfigError(exc)
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
        if not cluster:
            raise ArgumentValueError("'cluster' needs to be specified")
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
        cluster = __salt__['esxcluster.get_details']()['cluster']
    if cluster_dict.get('vsan') and (not salt.utils.vsan.vsan_supported(service_instance)):
        raise VMwareApiError('VSAN operations are not supported')
    si = service_instance
    cluster_spec = vim.ClusterConfigSpecEx()
    vsan_spec = None
    ha_config = None
    vsan_61 = None
    if cluster_dict.get('vsan'):
        vcenter_info = salt.utils.vmware.get_service_info(si)
        if float(vcenter_info.apiVersion) >= 6.0 and int(vcenter_info.build) >= 3634794:
            vsan_spec = vim.vsan.ReconfigSpec(modify=True)
            vsan_61 = False
            if cluster_dict.get('ha', {}).get('enabled'):
                enable_ha = True
                ha_config = cluster_dict['ha']
                del cluster_dict['ha']
        else:
            vsan_61 = True
    _apply_cluster_dict(cluster_spec, cluster_dict, vsan_spec, vsan_61)
    salt.utils.vmware.create_cluster(dc_ref, cluster, cluster_spec)
    if not vsan_61:
        if vsan_spec:
            cluster_ref = salt.utils.vmware.get_cluster(dc_ref, cluster)
            salt.utils.vsan.reconfigure_cluster_vsan(cluster_ref, vsan_spec)
        if enable_ha:
            _apply_cluster_dict(cluster_spec, {'ha': ha_config})
            salt.utils.vmware.update_cluster(cluster_ref, cluster_spec)
            cluster_dict['ha'] = ha_config
    return {'create_cluster': True}

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def update_cluster(cluster_dict, datacenter=None, cluster=None, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Updates a cluster.\n\n    config_dict\n        Dictionary with the config values of the new cluster.\n\n    datacenter\n        Name of datacenter containing the cluster.\n        Ignored if already contained by proxy details.\n        Default value is None.\n\n    cluster\n        Name of cluster.\n        Ignored if already contained by proxy details.\n        Default value is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # esxdatacenter proxy\n        salt '*' vsphere.update_cluster cluster_dict=$cluster_dict cluster=cl1\n\n        # esxcluster proxy\n        salt '*' vsphere.update_cluster cluster_dict=$cluster_dict\n\n    "
    schema = ESXClusterConfigSchema.serialize()
    try:
        jsonschema.validate(cluster_dict, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise InvalidConfigError(exc)
    proxy_type = get_proxy_type()
    if proxy_type == 'esxdatacenter':
        datacenter = __salt__['esxdatacenter.get_details']()['datacenter']
        dc_ref = _get_proxy_target(service_instance)
        if not cluster:
            raise ArgumentValueError("'cluster' needs to be specified")
    elif proxy_type == 'esxcluster':
        datacenter = __salt__['esxcluster.get_details']()['datacenter']
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
        cluster = __salt__['esxcluster.get_details']()['cluster']
    if cluster_dict.get('vsan') and (not salt.utils.vsan.vsan_supported(service_instance)):
        raise VMwareApiError('VSAN operations are not supported')
    cluster_ref = salt.utils.vmware.get_cluster(dc_ref, cluster)
    cluster_spec = vim.ClusterConfigSpecEx()
    props = salt.utils.vmware.get_properties_of_managed_object(cluster_ref, properties=['configurationEx'])
    for p in ['dasConfig', 'drsConfig']:
        setattr(cluster_spec, p, getattr(props['configurationEx'], p))
    if props['configurationEx'].vsanConfigInfo:
        cluster_spec.vsanConfig = props['configurationEx'].vsanConfigInfo
    vsan_spec = None
    vsan_61 = None
    if cluster_dict.get('vsan'):
        vcenter_info = salt.utils.vmware.get_service_info(service_instance)
        if float(vcenter_info.apiVersion) >= 6.0 and int(vcenter_info.build) >= 3634794:
            vsan_61 = False
            vsan_info = salt.utils.vsan.get_cluster_vsan_info(cluster_ref)
            vsan_spec = vim.vsan.ReconfigSpec(modify=True)
            vsan_spec.dataEfficiencyConfig = vsan_info.dataEfficiencyConfig
            vsan_info.dataEfficiencyConfig = None
        else:
            vsan_61 = True
    _apply_cluster_dict(cluster_spec, cluster_dict, vsan_spec, vsan_61)
    if vsan_spec:
        log.trace('vsan_spec = %s', vsan_spec)
        salt.utils.vsan.reconfigure_cluster_vsan(cluster_ref, vsan_spec)
        cluster_spec = vim.ClusterConfigSpecEx()
        props = salt.utils.vmware.get_properties_of_managed_object(cluster_ref, properties=['configurationEx'])
        for p in ['dasConfig', 'drsConfig']:
            setattr(cluster_spec, p, getattr(props['configurationEx'], p))
        if props['configurationEx'].vsanConfigInfo:
            cluster_spec.vsanConfig = props['configurationEx'].vsanConfigInfo
        _apply_cluster_dict(cluster_spec, cluster_dict)
    salt.utils.vmware.update_cluster(cluster_ref, cluster_spec)
    return {'update_cluster': True}

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_datastores_via_proxy(datastore_names=None, backing_disk_ids=None, backing_disk_scsi_addresses=None, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of dict representations of the datastores visible to the\n    proxy object. The list of datastores can be filtered by datastore names,\n    backing disk ids (canonical names) or backing disk scsi addresses.\n\n    Supported proxy types: esxi, esxcluster, esxdatacenter\n\n    datastore_names\n        List of the names of datastores to filter on\n\n    backing_disk_ids\n        List of canonical names of the backing disks of the datastores to filer.\n        Default is None.\n\n    backing_disk_scsi_addresses\n        List of scsi addresses of the backing disks of the datastores to filter.\n        Default is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_datastores_via_proxy\n\n        salt '*' vsphere.list_datastores_via_proxy datastore_names=[ds1, ds2]\n    "
    target = _get_proxy_target(service_instance)
    target_name = salt.utils.vmware.get_managed_object_name(target)
    log.trace('target name = %s', target_name)
    get_all_datastores = True if not (datastore_names or backing_disk_ids or backing_disk_scsi_addresses) else False
    if backing_disk_scsi_addresses:
        log.debug("Retrieving disk ids for scsi addresses '%s'", backing_disk_scsi_addresses)
        disk_ids = [d.canonicalName for d in salt.utils.vmware.get_disks(target, scsi_addresses=backing_disk_scsi_addresses)]
        log.debug("Found disk ids '%s'", disk_ids)
        backing_disk_ids = backing_disk_ids.extend(disk_ids) if backing_disk_ids else disk_ids
    datastores = salt.utils.vmware.get_datastores(service_instance, target, datastore_names, backing_disk_ids, get_all_datastores)
    mount_infos = []
    if isinstance(target, vim.HostSystem):
        storage_system = salt.utils.vmware.get_storage_system(service_instance, target, target_name)
        props = salt.utils.vmware.get_properties_of_managed_object(storage_system, ['fileSystemVolumeInfo.mountInfo'])
        mount_infos = props.get('fileSystemVolumeInfo.mountInfo', [])
    ret_dict = []
    for ds in datastores:
        ds_dict = {'name': ds.name, 'type': ds.summary.type, 'free_space': ds.summary.freeSpace, 'capacity': ds.summary.capacity}
        backing_disk_ids = []
        for vol in [i.volume for i in mount_infos if i.volume.name == ds.name and isinstance(i.volume, vim.HostVmfsVolume)]:
            backing_disk_ids.extend([e.diskName for e in vol.extent])
        if backing_disk_ids:
            ds_dict['backing_disk_ids'] = backing_disk_ids
        ret_dict.append(ds_dict)
    return ret_dict

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_vmfs_datastore(datastore_name, disk_id, vmfs_major_version, safety_checks=True, service_instance=None):
    if False:
        print('Hello World!')
    "\n    Creates a ESXi host disk group with the specified cache and capacity disks.\n\n    datastore_name\n        The name of the datastore to be created.\n\n    disk_id\n        The disk id (canonical name) on which the datastore is created.\n\n    vmfs_major_version\n        The VMFS major version.\n\n    safety_checks\n        Specify whether to perform safety check or to skip the checks and try\n        performing the required task. Default is True.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.create_vmfs_datastore datastore_name=ds1 disk_id=\n            vmfs_major_version=5\n    "
    log.debug('Validating vmfs datastore input')
    schema = VmfsDatastoreSchema.serialize()
    try:
        jsonschema.validate({'datastore': {'name': datastore_name, 'backing_disk_id': disk_id, 'vmfs_version': vmfs_major_version}}, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ArgumentValueError(exc)
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    if safety_checks:
        disks = salt.utils.vmware.get_disks(host_ref, disk_ids=[disk_id])
        if not disks:
            raise VMwareObjectRetrievalError(f"Disk '{disk_id}' was not found in host '{hostname}'")
    ds_ref = salt.utils.vmware.create_vmfs_datastore(host_ref, datastore_name, disks[0], vmfs_major_version)
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def rename_datastore(datastore_name, new_datastore_name, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Renames a datastore. The datastore needs to be visible to the proxy.\n\n    datastore_name\n        Current datastore name.\n\n    new_datastore_name\n        New datastore name.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.rename_datastore old_name new_name\n    "
    log.trace('Renaming datastore %s to %s', datastore_name, new_datastore_name)
    target = _get_proxy_target(service_instance)
    datastores = salt.utils.vmware.get_datastores(service_instance, target, datastore_names=[datastore_name])
    if not datastores:
        raise VMwareObjectRetrievalError(f"Datastore '{datastore_name}' was not found")
    ds = datastores[0]
    salt.utils.vmware.rename_datastore(ds, new_datastore_name)
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def remove_datastore(datastore, service_instance=None):
    if False:
        i = 10
        return i + 15
    "\n    Removes a datastore. If multiple datastores an error is raised.\n\n    datastore\n        Datastore name\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.remove_datastore ds_name\n    "
    log.trace("Removing datastore '%s'", datastore)
    target = _get_proxy_target(service_instance)
    datastores = salt.utils.vmware.get_datastores(service_instance, reference=target, datastore_names=[datastore])
    if not datastores:
        raise VMwareObjectRetrievalError(f"Datastore '{datastore}' was not found")
    if len(datastores) > 1:
        raise VMwareObjectRetrievalError(f"Multiple datastores '{datastore}' were found")
    salt.utils.vmware.remove_datastore(service_instance, datastores[0])
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_licenses(service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Lists all licenses on a vCenter.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_licenses\n    "
    log.trace('Retrieving all licenses')
    licenses = salt.utils.vmware.get_licenses(service_instance)
    ret_dict = [{'key': l.licenseKey, 'name': l.name, 'description': l.labels[0].value if l.labels else None, 'capacity': l.total if l.total > 0 else sys.maxsize, 'used': l.used if l.used else 0} for l in licenses]
    return ret_dict

@depends(HAS_PYVMOMI)
@_supports_proxies('esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def add_license(key, description, safety_checks=True, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Adds a license to the vCenter or ESXi host\n\n    key\n        License key.\n\n    description\n        License description added in as a label.\n\n    safety_checks\n        Specify whether to perform safety check or to skip the checks and try\n        performing the required task\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.add_license key=<license_key> desc='License desc'\n    "
    log.trace("Adding license '%s'", key)
    salt.utils.vmware.add_license(service_instance, key, description)
    return True

def _get_entity(service_instance, entity):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the entity associated with the entity dict representation\n\n    Supported entities: cluster, vcenter\n\n    Expected entity format:\n\n    .. code-block:: python\n\n        cluster:\n            {'type': 'cluster',\n             'datacenter': <datacenter_name>,\n             'cluster': <cluster_name>}\n        vcenter:\n            {'type': 'vcenter'}\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n\n    entity\n        Entity dict in the format above\n    "
    log.trace('Retrieving entity: %s', entity)
    if entity['type'] == 'cluster':
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, entity['datacenter'])
        return salt.utils.vmware.get_cluster(dc_ref, entity['cluster'])
    elif entity['type'] == 'vcenter':
        return None
    raise ArgumentValueError("Unsupported entity type '{}'".format(entity['type']))

def _validate_entity(entity):
    if False:
        while True:
            i = 10
    '\n    Validates the entity dict representation\n\n    entity\n        Dictionary representation of an entity.\n        See ``_get_entity`` docstrings for format.\n    '
    if entity['type'] == 'cluster':
        schema = ESXClusterEntitySchema.serialize()
    elif entity['type'] == 'vcenter':
        schema = VCenterEntitySchema.serialize()
    else:
        raise ArgumentValueError("Unsupported entity type '{}'".format(entity['type']))
    try:
        jsonschema.validate(entity, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise InvalidEntityError(exc)

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_assigned_licenses(entity, entity_display_name, license_keys=None, service_instance=None):
    if False:
        print('Hello World!')
    "\n    Lists the licenses assigned to an entity\n\n    entity\n        Dictionary representation of an entity.\n        See ``_get_entity`` docstrings for format.\n\n    entity_display_name\n        Entity name used in logging\n\n    license_keys:\n        List of license keys to be retrieved. Default is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_assigned_licenses\n            entity={type:cluster,datacenter:dc,cluster:cl}\n            entiy_display_name=cl\n    "
    log.trace('Listing assigned licenses of entity %s', entity)
    _validate_entity(entity)
    assigned_licenses = salt.utils.vmware.get_assigned_licenses(service_instance, entity_ref=_get_entity(service_instance, entity), entity_name=entity_display_name)
    return [{'key': l.licenseKey, 'name': l.name, 'description': l.labels[0].value if l.labels else None, 'capacity': l.total if l.total > 0 else sys.maxsize} for l in assigned_licenses if license_keys is None or l.licenseKey in license_keys]

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def assign_license(license_key, license_name, entity, entity_display_name, safety_checks=True, service_instance=None):
    if False:
        print('Hello World!')
    "\n    Assigns a license to an entity\n\n    license_key\n        Key of the license to assign\n        See ``_get_entity`` docstrings for format.\n\n    license_name\n        Display name of license\n\n    entity\n        Dictionary representation of an entity\n\n    entity_display_name\n        Entity name used in logging\n\n    safety_checks\n        Specify whether to perform safety check or to skip the checks and try\n        performing the required task. Default is False.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.assign_license license_key=AAAAA-11111-AAAAA-11111-AAAAA\n            license_name=test entity={type:cluster,datacenter:dc,cluster:cl}\n    "
    log.trace('Assigning license %s to entity %s', license_key, entity)
    _validate_entity(entity)
    if safety_checks:
        licenses = salt.utils.vmware.get_licenses(service_instance)
        if not [l for l in licenses if l.licenseKey == license_key]:
            raise VMwareObjectRetrievalError(f"License '{license_name}' wasn't found")
    salt.utils.vmware.assign_license(service_instance, license_key, license_name, entity_ref=_get_entity(service_instance, entity), entity_name=entity_display_name)

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter', 'vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_hosts_via_proxy(hostnames=None, datacenter=None, cluster=None, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of hosts for the specified VMware environment. The list\n    of hosts can be filtered by datacenter name and/or cluster name\n\n    hostnames\n        Hostnames to filter on.\n\n    datacenter_name\n        Name of datacenter. Only hosts in this datacenter will be retrieved.\n        Default is None.\n\n    cluster_name\n        Name of cluster. Only hosts in this cluster will be retrieved. If a\n        datacenter is not specified the first cluster with this name will be\n        considerred. Default is None.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_hosts_via_proxy\n\n        salt '*' vsphere.list_hosts_via_proxy hostnames=[esxi1.example.com]\n\n        salt '*' vsphere.list_hosts_via_proxy datacenter=dc1 cluster=cluster1\n    "
    if cluster:
        if not datacenter:
            raise salt.exceptions.ArgumentValueError('Datacenter is required when cluster is specified')
    get_all_hosts = False
    if not hostnames:
        get_all_hosts = True
    hosts = salt.utils.vmware.get_hosts(service_instance, datacenter_name=datacenter, host_names=hostnames, cluster_name=cluster, get_all_hosts=get_all_hosts)
    return [salt.utils.vmware.get_managed_object_name(h) for h in hosts]

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_disks(disk_ids=None, scsi_addresses=None, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Returns a list of dict representations of the disks in an ESXi host.\n    The list of disks can be filtered by disk canonical names or\n    scsi addresses.\n\n    disk_ids:\n        List of disk canonical names to be retrieved. Default is None.\n\n    scsi_addresses\n        List of scsi addresses of disks to be retrieved. Default is None\n\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_disks\n\n        salt '*' vsphere.list_disks disk_ids='[naa.00, naa.001]'\n\n        salt '*' vsphere.list_disks\n            scsi_addresses='[vmhba0:C0:T0:L0, vmhba1:C0:T0:L0]'\n    "
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    log.trace("Retrieving disks of host '%s'; disc ids = %s; scsi_address = %s", hostname, disk_ids, scsi_addresses)
    get_all_disks = True if not (disk_ids or scsi_addresses) else False
    ret_list = []
    scsi_address_to_lun = salt.utils.vmware.get_scsi_address_to_lun_map(host_ref, hostname=hostname)
    canonical_name_to_scsi_address = {lun.canonicalName: scsi_addr for (scsi_addr, lun) in scsi_address_to_lun.items()}
    for d in salt.utils.vmware.get_disks(host_ref, disk_ids, scsi_addresses, get_all_disks):
        ret_list.append({'id': d.canonicalName, 'scsi_address': canonical_name_to_scsi_address[d.canonicalName]})
    return ret_list

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def erase_disk_partitions(disk_id=None, scsi_address=None, service_instance=None):
    if False:
        i = 10
        return i + 15
    "\n    Erases the partitions on a disk.\n    The disk can be specified either by the canonical name, or by the\n    scsi_address.\n\n    disk_id\n        Canonical name of the disk.\n        Either ``disk_id`` or ``scsi_address`` needs to be specified\n        (``disk_id`` supersedes ``scsi_address``.\n\n    scsi_address\n        Scsi address of the disk.\n        ``disk_id`` or ``scsi_address`` needs to be specified\n        (``disk_id`` supersedes ``scsi_address``.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.erase_disk_partitions scsi_address='vmhaba0:C0:T0:L0'\n\n        salt '*' vsphere.erase_disk_partitions disk_id='naa.000000000000001'\n    "
    if not disk_id and (not scsi_address):
        raise ArgumentValueError("Either 'disk_id' or 'scsi_address' needs to be specified")
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    if not disk_id:
        scsi_address_to_lun = salt.utils.vmware.get_scsi_address_to_lun_map(host_ref)
        if scsi_address not in scsi_address_to_lun:
            raise VMwareObjectRetrievalError("Scsi lun with address '{}' was not found on host '{}'".format(scsi_address, hostname))
        disk_id = scsi_address_to_lun[scsi_address].canonicalName
        log.trace("[%s] Got disk id '%s' for scsi address '%s'", hostname, disk_id, scsi_address)
    log.trace("Erasing disk partitions on disk '%s' in host '%s'", disk_id, hostname)
    salt.utils.vmware.erase_disk_partitions(service_instance, host_ref, disk_id, hostname=hostname)
    log.info("Erased disk partitions on disk '%s' on host '%s'", disk_id, hostname)
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_disk_partitions(disk_id=None, scsi_address=None, service_instance=None):
    if False:
        i = 10
        return i + 15
    "\n    Lists the partitions on a disk.\n    The disk can be specified either by the canonical name, or by the\n    scsi_address.\n\n    disk_id\n        Canonical name of the disk.\n        Either ``disk_id`` or ``scsi_address`` needs to be specified\n        (``disk_id`` supersedes ``scsi_address``.\n\n    scsi_address`\n        Scsi address of the disk.\n        ``disk_id`` or ``scsi_address`` needs to be specified\n        (``disk_id`` supersedes ``scsi_address``.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_disk_partitions scsi_address='vmhaba0:C0:T0:L0'\n\n        salt '*' vsphere.list_disk_partitions disk_id='naa.000000000000001'\n    "
    if not disk_id and (not scsi_address):
        raise ArgumentValueError("Either 'disk_id' or 'scsi_address' needs to be specified")
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    if not disk_id:
        scsi_address_to_lun = salt.utils.vmware.get_scsi_address_to_lun_map(host_ref)
        if scsi_address not in scsi_address_to_lun:
            raise VMwareObjectRetrievalError("Scsi lun with address '{}' was not found on host '{}'".format(scsi_address, hostname))
        disk_id = scsi_address_to_lun[scsi_address].canonicalName
        log.trace("[%s] Got disk id '%s' for scsi address '%s'", hostname, disk_id, scsi_address)
    log.trace("Listing disk partitions on disk '%s' in host '%s'", disk_id, hostname)
    partition_info = salt.utils.vmware.get_disk_partition_info(host_ref, disk_id)
    ret_list = []
    for part_spec in partition_info.spec.partition:
        part_layout = [p for p in partition_info.layout.partition if p.partition == part_spec.partition][0]
        part_dict = {'hostname': hostname, 'device': disk_id, 'format': partition_info.spec.partitionFormat, 'partition': part_spec.partition, 'type': part_spec.type, 'sectors': part_spec.endSector - part_spec.startSector + 1, 'size_KB': (part_layout.end.block - part_layout.start.block + 1) * part_layout.start.blockSize / 1024}
        ret_list.append(part_dict)
    return ret_list

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_diskgroups(cache_disk_ids=None, service_instance=None):
    if False:
        return 10
    "\n    Returns a list of disk group dict representation on an ESXi host.\n    The list of disk groups can be filtered by the cache disks\n    canonical names. If no filtering is applied, all disk groups are returned.\n\n    cache_disk_ids:\n        List of cache disk canonical names of the disk groups to be retrieved.\n        Default is None.\n\n    use_proxy_details\n        Specify whether to use the proxy minion's details instead of the\n        arguments\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.list_diskgroups\n\n        salt '*' vsphere.list_diskgroups cache_disk_ids='[naa.000000000000001]'\n    "
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    log.trace("Listing diskgroups in '%s'", hostname)
    get_all_diskgroups = True if not cache_disk_ids else False
    ret_list = []
    for dg in salt.utils.vmware.get_diskgroups(host_ref, cache_disk_ids, get_all_diskgroups):
        ret_list.append({'cache_disk': dg.ssd.canonicalName, 'capacity_disks': [d.canonicalName for d in dg.nonSsd]})
    return ret_list

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_diskgroup(cache_disk_id, capacity_disk_ids, safety_checks=True, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates disk group on an ESXi host with the specified cache and\n    capacity disks.\n\n    cache_disk_id\n        The canonical name of the disk to be used as a cache. The disk must be\n        ssd.\n\n    capacity_disk_ids\n        A list containing canonical names of the capacity disks. Must contain at\n        least one id. Default is True.\n\n    safety_checks\n        Specify whether to perform safety check or to skip the checks and try\n        performing the required task. Default value is True.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.create_diskgroup cache_disk_id='naa.000000000000001'\n            capacity_disk_ids='[naa.000000000000002, naa.000000000000003]'\n    "
    log.trace('Validating diskgroup input')
    schema = DiskGroupsDiskIdSchema.serialize()
    try:
        jsonschema.validate({'diskgroups': [{'cache_id': cache_disk_id, 'capacity_ids': capacity_disk_ids}]}, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ArgumentValueError(exc)
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    if safety_checks:
        diskgroups = salt.utils.vmware.get_diskgroups(host_ref, [cache_disk_id])
        if diskgroups:
            raise VMwareObjectExistsError("Diskgroup with cache disk id '{}' already exists ESXi host '{}'".format(cache_disk_id, hostname))
    disk_ids = capacity_disk_ids[:]
    disk_ids.insert(0, cache_disk_id)
    disks = salt.utils.vmware.get_disks(host_ref, disk_ids=disk_ids)
    for id in disk_ids:
        if not [d for d in disks if d.canonicalName == id]:
            raise VMwareObjectRetrievalError(f"No disk with id '{id}' was found in ESXi host '{hostname}'")
    cache_disk = [d for d in disks if d.canonicalName == cache_disk_id][0]
    capacity_disks = [d for d in disks if d.canonicalName in capacity_disk_ids]
    vsan_disk_mgmt_system = salt.utils.vsan.get_vsan_disk_management_system(service_instance)
    dg = salt.utils.vsan.create_diskgroup(service_instance, vsan_disk_mgmt_system, host_ref, cache_disk, capacity_disks)
    return True

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def add_capacity_to_diskgroup(cache_disk_id, capacity_disk_ids, safety_checks=True, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Adds capacity disks to the disk group with the specified cache disk.\n\n    cache_disk_id\n        The canonical name of the cache disk.\n\n    capacity_disk_ids\n        A list containing canonical names of the capacity disks to add.\n\n    safety_checks\n        Specify whether to perform safety check or to skip the checks and try\n        performing the required task. Default value is True.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.add_capacity_to_diskgroup\n            cache_disk_id='naa.000000000000001'\n            capacity_disk_ids='[naa.000000000000002, naa.000000000000003]'\n    "
    log.trace('Validating diskgroup input')
    schema = DiskGroupsDiskIdSchema.serialize()
    try:
        jsonschema.validate({'diskgroups': [{'cache_id': cache_disk_id, 'capacity_ids': capacity_disk_ids}]}, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ArgumentValueError(exc)
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    disks = salt.utils.vmware.get_disks(host_ref, disk_ids=capacity_disk_ids)
    if safety_checks:
        for id in capacity_disk_ids:
            if not [d for d in disks if d.canonicalName == id]:
                raise VMwareObjectRetrievalError("No disk with id '{}' was found in ESXi host '{}'".format(id, hostname))
    diskgroups = salt.utils.vmware.get_diskgroups(host_ref, cache_disk_ids=[cache_disk_id])
    if not diskgroups:
        raise VMwareObjectRetrievalError("No diskgroup with cache disk id '{}' was found in ESXi host '{}'".format(cache_disk_id, hostname))
    vsan_disk_mgmt_system = salt.utils.vsan.get_vsan_disk_management_system(service_instance)
    salt.utils.vsan.add_capacity_to_diskgroup(service_instance, vsan_disk_mgmt_system, host_ref, diskgroups[0], disks)
    return True

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def remove_capacity_from_diskgroup(cache_disk_id, capacity_disk_ids, data_evacuation=True, safety_checks=True, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove capacity disks from the disk group with the specified cache disk.\n\n    cache_disk_id\n        The canonical name of the cache disk.\n\n    capacity_disk_ids\n        A list containing canonical names of the capacity disks to add.\n\n    data_evacuation\n        Specifies whether to gracefully evacuate the data on the capacity disks\n        before removing them from the disk group. Default value is True.\n\n    safety_checks\n        Specify whether to perform safety check or to skip the checks and try\n        performing the required task. Default value is True.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.remove_capacity_from_diskgroup\n            cache_disk_id='naa.000000000000001'\n            capacity_disk_ids='[naa.000000000000002, naa.000000000000003]'\n    "
    log.trace('Validating diskgroup input')
    schema = DiskGroupsDiskIdSchema.serialize()
    try:
        jsonschema.validate({'diskgroups': [{'cache_id': cache_disk_id, 'capacity_ids': capacity_disk_ids}]}, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ArgumentValueError(str(exc))
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    disks = salt.utils.vmware.get_disks(host_ref, disk_ids=capacity_disk_ids)
    if safety_checks:
        for id in capacity_disk_ids:
            if not [d for d in disks if d.canonicalName == id]:
                raise VMwareObjectRetrievalError("No disk with id '{}' was found in ESXi host '{}'".format(id, hostname))
    diskgroups = salt.utils.vmware.get_diskgroups(host_ref, cache_disk_ids=[cache_disk_id])
    if not diskgroups:
        raise VMwareObjectRetrievalError("No diskgroup with cache disk id '{}' was found in ESXi host '{}'".format(cache_disk_id, hostname))
    log.trace('data_evacuation = %s', data_evacuation)
    salt.utils.vsan.remove_capacity_from_diskgroup(service_instance, host_ref, diskgroups[0], capacity_disks=[d for d in disks if d.canonicalName in capacity_disk_ids], data_evacuation=data_evacuation)
    return True

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def remove_diskgroup(cache_disk_id, data_accessibility=True, service_instance=None):
    if False:
        i = 10
        return i + 15
    "\n    Remove the diskgroup with the specified cache disk.\n\n    cache_disk_id\n        The canonical name of the cache disk.\n\n    data_accessibility\n        Specifies whether to ensure data accessibility. Default value is True.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.remove_diskgroup cache_disk_id='naa.000000000000001'\n    "
    log.trace('Validating diskgroup input')
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    diskgroups = salt.utils.vmware.get_diskgroups(host_ref, cache_disk_ids=[cache_disk_id])
    if not diskgroups:
        raise VMwareObjectRetrievalError("No diskgroup with cache disk id '{}' was found in ESXi host '{}'".format(cache_disk_id, hostname))
    log.trace('data accessibility = %s', data_accessibility)
    salt.utils.vsan.remove_diskgroup(service_instance, host_ref, diskgroups[0], data_accessibility=data_accessibility)
    return True

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def get_host_cache(service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the host cache configuration on the proxy host.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.get_host_cache\n    "
    ret_dict = {}
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    hci = salt.utils.vmware.get_host_cache(host_ref)
    if not hci:
        log.debug("Host cache not configured on host '%s'", hostname)
        ret_dict['enabled'] = False
        return ret_dict
    return {'enabled': True, 'datastore': {'name': hci.key.name}, 'swap_size': f'{hci.swapSize}MiB'}

@depends(HAS_PYVMOMI)
@depends(HAS_JSONSCHEMA)
@_supports_proxies('esxi')
@_gets_service_instance_via_proxy
@_deprecation_message
def configure_host_cache(enabled, datastore=None, swap_size_MiB=None, service_instance=None):
    if False:
        print('Hello World!')
    "\n    Configures the host cache on the selected host.\n\n    enabled\n        Boolean flag specifying whether the host cache is enabled.\n\n    datastore\n        Name of the datastore that contains the host cache. Must be set if\n        enabled is ``true``.\n\n    swap_size_MiB\n        Swap size in Mibibytes. Needs to be set if enabled is ``true``. Must be\n        smaller than the datastore size.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.configure_host_cache enabled=False\n\n        salt '*' vsphere.configure_host_cache enabled=True datastore=ds1\n            swap_size_MiB=1024\n    "
    log.debug('Validating host cache input')
    schema = SimpleHostCacheSchema.serialize()
    try:
        jsonschema.validate({'enabled': enabled, 'datastore_name': datastore, 'swap_size_MiB': swap_size_MiB}, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ArgumentValueError(exc)
    if not enabled:
        raise ArgumentValueError('Disabling the host cache is not supported')
    ret_dict = {'enabled': False}
    host_ref = _get_proxy_target(service_instance)
    hostname = __proxy__['esxi.get_details']()['esxi_host']
    if datastore:
        ds_refs = salt.utils.vmware.get_datastores(service_instance, host_ref, datastore_names=[datastore])
        if not ds_refs:
            raise VMwareObjectRetrievalError(f"Datastore '{datastore}' was not found on host '{hostname}'")
        ds_ref = ds_refs[0]
    salt.utils.vmware.configure_host_cache(host_ref, ds_ref, swap_size_MiB)
    return True

def _check_hosts(service_instance, host, host_names):
    if False:
        i = 10
        return i + 15
    "\n    Helper function that checks to see if the host provided is a vCenter Server or\n    an ESXi host. If it's an ESXi host, returns a list of a single host_name.\n\n    If a host reference isn't found, we're trying to find a host object for a vCenter\n    server. Raises a CommandExecutionError in this case, as we need host references to\n    check against.\n    "
    if not host_names:
        host_name = _get_host_ref(service_instance, host)
        if host_name:
            host_names = [host]
        else:
            raise CommandExecutionError("No host reference found. If connecting to a vCenter Server, a list of 'host_names' must be provided.")
    elif not isinstance(host_names, list):
        raise CommandExecutionError("'host_names' must be a list.")
    return host_names

def _format_coredump_stdout(cmd_ret):
    if False:
        return 10
    '\n    Helper function to format the stdout from the get_coredump_network_config function.\n\n    cmd_ret\n        The return dictionary that comes from a cmd.run_all call.\n    '
    ret_dict = {}
    for line in cmd_ret['stdout'].splitlines():
        line = line.strip().lower()
        if line.startswith('enabled:'):
            enabled = line.split(':')
            if 'true' in enabled[1]:
                ret_dict['enabled'] = True
            else:
                ret_dict['enabled'] = False
                break
        if line.startswith('host vnic:'):
            host_vnic = line.split(':')
            ret_dict['host_vnic'] = host_vnic[1].strip()
        if line.startswith('network server ip:'):
            ip = line.split(':')
            ret_dict['ip'] = ip[1].strip()
        if line.startswith('network server port:'):
            ip_port = line.split(':')
            ret_dict['port'] = ip_port[1].strip()
    return ret_dict

def _format_firewall_stdout(cmd_ret):
    if False:
        print('Hello World!')
    '\n    Helper function to format the stdout from the get_firewall_status function.\n\n    cmd_ret\n        The return dictionary that comes from a cmd.run_all call.\n    '
    ret_dict = {'success': True, 'rulesets': {}}
    for line in cmd_ret['stdout'].splitlines():
        if line.startswith('Name'):
            continue
        if line.startswith('---'):
            continue
        ruleset_status = line.split()
        ret_dict['rulesets'][ruleset_status[0]] = bool(ruleset_status[1])
    return ret_dict

def _format_syslog_config(cmd_ret):
    if False:
        while True:
            i = 10
    '\n    Helper function to format the stdout from the get_syslog_config function.\n\n    cmd_ret\n        The return dictionary that comes from a cmd.run_all call.\n    '
    ret_dict = {'success': cmd_ret['retcode'] == 0}
    if cmd_ret['retcode'] != 0:
        ret_dict['message'] = cmd_ret['stdout']
    else:
        for line in cmd_ret['stdout'].splitlines():
            line = line.strip()
            cfgvars = line.split(': ')
            key = cfgvars[0].strip()
            value = cfgvars[1].strip()
            ret_dict[key] = value
    return ret_dict

def _get_date_time_mgr(host_reference):
    if False:
        while True:
            i = 10
    '\n    Helper function that returns a dateTimeManager object\n    '
    return host_reference.configManager.dateTimeSystem

def _get_host_ref(service_instance, host, host_name=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function that returns a host object either from the host location or the host_name.\n    If host_name is provided, that is the host_object that will be returned.\n\n    The function will first search for hosts by DNS Name. If no hosts are found, it will\n    try searching by IP Address.\n    '
    search_index = salt.utils.vmware.get_inventory(service_instance).searchIndex
    if host_name:
        host_ref = search_index.FindByDnsName(dnsName=host_name, vmSearch=False)
    else:
        host_ref = search_index.FindByDnsName(dnsName=host, vmSearch=False)
    if host_ref is None:
        host_ref = search_index.FindByIp(ip=host, vmSearch=False)
    return host_ref

def _get_host_ssds(host_reference):
    if False:
        i = 10
        return i + 15
    '\n    Helper function that returns a list of ssd objects for a given host.\n    '
    return _get_host_disks(host_reference).get('SSDs')

def _get_host_non_ssds(host_reference):
    if False:
        i = 10
        return i + 15
    '\n    Helper function that returns a list of Non-SSD objects for a given host.\n    '
    return _get_host_disks(host_reference).get('Non-SSDs')

def _get_host_disks(host_reference):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function that returns a dictionary containing a list of SSD and Non-SSD disks.\n    '
    storage_system = host_reference.configManager.storageSystem
    disks = storage_system.storageDeviceInfo.scsiLun
    ssds = []
    non_ssds = []
    for disk in disks:
        try:
            has_ssd_attr = disk.ssd
        except AttributeError:
            has_ssd_attr = False
        if has_ssd_attr:
            ssds.append(disk)
        else:
            non_ssds.append(disk)
    return {'SSDs': ssds, 'Non-SSDs': non_ssds}

def _get_service_manager(host_reference):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function that returns a service manager object from a given host object.\n    '
    return host_reference.configManager.serviceSystem

def _get_vsan_eligible_disks(service_instance, host, host_names):
    if False:
        i = 10
        return i + 15
    "\n    Helper function that returns a dictionary of host_name keys with either a list of eligible\n    disks that can be added to VSAN or either an 'Error' message or a message saying no\n    eligible disks were found. Possible keys/values look like:\n\n    return = {'host_1': {'Error': 'VSAN System Config Manager is unset ...'},\n              'host_2': {'Eligible': 'The host xxx does not have any VSAN eligible disks.'},\n              'host_3': {'Eligible': [disk1, disk2, disk3, disk4],\n              'host_4': {'Eligible': []}}\n    "
    ret = {}
    for host_name in host_names:
        host_ref = _get_host_ref(service_instance, host, host_name=host_name)
        vsan_system = host_ref.configManager.vsanSystem
        if vsan_system is None:
            msg = "VSAN System Config Manager is unset for host '{}'. VSAN configuration cannot be changed without a configured VSAN System.".format(host_name)
            log.debug(msg)
            ret.update({host_name: {'Error': msg}})
            continue
        suitable_disks = []
        query = vsan_system.QueryDisksForVsan()
        for item in query:
            if item.state == 'eligible':
                suitable_disks.append(item)
        if not suitable_disks:
            msg = "The host '{}' does not have any VSAN eligible disks.".format(host_name)
            log.warning(msg)
            ret.update({host_name: {'Eligible': msg}})
            continue
        disks = _get_host_ssds(host_ref) + _get_host_non_ssds(host_ref)
        matching = []
        for disk in disks:
            for suitable_disk in suitable_disks:
                if disk.canonicalName == suitable_disk.disk.canonicalName:
                    matching.append(disk)
        ret.update({host_name: {'Eligible': matching}})
    return ret

def _reset_syslog_config_params(host, username, password, cmd, resets, valid_resets, protocol=None, port=None, esxi_host=None, credstore=None):
    if False:
        return 10
    '\n    Helper function for reset_syslog_config that resets the config and populates the return dictionary.\n    '
    ret_dict = {}
    all_success = True
    if not isinstance(resets, list):
        resets = [resets]
    for reset_param in resets:
        if reset_param in valid_resets:
            ret = salt.utils.vmware.esxcli(host, username, password, cmd + reset_param, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
            ret_dict[reset_param] = {}
            ret_dict[reset_param]['success'] = ret['retcode'] == 0
            if ret['retcode'] != 0:
                all_success = False
                ret_dict[reset_param]['message'] = ret['stdout']
        else:
            all_success = False
            ret_dict[reset_param] = {}
            ret_dict[reset_param]['success'] = False
            ret_dict[reset_param]['message'] = 'Invalid syslog configuration parameter'
    ret_dict['success'] = all_success
    return ret_dict

def _set_syslog_config_helper(host, username, password, syslog_config, config_value, protocol=None, port=None, reset_service=None, esxi_host=None, credstore=None):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for set_syslog_config that sets the config and populates the return dictionary.\n    '
    cmd = f'system syslog config set --{syslog_config} {config_value}'
    ret_dict = {}
    valid_resets = ['logdir', 'loghost', 'default-rotate', 'default-size', 'default-timeout', 'logdir-unique']
    if syslog_config not in valid_resets:
        ret_dict.update({'success': False, 'message': f"'{syslog_config}' is not a valid config variable."})
        return ret_dict
    response = salt.utils.vmware.esxcli(host, username, password, cmd, protocol=protocol, port=port, esxi_host=esxi_host, credstore=credstore)
    if response['retcode'] != 0:
        ret_dict.update({syslog_config: {'success': False, 'message': response['stdout']}})
    else:
        ret_dict.update({syslog_config: {'success': True}})
    if reset_service:
        if esxi_host:
            host_name = esxi_host
            esxi_host = [esxi_host]
        else:
            host_name = host
        response = syslog_service_reload(host, username, password, protocol=protocol, port=port, esxi_hosts=esxi_host, credstore=credstore).get(host_name)
        ret_dict.update({'syslog_restart': {'success': response['retcode'] == 0}})
    return ret_dict

@depends(HAS_PYVMOMI)
@ignores_kwargs('credstore')
@_deprecation_message
def add_host_to_dvs(host, username, password, vmknic_name, vmnic_name, dvs_name, target_portgroup_name, uplink_portgroup_name, protocol=None, port=None, host_names=None, verify_ssl=True):
    if False:
        i = 10
        return i + 15
    '\n    Adds an ESXi host to a vSphere Distributed Virtual Switch and migrates\n    the desired adapters to the DVS from the standard switch.\n\n    host\n        The location of the vCenter server.\n\n    username\n        The username used to login to the vCenter server.\n\n    password\n        The password used to login to the vCenter server.\n\n    vmknic_name\n        The name of the virtual NIC to migrate.\n\n    vmnic_name\n        The name of the physical NIC to migrate.\n\n    dvs_name\n        The name of the Distributed Virtual Switch.\n\n    target_portgroup_name\n        The name of the distributed portgroup in which to migrate the\n        virtual NIC.\n\n    uplink_portgroup_name\n        The name of the uplink portgroup in which to migrate the\n        physical NIC.\n\n    protocol\n        Optionally set to alternate protocol if the vCenter server or ESX/ESXi host is not\n        using the default protocol. Default protocol is ``https``.\n\n    port\n        Optionally set to alternate port if the vCenter server or ESX/ESXi host is not\n        using the default port. Default port is ``443``.\n\n    host_names:\n        An array of VMware host names to migrate\n\n    verify_ssl\n        Verify the SSL certificate. Default: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt some_host vsphere.add_host_to_dvs host=\'vsphere.corp.com\'\n            username=\'administrator@vsphere.corp.com\' password=\'vsphere_password\'\n            vmknic_name=\'vmk0\' vmnic_name=\'vnmic0\' dvs_name=\'DSwitch\'\n            target_portgroup_name=\'DPortGroup\' uplink_portgroup_name=\'DSwitch1-DVUplinks-181\'\n            protocol=\'https\' port=\'443\', host_names="[\'esxi1.corp.com\',\'esxi2.corp.com\',\'esxi3.corp.com\']"\n\n    Return Example:\n\n    .. code-block:: yaml\n\n        somehost:\n            ----------\n            esxi1.corp.com:\n                ----------\n                dvs:\n                    DSwitch\n                portgroup:\n                    DPortGroup\n                status:\n                    True\n                uplink:\n                    DSwitch-DVUplinks-181\n                vmknic:\n                    vmk0\n                vmnic:\n                    vmnic0\n            esxi2.corp.com:\n                ----------\n                dvs:\n                    DSwitch\n                portgroup:\n                    DPortGroup\n                status:\n                    True\n                uplink:\n                    DSwitch-DVUplinks-181\n                vmknic:\n                    vmk0\n                vmnic:\n                    vmnic0\n            esxi3.corp.com:\n                ----------\n                dvs:\n                    DSwitch\n                portgroup:\n                    DPortGroup\n                status:\n                    True\n                uplink:\n                    DSwitch-DVUplinks-181\n                vmknic:\n                    vmk0\n                vmnic:\n                    vmnic0\n            message:\n            success:\n                True\n\n    This was very difficult to figure out.  VMware\'s PyVmomi documentation at\n\n    https://github.com/vmware/pyvmomi/blob/master/docs/vim/DistributedVirtualSwitch.rst\n    (which is a copy of the official documentation here:\n    https://www.vmware.com/support/developer/converter-sdk/conv60_apireference/vim.DistributedVirtualSwitch.html)\n\n    says to create the DVS, create distributed portgroups, and then add the\n    host to the DVS specifying which physical NIC to use as the port backing.\n    However, if the physical NIC is in use as the only link from the host\n    to vSphere, this will fail with an unhelpful "busy" error.\n\n    There is, however, a Powershell PowerCLI cmdlet called Add-VDSwitchPhysicalNetworkAdapter\n    that does what we want.  I used Onyx (https://labs.vmware.com/flings/onyx)\n    to sniff the SOAP stream from Powershell to our vSphere server and got\n    this snippet out:\n\n    .. code-block:: xml\n\n        <UpdateNetworkConfig xmlns="urn:vim25">\n          <_this type="HostNetworkSystem">networkSystem-187</_this>\n          <config>\n            <vswitch>\n              <changeOperation>edit</changeOperation>\n              <name>vSwitch0</name>\n              <spec>\n                <numPorts>7812</numPorts>\n              </spec>\n            </vswitch>\n            <proxySwitch>\n                <changeOperation>edit</changeOperation>\n                <uuid>73 a4 05 50 b0 d2 7e b9-38 80 5d 24 65 8f da 70</uuid>\n                <spec>\n                <backing xsi:type="DistributedVirtualSwitchHostMemberPnicBacking">\n                    <pnicSpec><pnicDevice>vmnic0</pnicDevice></pnicSpec>\n                </backing>\n                </spec>\n            </proxySwitch>\n            <portgroup>\n              <changeOperation>remove</changeOperation>\n              <spec>\n                <name>Management Network</name><vlanId>-1</vlanId><vswitchName /><policy />\n              </spec>\n            </portgroup>\n            <vnic>\n              <changeOperation>edit</changeOperation>\n              <device>vmk0</device>\n              <portgroup />\n              <spec>\n                <distributedVirtualPort>\n                  <switchUuid>73 a4 05 50 b0 d2 7e b9-38 80 5d 24 65 8f da 70</switchUuid>\n                  <portgroupKey>dvportgroup-191</portgroupKey>\n                </distributedVirtualPort>\n              </spec>\n            </vnic>\n          </config>\n          <changeMode>modify</changeMode>\n        </UpdateNetworkConfig>\n\n    The SOAP API maps closely to PyVmomi, so from there it was (relatively)\n    easy to figure out what Python to write.\n    '
    ret = {}
    ret['success'] = True
    ret['message'] = []
    service_instance = salt.utils.vmware.get_service_instance(host=host, username=username, password=password, protocol=protocol, port=port, verify_ssl=verify_ssl)
    dvs = salt.utils.vmware._get_dvs(service_instance, dvs_name)
    if not dvs:
        ret['message'].append(f'No Distributed Virtual Switch found with name {dvs_name}')
        ret['success'] = False
    target_portgroup = salt.utils.vmware._get_dvs_portgroup(dvs, target_portgroup_name)
    if not target_portgroup:
        ret['message'].append(f'No target portgroup found with name {target_portgroup_name}')
        ret['success'] = False
    uplink_portgroup = salt.utils.vmware._get_dvs_uplink_portgroup(dvs, uplink_portgroup_name)
    if not uplink_portgroup:
        ret['message'].append(f'No uplink portgroup found with name {uplink_portgroup_name}')
        ret['success'] = False
    if ret['message']:
        return ret
    dvs_uuid = dvs.config.uuid
    try:
        host_names = _check_hosts(service_instance, host, host_names)
    except CommandExecutionError as e:
        ret['message'] = f'Error retrieving hosts: {e.msg}'
        return ret
    for host_name in host_names:
        ret[host_name] = {}
        ret[host_name].update({'status': False, 'uplink': uplink_portgroup_name, 'portgroup': target_portgroup_name, 'vmknic': vmknic_name, 'vmnic': vmnic_name, 'dvs': dvs_name})
        host_ref = _get_host_ref(service_instance, host, host_name)
        if not host_ref:
            ret[host_name].update({'message': 'Host {1} not found'.format(host_name)})
            ret['success'] = False
            continue
        dvs_hostmember_config = vim.dvs.HostMember.ConfigInfo(host=host_ref)
        dvs_hostmember = vim.dvs.HostMember(config=dvs_hostmember_config)
        p_nics = salt.utils.vmware._get_pnics(host_ref)
        p_nic = [x for x in p_nics if x.device == vmnic_name]
        if not p_nic:
            ret[host_name].update({'message': f'Physical nic {vmknic_name} not found'})
            ret['success'] = False
            continue
        v_nics = salt.utils.vmware._get_vnics(host_ref)
        v_nic = [x for x in v_nics if x.device == vmknic_name]
        if not v_nic:
            ret[host_name].update({'message': f'Virtual nic {vmnic_name} not found'})
            ret['success'] = False
            continue
        v_nic_mgr = salt.utils.vmware._get_vnic_manager(host_ref)
        if not v_nic_mgr:
            ret[host_name].update({'message': "Unable to get the host's virtual nic manager."})
            ret['success'] = False
            continue
        dvs_pnic_spec = vim.dvs.HostMember.PnicSpec(pnicDevice=vmnic_name, uplinkPortgroupKey=uplink_portgroup.key)
        pnic_backing = vim.dvs.HostMember.PnicBacking(pnicSpec=[dvs_pnic_spec])
        dvs_hostmember_config_spec = vim.dvs.HostMember.ConfigSpec(host=host_ref, operation='add')
        dvs_config = vim.DVSConfigSpec(configVersion=dvs.config.configVersion, host=[dvs_hostmember_config_spec])
        task = dvs.ReconfigureDvs_Task(spec=dvs_config)
        try:
            salt.utils.vmware.wait_for_task(task, host_name, 'Adding host to the DVS', sleep_seconds=3)
        except Exception as e:
            if hasattr(e, 'message') and hasattr(e.message, 'msg'):
                if not (host_name in e.message.msg and 'already exists' in e.message.msg):
                    ret['success'] = False
                    ret[host_name].update({'message': e.message.msg})
                    continue
            else:
                raise
        network_system = host_ref.configManager.networkSystem
        source_portgroup = None
        for pg in host_ref.config.network.portgroup:
            if pg.spec.name == v_nic[0].portgroup:
                source_portgroup = pg
                break
        if not source_portgroup:
            ret[host_name].update({'message': 'No matching portgroup on the vSwitch'})
            ret['success'] = False
            continue
        virtual_nic_config = vim.HostVirtualNicConfig(changeOperation='edit', device=v_nic[0].device, portgroup=source_portgroup.spec.name, spec=vim.HostVirtualNicSpec(distributedVirtualPort=vim.DistributedVirtualSwitchPortConnection(portgroupKey=target_portgroup.key, switchUuid=target_portgroup.config.distributedVirtualSwitch.uuid)))
        current_vswitch_ports = host_ref.config.network.vswitch[0].numPorts
        vswitch_config = vim.HostVirtualSwitchConfig(changeOperation='edit', name='vSwitch0', spec=vim.HostVirtualSwitchSpec(numPorts=current_vswitch_ports))
        proxyswitch_config = vim.HostProxySwitchConfig(changeOperation='edit', uuid=dvs_uuid, spec=vim.HostProxySwitchSpec(backing=pnic_backing))
        host_network_config = vim.HostNetworkConfig(vswitch=[vswitch_config], proxySwitch=[proxyswitch_config], portgroup=[vim.HostPortGroupConfig(changeOperation='remove', spec=source_portgroup.spec)], vnic=[virtual_nic_config])
        try:
            network_system.UpdateNetworkConfig(changeMode='modify', config=host_network_config)
            ret[host_name].update({'status': True})
        except Exception as e:
            if hasattr(e, 'msg'):
                ret[host_name].update({'message': f'Failed to migrate adapters ({e.msg})'})
                continue
            else:
                raise
    return ret

@depends(HAS_PYVMOMI)
@_supports_proxies('esxi', 'esxcluster', 'esxdatacenter', 'vcenter')
def _get_proxy_target(service_instance):
    if False:
        i = 10
        return i + 15
    "\n    Returns the target object of a proxy.\n\n    If the object doesn't exist a VMwareObjectRetrievalError is raised\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter/ESXi host.\n    "
    proxy_type = get_proxy_type()
    if not salt.utils.vmware.is_connection_to_a_vcenter(service_instance):
        raise CommandExecutionError("'_get_proxy_target' not supported when connected via the ESXi host")
    reference = None
    if proxy_type == 'esxcluster':
        (host, username, password, protocol, port, mechanism, principal, domain, datacenter, cluster) = _get_esxcluster_proxy_details()
        dc_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
        reference = salt.utils.vmware.get_cluster(dc_ref, cluster)
    elif proxy_type == 'esxdatacenter':
        (host, username, password, protocol, port, mechanism, principal, domain, datacenter) = _get_esxdatacenter_proxy_details()
        reference = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    elif proxy_type == 'vcenter':
        reference = salt.utils.vmware.get_root_folder(service_instance)
    elif proxy_type == 'esxi':
        details = __proxy__['esxi.get_details']()
        if 'vcenter' not in details:
            raise InvalidEntityError('Proxies connected directly to ESXi hosts are not supported')
        references = salt.utils.vmware.get_hosts(service_instance, host_names=details['esxi_host'])
        if not references:
            raise VMwareObjectRetrievalError("ESXi host '{}' was not found".format(details['esxi_host']))
        reference = references[0]
    log.trace('reference = %s', reference)
    return reference

def _get_esxdatacenter_proxy_details():
    if False:
        return 10
    "\n    Returns the running esxdatacenter's proxy details\n    "
    det = __salt__['esxdatacenter.get_details']()
    return (det.get('vcenter'), det.get('username'), det.get('password'), det.get('protocol'), det.get('port'), det.get('mechanism'), det.get('principal'), det.get('domain'), det.get('datacenter'))

def _get_esxcluster_proxy_details():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the running esxcluster's proxy details\n    "
    det = __salt__['esxcluster.get_details']()
    return (det.get('vcenter'), det.get('username'), det.get('password'), det.get('protocol'), det.get('port'), det.get('mechanism'), det.get('principal'), det.get('domain'), det.get('datacenter'), det.get('cluster'))

def _get_esxi_proxy_details():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the running esxi's proxy details\n    "
    det = __proxy__['esxi.get_details']()
    host = det.get('host')
    if det.get('vcenter'):
        host = det['vcenter']
    esxi_hosts = None
    if det.get('esxi_host'):
        esxi_hosts = [det['esxi_host']]
    return (host, det.get('username'), det.get('password'), det.get('protocol'), det.get('port'), det.get('mechanism'), det.get('principal'), det.get('domain'), esxi_hosts)

@depends(HAS_PYVMOMI)
@_gets_service_instance_via_proxy
@_deprecation_message
def get_vm(name, datacenter=None, vm_properties=None, traversal_spec=None, parent_ref=None, service_instance=None):
    if False:
        print('Hello World!')
    '\n    Returns vm object properties.\n\n    name\n        Name of the virtual machine.\n\n    datacenter\n        Datacenter name\n\n    vm_properties\n        List of vm properties.\n\n    traversal_spec\n        Traversal Spec object(s) for searching.\n\n    parent_ref\n        Container Reference object for searching under a given object.\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n    '
    virtual_machine = salt.utils.vmware.get_vm_by_property(service_instance, name, datacenter=datacenter, vm_properties=vm_properties, traversal_spec=traversal_spec, parent_ref=parent_ref)
    return virtual_machine

@depends(HAS_PYVMOMI)
@_gets_service_instance_via_proxy
@_deprecation_message
def get_vm_config_file(name, datacenter, placement, datastore, service_instance=None):
    if False:
        return 10
    '\n    Queries the virtual machine config file and returns\n    vim.host.DatastoreBrowser.SearchResults object on success None on failure\n\n    name\n        Name of the virtual machine\n\n    datacenter\n        Datacenter name\n\n    datastore\n        Datastore where the virtual machine files are stored\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n    '
    browser_spec = vim.host.DatastoreBrowser.SearchSpec()
    directory = name
    browser_spec.query = [vim.host.DatastoreBrowser.VmConfigQuery()]
    datacenter_object = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    if 'cluster' in placement:
        container_object = salt.utils.vmware.get_cluster(datacenter_object, placement['cluster'])
    else:
        container_objects = salt.utils.vmware.get_hosts(service_instance, datacenter_name=datacenter, host_names=[placement['host']])
        if not container_objects:
            raise salt.exceptions.VMwareObjectRetrievalError("ESXi host named '{}' wasn't found.".format(placement['host']))
        container_object = container_objects[0]
    files = salt.utils.vmware.get_datastore_files(service_instance, directory, [datastore], container_object, browser_spec)
    if files and len(files[0].file) > 1:
        raise salt.exceptions.VMwareMultipleObjectsError('Multiple configuration files found in the same virtual machine folder')
    elif files and files[0].file:
        return files[0]
    else:
        return None

def _apply_hardware_version(hardware_version, config_spec, operation='add'):
    if False:
        print('Hello World!')
    "\n    Specifies vm container version or schedules upgrade,\n    returns True on change and False if nothing have been changed.\n\n    hardware_version\n        Hardware version string eg. vmx-08\n\n    config_spec\n        Configuration spec object\n\n    operation\n        Defines the operation which should be used,\n        the possibles values: 'add' and 'edit', the default value is 'add'\n    "
    log.trace('Configuring virtual machine hardware version version=%s', hardware_version)
    if operation == 'edit':
        log.trace('Scheduling hardware version upgrade to %s', hardware_version)
        scheduled_hardware_upgrade = vim.vm.ScheduledHardwareUpgradeInfo()
        scheduled_hardware_upgrade.upgradePolicy = 'always'
        scheduled_hardware_upgrade.versionKey = hardware_version
        config_spec.scheduledHardwareUpgradeInfo = scheduled_hardware_upgrade
    elif operation == 'add':
        config_spec.version = str(hardware_version)

def _apply_cpu_config(config_spec, cpu_props):
    if False:
        i = 10
        return i + 15
    '\n    Sets CPU core count to the given value\n\n    config_spec\n        vm.ConfigSpec object\n\n    cpu_props\n        CPU properties dict\n    '
    log.trace('Configuring virtual machine CPU settings cpu_props=%s', cpu_props)
    if 'count' in cpu_props:
        config_spec.numCPUs = int(cpu_props['count'])
    if 'cores_per_socket' in cpu_props:
        config_spec.numCoresPerSocket = int(cpu_props['cores_per_socket'])
    if 'nested' in cpu_props and cpu_props['nested']:
        config_spec.nestedHVEnabled = cpu_props['nested']
    if 'hotadd' in cpu_props and cpu_props['hotadd']:
        config_spec.cpuHotAddEnabled = cpu_props['hotadd']
    if 'hotremove' in cpu_props and cpu_props['hotremove']:
        config_spec.cpuHotRemoveEnabled = cpu_props['hotremove']

def _apply_memory_config(config_spec, memory):
    if False:
        print('Hello World!')
    '\n    Sets memory size to the given value\n\n    config_spec\n        vm.ConfigSpec object\n\n    memory\n        Memory size and unit\n    '
    log.trace('Configuring virtual machine memory settings memory=%s', memory)
    if 'size' in memory and 'unit' in memory:
        try:
            if memory['unit'].lower() == 'kb':
                memory_mb = memory['size'] / 1024
            elif memory['unit'].lower() == 'mb':
                memory_mb = memory['size']
            elif memory['unit'].lower() == 'gb':
                memory_mb = int(float(memory['size']) * 1024)
        except (TypeError, ValueError):
            memory_mb = int(memory['size'])
        config_spec.memoryMB = memory_mb
    if 'reservation_max' in memory:
        config_spec.memoryReservationLockedToMax = memory['reservation_max']
    if 'hotadd' in memory:
        config_spec.memoryHotAddEnabled = memory['hotadd']

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def get_advanced_configs(vm_name, datacenter, service_instance=None):
    if False:
        return 10
    '\n    Returns extra config parameters from a virtual machine advanced config list\n\n    vm_name\n        Virtual machine name\n\n    datacenter\n        Datacenter name where the virtual machine is available\n\n    service_instance\n        vCenter service instance for connection and configuration\n    '
    current_config = get_vm_config(vm_name, datacenter=datacenter, objects=True, service_instance=service_instance)
    return current_config['advanced_configs']

def _apply_advanced_config(config_spec, advanced_config, vm_extra_config=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets configuration parameters for the vm\n\n    config_spec\n        vm.ConfigSpec object\n\n    advanced_config\n        config key value pairs\n\n    vm_extra_config\n        Virtual machine vm_ref.config.extraConfig object\n    '
    log.trace('Configuring advanced configuration parameters %s', advanced_config)
    if isinstance(advanced_config, str):
        raise salt.exceptions.ArgumentValueError("The specified 'advanced_configs' configuration option cannot be parsed, please check the parameters")
    for (key, value) in advanced_config.items():
        if vm_extra_config:
            for option in vm_extra_config:
                if option.key == key and option.value == str(value):
                    continue
        else:
            option = vim.option.OptionValue(key=key, value=value)
            config_spec.extraConfig.append(option)

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def set_advanced_configs(vm_name, datacenter, advanced_configs, service_instance=None):
    if False:
        return 10
    '\n    Appends extra config parameters to a virtual machine advanced config list\n\n    vm_name\n        Virtual machine name\n\n    datacenter\n        Datacenter name where the virtual machine is available\n\n    advanced_configs\n        Dictionary with advanced parameter key value pairs\n\n    service_instance\n        vCenter service instance for connection and configuration\n    '
    current_config = get_vm_config(vm_name, datacenter=datacenter, objects=True, service_instance=service_instance)
    diffs = compare_vm_configs({'name': vm_name, 'advanced_configs': advanced_configs}, current_config)
    datacenter_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    vm_ref = salt.utils.vmware.get_mor_by_property(service_instance, vim.VirtualMachine, vm_name, property_name='name', container_ref=datacenter_ref)
    config_spec = vim.vm.ConfigSpec()
    changes = diffs['advanced_configs'].diffs
    _apply_advanced_config(config_spec, diffs['advanced_configs'].new_values, vm_ref.config.extraConfig)
    if changes:
        salt.utils.vmware.update_vm(vm_ref, config_spec)
    return {'advanced_config_changes': changes}

def _delete_advanced_config(config_spec, advanced_config, vm_extra_config):
    if False:
        print('Hello World!')
    '\n    Removes configuration parameters for the vm\n\n    config_spec\n        vm.ConfigSpec object\n\n    advanced_config\n        List of advanced config keys to be deleted\n\n    vm_extra_config\n        Virtual machine vm_ref.config.extraConfig object\n    '
    log.trace('Removing advanced configuration parameters %s', advanced_config)
    if isinstance(advanced_config, str):
        raise salt.exceptions.ArgumentValueError("The specified 'advanced_configs' configuration option cannot be parsed, please check the parameters")
    removed_configs = []
    for key in advanced_config:
        for option in vm_extra_config:
            if option.key == key:
                option = vim.option.OptionValue(key=key, value='')
                config_spec.extraConfig.append(option)
                removed_configs.append(key)
    return removed_configs

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def delete_advanced_configs(vm_name, datacenter, advanced_configs, service_instance=None):
    if False:
        return 10
    '\n    Removes extra config parameters from a virtual machine\n\n    vm_name\n        Virtual machine name\n\n    datacenter\n        Datacenter name where the virtual machine is available\n\n    advanced_configs\n        List of advanced config values to be removed\n\n    service_instance\n        vCenter service instance for connection and configuration\n    '
    datacenter_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    vm_ref = salt.utils.vmware.get_mor_by_property(service_instance, vim.VirtualMachine, vm_name, property_name='name', container_ref=datacenter_ref)
    config_spec = vim.vm.ConfigSpec()
    removed_configs = _delete_advanced_config(config_spec, advanced_configs, vm_ref.config.extraConfig)
    if removed_configs:
        salt.utils.vmware.update_vm(vm_ref, config_spec)
    return {'removed_configs': removed_configs}

def _get_scsi_controller_key(bus_number, scsi_ctrls):
    if False:
        i = 10
        return i + 15
    '\n    Returns key number of the SCSI controller keys\n\n    bus_number\n        Controller bus number from the adapter\n\n    scsi_ctrls\n        List of SCSI Controller objects (old+newly created)\n    '
    keys = [ctrl.key for ctrl in scsi_ctrls if scsi_ctrls and ctrl.busNumber == bus_number]
    if not keys:
        raise salt.exceptions.VMwareVmCreationError(f"SCSI controller number {bus_number} doesn't exist")
    return keys[0]

def _apply_hard_disk(unit_number, key, operation, disk_label=None, size=None, unit='GB', controller_key=None, thin_provision=None, eagerly_scrub=None, datastore=None, filename=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a vim.vm.device.VirtualDeviceSpec object specifying to add/edit\n    a virtual disk device\n\n    unit_number\n        Add network adapter to this address\n\n    key\n        Device key number\n\n    operation\n        Action which should be done on the device add or edit\n\n    disk_label\n        Label of the new disk, can be overridden\n\n    size\n        Size of the disk\n\n    unit\n        Unit of the size, can be GB, MB, KB\n\n    controller_key\n        Unique umber of the controller key\n\n    thin_provision\n        Boolean for thin provision\n\n    eagerly_scrub\n        Boolean for eagerly scrubbing\n\n    datastore\n        Datastore name where the disk will be located\n\n    filename\n        Full file name of the vm disk\n    '
    log.trace('Configuring hard disk %s size=%s, unit=%s, controller_key=%s, thin_provision=%s, eagerly_scrub=%s, datastore=%s, filename=%s', disk_label, size, unit, controller_key, thin_provision, eagerly_scrub, datastore, filename)
    disk_spec = vim.vm.device.VirtualDeviceSpec()
    disk_spec.device = vim.vm.device.VirtualDisk()
    disk_spec.device.key = key
    disk_spec.device.unitNumber = unit_number
    disk_spec.device.deviceInfo = vim.Description()
    if size:
        convert_size = salt.utils.vmware.convert_to_kb(unit, size)
        disk_spec.device.capacityInKB = convert_size['size']
    if disk_label:
        disk_spec.device.deviceInfo.label = disk_label
    if thin_provision is not None or eagerly_scrub is not None:
        disk_spec.device.backing = vim.vm.device.VirtualDisk.FlatVer2BackingInfo()
        disk_spec.device.backing.diskMode = 'persistent'
    if thin_provision is not None:
        disk_spec.device.backing.thinProvisioned = thin_provision
    if eagerly_scrub is not None and eagerly_scrub != 'None':
        disk_spec.device.backing.eagerlyScrub = eagerly_scrub
    if controller_key:
        disk_spec.device.controllerKey = controller_key
    if operation == 'add':
        disk_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
        disk_spec.device.backing.fileName = '[{}] {}'.format(salt.utils.vmware.get_managed_object_name(datastore), filename)
        disk_spec.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.create
    elif operation == 'edit':
        disk_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    return disk_spec

def _create_adapter_type(network_adapter, adapter_type, network_adapter_label=''):
    if False:
        print('Hello World!')
    '\n    Returns a vim.vm.device.VirtualEthernetCard object specifying a virtual\n    ethernet card information\n\n    network_adapter\n        None or VirtualEthernet object\n\n    adapter_type\n        String, type of adapter\n\n    network_adapter_label\n        string, network adapter name\n    '
    log.trace('Configuring virtual machine network adapter adapter_type=%s', adapter_type)
    if adapter_type in ['vmxnet', 'vmxnet2', 'vmxnet3', 'e1000', 'e1000e']:
        edited_network_adapter = salt.utils.vmware.get_network_adapter_type(adapter_type)
        if isinstance(network_adapter, type(edited_network_adapter)):
            edited_network_adapter = network_adapter
        elif network_adapter:
            log.trace("Changing type of '%s' from '%s' to '%s'", network_adapter.deviceInfo.label, type(network_adapter).__name__.rsplit('.', 1)[1][7:].lower(), adapter_type)
    elif network_adapter:
        if adapter_type:
            log.error("Cannot change type of '%s' to '%s'. Not changing type", network_adapter.deviceInfo.label, adapter_type)
        edited_network_adapter = network_adapter
    else:
        if not adapter_type:
            log.trace("The type of '%s' has not been specified. Creating of default type 'vmxnet3'", network_adapter_label)
        edited_network_adapter = vim.vm.device.VirtualVmxnet3()
    return edited_network_adapter

def _create_network_backing(network_name, switch_type, parent_ref):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a vim.vm.device.VirtualDevice.BackingInfo object specifying a\n    virtual ethernet card backing information\n\n    network_name\n        string, network name\n\n    switch_type\n        string, type of switch\n\n    parent_ref\n        Parent reference to search for network\n    '
    log.trace('Configuring virtual machine network backing network_name=%s switch_type=%s parent=%s', network_name, switch_type, salt.utils.vmware.get_managed_object_name(parent_ref))
    backing = {}
    if network_name:
        if switch_type == 'standard':
            networks = salt.utils.vmware.get_networks(parent_ref, network_names=[network_name])
            if not networks:
                raise salt.exceptions.VMwareObjectRetrievalError(f"The network '{network_name}' could not be retrieved.")
            network_ref = networks[0]
            backing = vim.vm.device.VirtualEthernetCard.NetworkBackingInfo()
            backing.deviceName = network_name
            backing.network = network_ref
        elif switch_type == 'distributed':
            networks = salt.utils.vmware.get_dvportgroups(parent_ref, portgroup_names=[network_name])
            if not networks:
                raise salt.exceptions.VMwareObjectRetrievalError(f"The port group '{network_name}' could not be retrieved.")
            network_ref = networks[0]
            dvs_port_connection = vim.dvs.PortConnection(portgroupKey=network_ref.key, switchUuid=network_ref.config.distributedVirtualSwitch.uuid)
            backing = vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo()
            backing.port = dvs_port_connection
    return backing

def _apply_network_adapter_config(key, network_name, adapter_type, switch_type, network_adapter_label=None, operation='add', connectable=None, mac=None, parent=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns a vim.vm.device.VirtualDeviceSpec object specifying to add/edit a\n    network device\n\n    network_adapter_label\n        Network adapter label\n\n    key\n        Unique key for device creation\n\n    network_name\n        Network or port group name\n\n    adapter_type\n        Type of the adapter eg. vmxnet3\n\n    switch_type\n        Type of the switch: standard or distributed\n\n    operation\n        Type of operation: add or edit\n\n    connectable\n        Dictionary with the device connection properties\n\n    mac\n        MAC address of the network adapter\n\n    parent\n        Parent object reference\n    '
    adapter_type.strip().lower()
    switch_type.strip().lower()
    log.trace('Configuring virtual machine network adapter network_adapter_label=%s network_name=%s adapter_type=%s switch_type=%s mac=%s', network_adapter_label, network_name, adapter_type, switch_type, mac)
    network_spec = vim.vm.device.VirtualDeviceSpec()
    network_spec.device = _create_adapter_type(network_spec.device, adapter_type, network_adapter_label=network_adapter_label)
    network_spec.device.deviceInfo = vim.Description()
    if operation == 'add':
        network_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    elif operation == 'edit':
        network_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    if switch_type and network_name:
        network_spec.device.backing = _create_network_backing(network_name, switch_type, parent)
        network_spec.device.deviceInfo.summary = network_name
    if key:
        network_spec.device.key = key
    if network_adapter_label:
        network_spec.device.deviceInfo.label = network_adapter_label
    if mac:
        network_spec.device.macAddress = mac
        network_spec.device.addressType = 'Manual'
    network_spec.device.wakeOnLanEnabled = True
    if connectable:
        network_spec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
        network_spec.device.connectable.startConnected = connectable['start_connected']
        network_spec.device.connectable.allowGuestControl = connectable['allow_guest_control']
    return network_spec

def _apply_scsi_controller(adapter, adapter_type, bus_sharing, key, bus_number, operation):
    if False:
        return 10
    "\n    Returns a vim.vm.device.VirtualDeviceSpec object specifying to\n    add/edit a SCSI controller\n\n    adapter\n        SCSI controller adapter name\n\n    adapter_type\n        SCSI controller adapter type eg. paravirtual\n\n    bus_sharing\n         SCSI controller bus sharing eg. virtual_sharing\n\n    key\n        SCSI controller unique key\n\n    bus_number\n        Device bus number property\n\n    operation\n        Describes the operation which should be done on the object,\n        the possibles values: 'add' and 'edit', the default value is 'add'\n\n    .. code-block:: bash\n\n        scsi:\n          adapter: 'SCSI controller 0'\n          type: paravirtual or lsilogic or lsilogic_sas\n          bus_sharing: 'no_sharing' or 'virtual_sharing' or 'physical_sharing'\n    "
    log.trace('Configuring scsi controller adapter=%s adapter_type=%s bus_sharing=%s key=%s bus_number=%s', adapter, adapter_type, bus_sharing, key, bus_number)
    scsi_spec = vim.vm.device.VirtualDeviceSpec()
    if adapter_type == 'lsilogic':
        summary = 'LSI Logic'
        scsi_spec.device = vim.vm.device.VirtualLsiLogicController()
    elif adapter_type == 'lsilogic_sas':
        summary = 'LSI Logic Sas'
        scsi_spec.device = vim.vm.device.VirtualLsiLogicSASController()
    elif adapter_type == 'paravirtual':
        summary = 'VMware paravirtual SCSI'
        scsi_spec.device = vim.vm.device.ParaVirtualSCSIController()
    elif adapter_type == 'buslogic':
        summary = 'Bus Logic'
        scsi_spec.device = vim.vm.device.VirtualBusLogicController()
    if operation == 'add':
        scsi_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    elif operation == 'edit':
        scsi_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    scsi_spec.device.key = key
    scsi_spec.device.busNumber = bus_number
    scsi_spec.device.deviceInfo = vim.Description()
    scsi_spec.device.deviceInfo.label = adapter
    scsi_spec.device.deviceInfo.summary = summary
    if bus_sharing == 'virtual_sharing':
        scsi_spec.device.sharedBus = vim.vm.device.VirtualSCSIController.Sharing.virtualSharing
    elif bus_sharing == 'physical_sharing':
        scsi_spec.device.sharedBus = vim.vm.device.VirtualSCSIController.Sharing.physicalSharing
    elif bus_sharing == 'no_sharing':
        scsi_spec.device.sharedBus = vim.vm.device.VirtualSCSIController.Sharing.noSharing
    return scsi_spec

def _create_ide_controllers(ide_controllers):
    if False:
        while True:
            i = 10
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec objects representing\n    IDE controllers\n\n    ide_controllers\n        IDE properties\n    '
    ide_ctrls = []
    keys = range(-200, -250, -1)
    if ide_controllers:
        devs = [ide['adapter'] for ide in ide_controllers]
        log.trace('Creating IDE controllers %s', devs)
        for (ide, key) in zip(ide_controllers, keys):
            ide_ctrls.append(_apply_ide_controller_config(ide['adapter'], 'add', key, abs(key + 200)))
    return ide_ctrls

def _apply_ide_controller_config(ide_controller_label, operation, key, bus_number=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a vim.vm.device.VirtualDeviceSpec object specifying to add/edit an\n    IDE controller\n\n    ide_controller_label\n        Controller label of the IDE adapter\n\n    operation\n        Type of operation: add or edit\n\n    key\n        Unique key of the device\n\n    bus_number\n        Device bus number property\n    '
    log.trace('Configuring IDE controller ide_controller_label=%s', ide_controller_label)
    ide_spec = vim.vm.device.VirtualDeviceSpec()
    ide_spec.device = vim.vm.device.VirtualIDEController()
    if operation == 'add':
        ide_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    if operation == 'edit':
        ide_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    ide_spec.device.key = key
    ide_spec.device.busNumber = bus_number
    if ide_controller_label:
        ide_spec.device.deviceInfo = vim.Description()
        ide_spec.device.deviceInfo.label = ide_controller_label
        ide_spec.device.deviceInfo.summary = ide_controller_label
    return ide_spec

def _create_sata_controllers(sata_controllers):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec objects representing\n    SATA controllers\n\n    sata_controllers\n        SATA properties\n    '
    sata_ctrls = []
    keys = range(-15000, -15050, -1)
    if sata_controllers:
        devs = [sata['adapter'] for sata in sata_controllers]
        log.trace('Creating SATA controllers %s', devs)
        for (sata, key) in zip(sata_controllers, keys):
            sata_ctrls.append(_apply_sata_controller_config(sata['adapter'], 'add', key, sata['bus_number']))
    return sata_ctrls

def _apply_sata_controller_config(sata_controller_label, operation, key, bus_number=0):
    if False:
        while True:
            i = 10
    '\n    Returns a vim.vm.device.VirtualDeviceSpec object specifying to add/edit a\n    SATA controller\n\n    sata_controller_label\n        Controller label of the SATA adapter\n\n    operation\n        Type of operation: add or edit\n\n    key\n        Unique key of the device\n\n    bus_number\n        Device bus number property\n    '
    log.trace('Configuring SATA controller sata_controller_label=%s', sata_controller_label)
    sata_spec = vim.vm.device.VirtualDeviceSpec()
    sata_spec.device = vim.vm.device.VirtualAHCIController()
    if operation == 'add':
        sata_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    elif operation == 'edit':
        sata_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    sata_spec.device.key = key
    sata_spec.device.controllerKey = 100
    sata_spec.device.busNumber = bus_number
    if sata_controller_label:
        sata_spec.device.deviceInfo = vim.Description()
        sata_spec.device.deviceInfo.label = sata_controller_label
        sata_spec.device.deviceInfo.summary = sata_controller_label
    return sata_spec

def _apply_cd_drive(drive_label, key, device_type, operation, client_device=None, datastore_iso_file=None, connectable=None, controller_key=200, parent_ref=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns a vim.vm.device.VirtualDeviceSpec object specifying to add/edit a\n    CD/DVD drive\n\n    drive_label\n        Leble of the CD/DVD drive\n\n    key\n        Unique key of the device\n\n    device_type\n        Type of the device: client or iso\n\n    operation\n        Type of operation: add or edit\n\n    client_device\n        Client device properties\n\n    datastore_iso_file\n        ISO properties\n\n    connectable\n        Connection info for the device\n\n    controller_key\n        Controller unique identifier to which we will attach this device\n\n    parent_ref\n        Parent object\n\n    .. code-block:: bash\n\n        cd:\n            adapter: "CD/DVD drive 1"\n            device_type: datastore_iso_file or client_device\n            client_device:\n              mode: atapi or passthrough\n            datastore_iso_file:\n              path: "[share] iso/disk.iso"\n            connectable:\n              start_connected: True\n              allow_guest_control:\n    '
    log.trace('Configuring CD/DVD drive drive_label=%s device_type=%s client_device=%s datastore_iso_file=%s', drive_label, device_type, client_device, datastore_iso_file)
    drive_spec = vim.vm.device.VirtualDeviceSpec()
    drive_spec.device = vim.vm.device.VirtualCdrom()
    drive_spec.device.deviceInfo = vim.Description()
    if operation == 'add':
        drive_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    elif operation == 'edit':
        drive_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    if device_type == 'datastore_iso_file':
        drive_spec.device.backing = vim.vm.device.VirtualCdrom.IsoBackingInfo()
        drive_spec.device.backing.fileName = datastore_iso_file['path']
        datastore = datastore_iso_file['path'].partition('[')[-1].rpartition(']')[0]
        datastore_object = salt.utils.vmware.get_datastores(salt.utils.vmware.get_service_instance_from_managed_object(parent_ref), parent_ref, datastore_names=[datastore])[0]
        if datastore_object:
            drive_spec.device.backing.datastore = datastore_object
        drive_spec.device.deviceInfo.summary = '{}'.format(datastore_iso_file['path'])
    elif device_type == 'client_device':
        if client_device['mode'] == 'passthrough':
            drive_spec.device.backing = vim.vm.device.VirtualCdrom.RemotePassthroughBackingInfo()
        elif client_device['mode'] == 'atapi':
            drive_spec.device.backing = vim.vm.device.VirtualCdrom.RemoteAtapiBackingInfo()
    drive_spec.device.key = key
    drive_spec.device.deviceInfo.label = drive_label
    drive_spec.device.controllerKey = controller_key
    drive_spec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
    if connectable:
        drive_spec.device.connectable.startConnected = connectable['start_connected']
        drive_spec.device.connectable.allowGuestControl = connectable['allow_guest_control']
    return drive_spec

def _set_network_adapter_mapping(domain, gateway, ip_addr, subnet_mask, mac):
    if False:
        i = 10
        return i + 15
    '\n    Returns a vim.vm.customization.AdapterMapping object containing the IP\n    properties of a network adapter card\n\n    domain\n        Domain of the host\n\n    gateway\n        Gateway address\n\n    ip_addr\n        IP address\n\n    subnet_mask\n        Subnet mask\n\n    mac\n        MAC address of the guest\n    '
    adapter_mapping = vim.vm.customization.AdapterMapping()
    adapter_mapping.macAddress = mac
    adapter_mapping.adapter = vim.vm.customization.IPSettings()
    if domain:
        adapter_mapping.adapter.dnsDomain = domain
    if gateway:
        adapter_mapping.adapter.gateway = gateway
    if ip_addr:
        adapter_mapping.adapter.ip = vim.vm.customization.FixedIp(ipAddress=ip_addr)
        adapter_mapping.adapter.subnetMask = subnet_mask
    else:
        adapter_mapping.adapter.ip = vim.vm.customization.DhcpIpGenerator()
    return adapter_mapping

def _apply_serial_port(serial_device_spec, key, operation='add'):
    if False:
        i = 10
        return i + 15
    "\n    Returns a vim.vm.device.VirtualSerialPort representing a serial port\n    component\n\n    serial_device_spec\n        Serial device properties\n\n    key\n        Unique key of the device\n\n    operation\n        Add or edit the given device\n\n    .. code-block:: bash\n\n        serial_ports:\n            adapter: 'Serial port 1'\n            backing:\n              type: uri\n              uri: 'telnet://something:port'\n              direction: <client|server>\n              filename: 'service_uri'\n            connectable:\n              allow_guest_control: True\n              start_connected: True\n            yield: False\n    "
    log.trace('Creating serial port adapter=%s type=%s connectable=%s yield=%s', serial_device_spec['adapter'], serial_device_spec['type'], serial_device_spec['connectable'], serial_device_spec['yield'])
    device_spec = vim.vm.device.VirtualDeviceSpec()
    device_spec.device = vim.vm.device.VirtualSerialPort()
    if operation == 'add':
        device_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    elif operation == 'edit':
        device_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    connect_info = vim.vm.device.VirtualDevice.ConnectInfo()
    type_backing = None
    if serial_device_spec['type'] == 'network':
        type_backing = vim.vm.device.VirtualSerialPort.URIBackingInfo()
        if 'uri' not in serial_device_spec['backing'].keys():
            raise ValueError('vSPC proxy URI not specified in config')
        if 'uri' not in serial_device_spec['backing'].keys():
            raise ValueError('vSPC Direction not specified in config')
        if 'filename' not in serial_device_spec['backing'].keys():
            raise ValueError('vSPC Filename not specified in config')
        type_backing.proxyURI = serial_device_spec['backing']['uri']
        type_backing.direction = serial_device_spec['backing']['direction']
        type_backing.serviceURI = serial_device_spec['backing']['filename']
    if serial_device_spec['type'] == 'pipe':
        type_backing = vim.vm.device.VirtualSerialPort.PipeBackingInfo()
    if serial_device_spec['type'] == 'file':
        type_backing = vim.vm.device.VirtualSerialPort.FileBackingInfo()
    if serial_device_spec['type'] == 'device':
        type_backing = vim.vm.device.VirtualSerialPort.DeviceBackingInfo()
    connect_info.allowGuestControl = serial_device_spec['connectable']['allow_guest_control']
    connect_info.startConnected = serial_device_spec['connectable']['start_connected']
    device_spec.device.backing = type_backing
    device_spec.device.connectable = connect_info
    device_spec.device.unitNumber = 1
    device_spec.device.key = key
    device_spec.device.yieldOnPoll = serial_device_spec['yield']
    return device_spec

def _create_disks(service_instance, disks, scsi_controllers=None, parent=None):
    if False:
        return 10
    "\n    Returns a list of disk specs representing the disks to be created for a\n    virtual machine\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    disks\n        List of disks with properties\n\n    scsi_controllers\n        List of SCSI controllers\n\n    parent\n        Parent object reference\n\n    .. code-block:: bash\n\n        disk:\n          adapter: 'Hard disk 1'\n          size: 16\n          unit: GB\n          address: '0:0'\n          controller: 'SCSI controller 0'\n          thin_provision: False\n          eagerly_scrub: False\n          datastore: 'myshare'\n          filename: 'vm/mydisk.vmdk'\n    "
    disk_specs = []
    keys = range(-2000, -2050, -1)
    if disks:
        devs = [disk['adapter'] for disk in disks]
        log.trace('Creating disks %s', devs)
    for (disk, key) in zip(disks, keys):
        (filename, datastore, datastore_ref) = (None, None, None)
        size = float(disk['size'])
        controller_key = 1000
        if 'address' in disk:
            (controller_bus_number, unit_number) = disk['address'].split(':')
            controller_bus_number = int(controller_bus_number)
            unit_number = int(unit_number)
            controller_key = _get_scsi_controller_key(controller_bus_number, scsi_ctrls=scsi_controllers)
        elif 'controller' in disk:
            for contr in scsi_controllers:
                if contr['label'] == disk['controller']:
                    controller_key = contr['key']
                    break
            else:
                raise salt.exceptions.VMwareObjectNotFoundError('The given controller does not exist: {}'.format(disk['controller']))
        if 'datastore' in disk:
            datastore_ref = salt.utils.vmware.get_datastores(service_instance, parent, datastore_names=[disk['datastore']])[0]
            datastore = disk['datastore']
        if 'filename' in disk:
            filename = disk['filename']
        if not filename and datastore or (filename and (not datastore)):
            raise salt.exceptions.ArgumentValueError('You must specify both filename and datastore attributes to place your disk to a specific datastore {}, {}'.format(datastore, filename))
        disk_spec = _apply_hard_disk(unit_number, key, disk_label=disk['adapter'], size=size, unit=disk['unit'], controller_key=controller_key, operation='add', thin_provision=disk['thin_provision'], eagerly_scrub=disk['eagerly_scrub'] if 'eagerly_scrub' in disk else None, datastore=datastore_ref, filename=filename)
        disk_specs.append(disk_spec)
        unit_number += 1
    return disk_specs

def _create_scsi_devices(scsi_devices):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec objects representing\n    SCSI controllers\n\n    scsi_devices:\n        List of SCSI device properties\n    '
    keys = range(-1000, -1050, -1)
    scsi_specs = []
    if scsi_devices:
        devs = [scsi['adapter'] for scsi in scsi_devices]
        log.trace('Creating SCSI devices %s', devs)
        for (key, scsi_controller) in zip(keys, scsi_devices):
            scsi_spec = _apply_scsi_controller(scsi_controller['adapter'], scsi_controller['type'], scsi_controller['bus_sharing'], key, scsi_controller['bus_number'], 'add')
            scsi_specs.append(scsi_spec)
    return scsi_specs

def _create_network_adapters(network_interfaces, parent=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns a list of vim.vm.device.VirtualDeviceSpec objects representing\n    the interfaces to be created for a virtual machine\n\n    network_interfaces\n        List of network interfaces and properties\n\n    parent\n        Parent object reference\n\n    .. code-block:: bash\n\n        interfaces:\n          adapter: 'Network adapter 1'\n          name: vlan100\n          switch_type: distributed or standard\n          adapter_type: vmxnet3 or vmxnet, vmxnet2, vmxnet3, e1000, e1000e\n          mac: '00:11:22:33:44:55'\n    "
    network_specs = []
    nics_settings = []
    keys = range(-4000, -4050, -1)
    if network_interfaces:
        devs = [inter['adapter'] for inter in network_interfaces]
        log.trace('Creating network interfaces %s', devs)
        for (interface, key) in zip(network_interfaces, keys):
            network_spec = _apply_network_adapter_config(key, interface['name'], interface['adapter_type'], interface['switch_type'], network_adapter_label=interface['adapter'], operation='add', connectable=interface['connectable'] if 'connectable' in interface else None, mac=interface['mac'], parent=parent)
            network_specs.append(network_spec)
            if 'mapping' in interface:
                adapter_mapping = _set_network_adapter_mapping(interface['mapping']['domain'], interface['mapping']['gateway'], interface['mapping']['ip_addr'], interface['mapping']['subnet_mask'], interface['mac'])
                nics_settings.append(adapter_mapping)
    return (network_specs, nics_settings)

def _create_serial_ports(serial_ports):
    if False:
        print('Hello World!')
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec objects representing the\n    serial ports to be created for a virtual machine\n\n    serial_ports\n        Serial port properties\n    '
    ports = []
    keys = range(-9000, -9050, -1)
    if serial_ports:
        devs = [serial['adapter'] for serial in serial_ports]
        log.trace('Creating serial ports %s', devs)
        for (port, key) in zip(serial_ports, keys):
            serial_port_device = _apply_serial_port(port, key, 'add')
            ports.append(serial_port_device)
    return ports

def _create_cd_drives(cd_drives, controllers=None, parent_ref=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec objects representing the\n    CD/DVD drives to be created for a virtual machine\n\n    cd_drives\n        CD/DVD drive properties\n\n    controllers\n        CD/DVD drive controllers (IDE, SATA)\n\n    parent_ref\n        Parent object reference\n    '
    cd_drive_specs = []
    keys = range(-3000, -3050, -1)
    if cd_drives:
        devs = [dvd['adapter'] for dvd in cd_drives]
        log.trace('Creating cd/dvd drives %s', devs)
        for (drive, key) in zip(cd_drives, keys):
            controller_key = 200
            if controllers:
                controller = _get_device_by_label(controllers, drive['controller'])
                controller_key = controller.key
            cd_drive_specs.append(_apply_cd_drive(drive['adapter'], key, drive['device_type'], 'add', client_device=drive['client_device'] if 'client_device' in drive else None, datastore_iso_file=drive['datastore_iso_file'] if 'datastore_iso_file' in drive else None, connectable=drive['connectable'] if 'connectable' in drive else None, controller_key=controller_key, parent_ref=parent_ref))
    return cd_drive_specs

def _get_device_by_key(devices, key):
    if False:
        return 10
    '\n    Returns the device with the given key, raises error if the device is\n    not found.\n\n    devices\n        list of vim.vm.device.VirtualDevice objects\n\n    key\n        Unique key of device\n    '
    device_keys = [d for d in devices if d.key == key]
    if device_keys:
        return device_keys[0]
    else:
        raise salt.exceptions.VMwareObjectNotFoundError(f'Virtual machine device with unique key {key} does not exist')

def _get_device_by_label(devices, label):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the device with the given label, raises error if the device is\n    not found.\n\n    devices\n        list of vim.vm.device.VirtualDevice objects\n\n    key\n        Unique key of device\n    '
    device_labels = [d for d in devices if d.deviceInfo.label == label]
    if device_labels:
        return device_labels[0]
    else:
        raise salt.exceptions.VMwareObjectNotFoundError(f'Virtual machine device with label {label} does not exist')

def _convert_units(devices):
    if False:
        print('Hello World!')
    '\n    Updates the size and unit dictionary values with the new unit values\n\n    devices\n        List of device data objects\n    '
    if devices:
        for device in devices:
            if 'unit' in device and 'size' in device:
                device.update(salt.utils.vmware.convert_to_kb(device['unit'], device['size']))
    else:
        return False
    return True

@_deprecation_message
def compare_vm_configs(new_config, current_config):
    if False:
        while True:
            i = 10
    '\n    Compares virtual machine current and new configuration, the current is the\n    one which is deployed now, and the new is the target config. Returns the\n    differences between the objects in a dictionary, the keys are the\n    configuration parameter keys and the values are differences objects: either\n    list or recursive difference\n\n    new_config:\n        New config dictionary with every available parameter\n\n    current_config\n        Currently deployed configuration\n    '
    diffs = {}
    keys = set(new_config.keys())
    keys.discard('name')
    keys.discard('datacenter')
    keys.discard('datastore')
    for property_key in ('version', 'image'):
        if property_key in keys:
            single_value_diff = recursive_diff({property_key: current_config[property_key]}, {property_key: new_config[property_key]})
            if single_value_diff.diffs:
                diffs[property_key] = single_value_diff
            keys.discard(property_key)
    if 'cpu' in keys:
        keys.remove('cpu')
        cpu_diff = recursive_diff(current_config['cpu'], new_config['cpu'])
        if cpu_diff.diffs:
            diffs['cpu'] = cpu_diff
    if 'memory' in keys:
        keys.remove('memory')
        _convert_units([current_config['memory']])
        _convert_units([new_config['memory']])
        memory_diff = recursive_diff(current_config['memory'], new_config['memory'])
        if memory_diff.diffs:
            diffs['memory'] = memory_diff
    if 'advanced_configs' in keys:
        keys.remove('advanced_configs')
        key = 'advanced_configs'
        advanced_diff = recursive_diff(current_config[key], new_config[key])
        if advanced_diff.diffs:
            diffs[key] = advanced_diff
    if 'disks' in keys:
        keys.remove('disks')
        _convert_units(current_config['disks'])
        _convert_units(new_config['disks'])
        disk_diffs = list_diff(current_config['disks'], new_config['disks'], 'address')
        disk_diffs.remove_diff(diff_key='eagerly_scrub')
        disk_diffs.remove_diff(diff_key='filename')
        disk_diffs.remove_diff(diff_key='adapter')
        if disk_diffs.diffs:
            diffs['disks'] = disk_diffs
    if 'interfaces' in keys:
        keys.remove('interfaces')
        interface_diffs = list_diff(current_config['interfaces'], new_config['interfaces'], 'mac')
        interface_diffs.remove_diff(diff_key='adapter')
        if interface_diffs.diffs:
            diffs['interfaces'] = interface_diffs
    for key in keys:
        if key not in current_config or key not in new_config:
            raise ValueError('A general device {} configuration was not supplied or it was not retrieved from remote configuration'.format(key))
        device_diffs = list_diff(current_config[key], new_config[key], 'adapter')
        if device_diffs.diffs:
            diffs[key] = device_diffs
    return diffs

@_gets_service_instance_via_proxy
@_deprecation_message
def get_vm_config(name, datacenter=None, objects=True, service_instance=None):
    if False:
        i = 10
        return i + 15
    "\n    Queries and converts the virtual machine properties to the available format\n    from the schema. If the objects attribute is True the config objects will\n    have extra properties, like 'object' which will include the\n    vim.vm.device.VirtualDevice, this is necessary for deletion and update\n    actions.\n\n    name\n        Name of the virtual machine\n\n    datacenter\n        Datacenter's name where the virtual machine is available\n\n    objects\n        Indicates whether to return the vmware object properties\n        (eg. object, key) or just the properties which can be set\n\n    service_instance\n        vCenter service instance for connection and configuration\n    "
    properties = ['config.hardware.device', 'config.hardware.numCPU', 'config.hardware.numCoresPerSocket', 'config.nestedHVEnabled', 'config.cpuHotAddEnabled', 'config.cpuHotRemoveEnabled', 'config.hardware.memoryMB', 'config.memoryReservationLockedToMax', 'config.memoryHotAddEnabled', 'config.version', 'config.guestId', 'config.extraConfig', 'name']
    virtual_machine = salt.utils.vmware.get_vm_by_property(service_instance, name, vm_properties=properties, datacenter=datacenter)
    parent_ref = salt.utils.vmware.get_datacenter(service_instance=service_instance, datacenter_name=datacenter)
    current_config = {'name': name}
    current_config['cpu'] = {'count': virtual_machine['config.hardware.numCPU'], 'cores_per_socket': virtual_machine['config.hardware.numCoresPerSocket'], 'nested': virtual_machine['config.nestedHVEnabled'], 'hotadd': virtual_machine['config.cpuHotAddEnabled'], 'hotremove': virtual_machine['config.cpuHotRemoveEnabled']}
    current_config['memory'] = {'size': virtual_machine['config.hardware.memoryMB'], 'unit': 'MB', 'reservation_max': virtual_machine['config.memoryReservationLockedToMax'], 'hotadd': virtual_machine['config.memoryHotAddEnabled']}
    current_config['image'] = virtual_machine['config.guestId']
    current_config['version'] = virtual_machine['config.version']
    current_config['advanced_configs'] = {}
    for extra_conf in virtual_machine['config.extraConfig']:
        try:
            current_config['advanced_configs'][extra_conf.key] = int(extra_conf.value)
        except ValueError:
            current_config['advanced_configs'][extra_conf.key] = extra_conf.value
    current_config['disks'] = []
    current_config['scsi_devices'] = []
    current_config['interfaces'] = []
    current_config['serial_ports'] = []
    current_config['cd_drives'] = []
    current_config['sata_controllers'] = []
    for device in virtual_machine['config.hardware.device']:
        if isinstance(device, vim.vm.device.VirtualSCSIController):
            controller = {}
            controller['adapter'] = device.deviceInfo.label
            controller['bus_number'] = device.busNumber
            bus_sharing = device.sharedBus
            if bus_sharing == 'noSharing':
                controller['bus_sharing'] = 'no_sharing'
            elif bus_sharing == 'virtualSharing':
                controller['bus_sharing'] = 'virtual_sharing'
            elif bus_sharing == 'physicalSharing':
                controller['bus_sharing'] = 'physical_sharing'
            if isinstance(device, vim.vm.device.ParaVirtualSCSIController):
                controller['type'] = 'paravirtual'
            elif isinstance(device, vim.vm.device.VirtualBusLogicController):
                controller['type'] = 'buslogic'
            elif isinstance(device, vim.vm.device.VirtualLsiLogicController):
                controller['type'] = 'lsilogic'
            elif isinstance(device, vim.vm.device.VirtualLsiLogicSASController):
                controller['type'] = 'lsilogic_sas'
            if objects:
                controller['device'] = device.device
                controller['key'] = device.key
                controller['object'] = device
            current_config['scsi_devices'].append(controller)
        if isinstance(device, vim.vm.device.VirtualDisk):
            disk = {}
            disk['adapter'] = device.deviceInfo.label
            disk['size'] = device.capacityInKB
            disk['unit'] = 'KB'
            controller = _get_device_by_key(virtual_machine['config.hardware.device'], device.controllerKey)
            disk['controller'] = controller.deviceInfo.label
            disk['address'] = str(controller.busNumber) + ':' + str(device.unitNumber)
            disk['datastore'] = salt.utils.vmware.get_managed_object_name(device.backing.datastore)
            disk['thin_provision'] = device.backing.thinProvisioned
            disk['eagerly_scrub'] = device.backing.eagerlyScrub
            if objects:
                disk['key'] = device.key
                disk['unit_number'] = device.unitNumber
                disk['bus_number'] = controller.busNumber
                disk['controller_key'] = device.controllerKey
                disk['object'] = device
            current_config['disks'].append(disk)
        if isinstance(device, vim.vm.device.VirtualEthernetCard):
            interface = {}
            interface['adapter'] = device.deviceInfo.label
            interface['adapter_type'] = salt.utils.vmware.get_network_adapter_object_type(device)
            interface['connectable'] = {'allow_guest_control': device.connectable.allowGuestControl, 'connected': device.connectable.connected, 'start_connected': device.connectable.startConnected}
            interface['mac'] = device.macAddress
            if isinstance(device.backing, vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo):
                interface['switch_type'] = 'distributed'
                pg_key = device.backing.port.portgroupKey
                network_ref = salt.utils.vmware.get_mor_by_property(service_instance, vim.DistributedVirtualPortgroup, pg_key, property_name='key', container_ref=parent_ref)
            elif isinstance(device.backing, vim.vm.device.VirtualEthernetCard.NetworkBackingInfo):
                interface['switch_type'] = 'standard'
                network_ref = device.backing.network
            interface['name'] = salt.utils.vmware.get_managed_object_name(network_ref)
            if objects:
                interface['key'] = device.key
                interface['object'] = device
            current_config['interfaces'].append(interface)
        if isinstance(device, vim.vm.device.VirtualCdrom):
            drive = {}
            drive['adapter'] = device.deviceInfo.label
            controller = _get_device_by_key(virtual_machine['config.hardware.device'], device.controllerKey)
            drive['controller'] = controller.deviceInfo.label
            if isinstance(device.backing, vim.vm.device.VirtualCdrom.RemotePassthroughBackingInfo):
                drive['device_type'] = 'client_device'
                drive['client_device'] = {'mode': 'passthrough'}
            if isinstance(device.backing, vim.vm.device.VirtualCdrom.RemoteAtapiBackingInfo):
                drive['device_type'] = 'client_device'
                drive['client_device'] = {'mode': 'atapi'}
            if isinstance(device.backing, vim.vm.device.VirtualCdrom.IsoBackingInfo):
                drive['device_type'] = 'datastore_iso_file'
                drive['datastore_iso_file'] = {'path': device.backing.fileName}
            drive['connectable'] = {'allow_guest_control': device.connectable.allowGuestControl, 'connected': device.connectable.connected, 'start_connected': device.connectable.startConnected}
            if objects:
                drive['key'] = device.key
                drive['controller_key'] = device.controllerKey
                drive['object'] = device
            current_config['cd_drives'].append(drive)
        if isinstance(device, vim.vm.device.VirtualSerialPort):
            port = {}
            port['adapter'] = device.deviceInfo.label
            if isinstance(device.backing, vim.vm.device.VirtualSerialPort.URIBackingInfo):
                port['type'] = 'network'
                port['backing'] = {'uri': device.backing.proxyURI, 'direction': device.backing.direction, 'filename': device.backing.serviceURI}
            if isinstance(device.backing, vim.vm.device.VirtualSerialPort.PipeBackingInfo):
                port['type'] = 'pipe'
            if isinstance(device.backing, vim.vm.device.VirtualSerialPort.FileBackingInfo):
                port['type'] = 'file'
            if isinstance(device.backing, vim.vm.device.VirtualSerialPort.DeviceBackingInfo):
                port['type'] = 'device'
            port['yield'] = device.yieldOnPoll
            port['connectable'] = {'allow_guest_control': device.connectable.allowGuestControl, 'connected': device.connectable.connected, 'start_connected': device.connectable.startConnected}
            if objects:
                port['key'] = device.key
                port['object'] = device
            current_config['serial_ports'].append(port)
        if isinstance(device, vim.vm.device.VirtualSATAController):
            sata = {}
            sata['adapter'] = device.deviceInfo.label
            sata['bus_number'] = device.busNumber
            if objects:
                sata['device'] = device.device
                sata['key'] = device.key
                sata['object'] = device
            current_config['sata_controllers'].append(sata)
    return current_config

def _update_disks(disks_old_new):
    if False:
        return 10
    '\n    Changes the disk size and returns the config spec objects in a list.\n    The controller property cannot be updated, because controller address\n    identifies the disk by the unit and bus number properties.\n\n    disks_diffs\n        List of old and new disk properties, the properties are dictionary\n        objects\n    '
    disk_changes = []
    if disks_old_new:
        devs = [disk['old']['address'] for disk in disks_old_new]
        log.trace('Updating disks %s', devs)
        for item in disks_old_new:
            current_disk = item['old']
            next_disk = item['new']
            difference = recursive_diff(current_disk, next_disk)
            difference.ignore_unset_values = False
            if difference.changed():
                if next_disk['size'] < current_disk['size']:
                    raise salt.exceptions.VMwareSaltError('Disk cannot be downsized size={} unit={} controller_key={} unit_number={}'.format(next_disk['size'], next_disk['unit'], current_disk['controller_key'], current_disk['unit_number']))
                log.trace('Virtual machine disk will be updated size=%s unit=%s controller_key=%s unit_number=%s', next_disk['size'], next_disk['unit'], current_disk['controller_key'], current_disk['unit_number'])
                device_config_spec = _apply_hard_disk(current_disk['unit_number'], current_disk['key'], 'edit', size=next_disk['size'], unit=next_disk['unit'], controller_key=current_disk['controller_key'])
                device_config_spec.device.backing = current_disk['object'].backing
                disk_changes.append(device_config_spec)
    return disk_changes

def _update_scsi_devices(scsis_old_new, current_disks):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec specifying  the scsi\n    properties as input the old and new configs are defined in a dictionary.\n\n    scsi_diffs\n        List of old and new scsi properties\n    '
    device_config_specs = []
    if scsis_old_new:
        devs = [scsi['old']['adapter'] for scsi in scsis_old_new]
        log.trace('Updating SCSI controllers %s', devs)
        for item in scsis_old_new:
            next_scsi = item['new']
            current_scsi = item['old']
            difference = recursive_diff(current_scsi, next_scsi)
            difference.ignore_unset_values = False
            if difference.changed():
                log.trace('Virtual machine scsi device will be updated key=%s bus_number=%s type=%s bus_sharing=%s', current_scsi['key'], current_scsi['bus_number'], next_scsi['type'], next_scsi['bus_sharing'])
                if next_scsi['type'] != current_scsi['type']:
                    device_config_specs.append(_delete_device(current_scsi['object']))
                    device_config_specs.append(_apply_scsi_controller(current_scsi['adapter'], next_scsi['type'], next_scsi['bus_sharing'], current_scsi['key'], current_scsi['bus_number'], 'add'))
                    disks_to_update = []
                    for disk_key in current_scsi['device']:
                        disk_objects = [disk['object'] for disk in current_disks]
                        disks_to_update.append(_get_device_by_key(disk_objects, disk_key))
                    for current_disk in disks_to_update:
                        disk_spec = vim.vm.device.VirtualDeviceSpec()
                        disk_spec.device = current_disk
                        disk_spec.operation = 'edit'
                        device_config_specs.append(disk_spec)
                else:
                    device_config_specs.append(_apply_scsi_controller(current_scsi['adapter'], current_scsi['type'], next_scsi['bus_sharing'], current_scsi['key'], current_scsi['bus_number'], 'edit'))
    return device_config_specs

def _update_network_adapters(interface_old_new, parent):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec specifying\n    configuration(s) for changed network adapters, the adapter type cannot\n    be changed, as input the old and new configs are defined in a dictionary.\n\n    interface_old_new\n        Dictionary with old and new keys which contains the current and the\n        next config for a network device\n\n    parent\n        Parent managed object reference\n    '
    network_changes = []
    if interface_old_new:
        devs = [inter['old']['mac'] for inter in interface_old_new]
        log.trace('Updating network interfaces %s', devs)
        for item in interface_old_new:
            current_interface = item['old']
            next_interface = item['new']
            difference = recursive_diff(current_interface, next_interface)
            difference.ignore_unset_values = False
            if difference.changed():
                log.trace('Virtual machine network adapter will be updated switch_type=%s name=%s adapter_type=%s mac=%s', next_interface['switch_type'], next_interface['name'], current_interface['adapter_type'], current_interface['mac'])
                device_config_spec = _apply_network_adapter_config(current_interface['key'], next_interface['name'], current_interface['adapter_type'], next_interface['switch_type'], operation='edit', mac=current_interface['mac'], parent=parent)
                network_changes.append(device_config_spec)
    return network_changes

def _update_serial_ports(serial_old_new):
    if False:
        print('Hello World!')
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec specifying to edit a\n    deployed serial port configuration to the new given config\n\n    serial_old_new\n         Dictionary with old and new keys which contains the current and the\n          next config for a serial port device\n    '
    serial_changes = []
    if serial_old_new:
        devs = [serial['old']['adapter'] for serial in serial_old_new]
        log.trace('Updating serial ports %s', devs)
        for item in serial_old_new:
            current_serial = item['old']
            next_serial = item['new']
            difference = recursive_diff(current_serial, next_serial)
            difference.ignore_unset_values = False
            if difference.changed():
                serial_changes.append(_apply_serial_port(next_serial, current_serial['key'], 'edit'))
        return serial_changes

def _update_cd_drives(drives_old_new, controllers=None, parent=None):
    if False:
        return 10
    '\n    Returns a list of vim.vm.device.VirtualDeviceSpec specifying to edit a\n    deployed cd drive configuration to the new given config\n\n    drives_old_new\n        Dictionary with old and new keys which contains the current and the\n        next config for a cd drive\n\n    controllers\n        Controller device list\n\n    parent\n        Managed object reference of the parent object\n    '
    cd_changes = []
    if drives_old_new:
        devs = [drive['old']['adapter'] for drive in drives_old_new]
        log.trace('Updating cd/dvd drives %s', devs)
        for item in drives_old_new:
            current_drive = item['old']
            new_drive = item['new']
            difference = recursive_diff(current_drive, new_drive)
            difference.ignore_unset_values = False
            if difference.changed():
                if controllers:
                    controller = _get_device_by_label(controllers, new_drive['controller'])
                    controller_key = controller.key
                else:
                    controller_key = current_drive['controller_key']
                cd_changes.append(_apply_cd_drive(current_drive['adapter'], current_drive['key'], new_drive['device_type'], 'edit', client_device=new_drive['client_device'] if 'client_device' in new_drive else None, datastore_iso_file=new_drive['datastore_iso_file'] if 'datastore_iso_file' in new_drive else None, connectable=new_drive['connectable'], controller_key=controller_key, parent_ref=parent))
    return cd_changes

def _delete_device(device):
    if False:
        return 10
    '\n    Returns a vim.vm.device.VirtualDeviceSpec specifying to remove a virtual\n    machine device\n\n    device\n        Device data type object\n    '
    log.trace('Deleting device with type %s', type(device))
    device_spec = vim.vm.device.VirtualDeviceSpec()
    device_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.remove
    device_spec.device = device
    return device_spec

def _get_client(server, username, password, verify_ssl=None, ca_bundle=None):
    if False:
        while True:
            i = 10
    '\n    Establish client through proxy or with user provided credentials.\n\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :returns:\n        vSphere Client instance.\n    :rtype:\n        vSphere.Client\n    '
    details = None
    if not (server and username and password):
        details = __salt__['vcenter.get_details']()
        server = details['vcenter']
        username = details['username']
        password = details['password']
    if verify_ssl is None:
        if details is None:
            details = __salt__['vcenter.get_details']()
        verify_ssl = details.get('verify_ssl', True)
        if verify_ssl is None:
            verify_ssl = True
    if ca_bundle is None:
        if details is None:
            details = __salt__['vcenter.get_details']()
        ca_bundle = details.get('ca_bundle', None)
    if verify_ssl is False and ca_bundle is not None:
        log.error('Cannot set verify_ssl to False and ca_bundle together')
        return False
    if ca_bundle:
        ca_bundle = salt.utils.http.get_ca_bundle({'ca_bundle': ca_bundle})
    client = salt.utils.vmware.get_vsphere_client(server=server, username=username, password=password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    return client

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_tag_categories(server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List existing categories a user has access to.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.list_tag_categories\n\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :returns:\n        Value(s) of category_id.\n    :rtype:\n        list of str\n    '
    categories = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        categories = client.tagging.Category.list()
    return {'Categories': categories}

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_tags(server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        while True:
            i = 10
    '\n    List existing tags a user has access to.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.list_tags\n\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :return:\n        Value(s) of tag_id.\n    :rtype:\n        list of str\n    '
    tags = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        tags = client.tagging.Tag.list()
    return {'Tags': tags}

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def attach_tag(object_id, tag_id, managed_obj='ClusterComputeResource', server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        i = 10
        return i + 15
    '\n    Attach an existing tag to an input object.\n\n    The tag needs to meet the cardinality (`CategoryModel.cardinality`) and\n    associability (`CategoryModel.associable_types`) criteria in order to be\n    eligible for attachment. If the tag is already attached to the object,\n    then this method is a no-op and an error will not be thrown. To invoke\n    this method, you need the attach tag privilege on the tag and the read\n    privilege on the object.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.attach_tag domain-c2283                 urn:vmomi:InventoryServiceTag:b55ecc77-f4a5-49f8-ab52-38865467cfbe:GLOBAL\n\n    :param str object_id:\n        The identifier of the input object.\n    :param str tag_id:\n        The identifier of the tag object.\n    :param str managed_obj:\n        Classes that contain methods for creating and deleting resources\n        typically contain a class attribute specifying the resource type\n        for the resources being created and deleted.\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :return:\n        The list of all tag identifiers that correspond to the\n        tags attached to the given object.\n    :rtype:\n        list of tags\n    :raise: Unauthorized\n        if you do not have the privilege to read the object.\n    :raise: Unauthenticated\n        if the user can not be authenticated.\n    '
    tag_attached = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        dynamic_id = DynamicID(type=managed_obj, id=object_id)
        try:
            tag_attached = client.tagging.TagAssociation.attach(tag_id=tag_id, object_id=dynamic_id)
        except vsphere_errors:
            log.warning('Unable to attach tag. Check user privileges and object_id (must be a string).')
    return {'Tag attached': tag_attached}

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def list_attached_tags(object_id, managed_obj='ClusterComputeResource', server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List existing tags a user has access to.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.list_attached_tags domain-c2283\n\n    :param str object_id:\n        The identifier of the input object.\n    :param str managed_obj:\n        Classes that contain methods for creating and deleting resources\n        typically contain a class attribute specifying the resource type\n        for the resources being created and deleted.\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :return:\n        The list of all tag identifiers that correspond to the\n        tags attached to the given object.\n    :rtype:\n        list of tags\n    :raise: Unauthorized\n        if you do not have the privilege to read the object.\n    :raise: Unauthenticated\n        if the user can not be authenticated.\n    '
    attached_tags = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        dynamic_id = DynamicID(type=managed_obj, id=object_id)
        try:
            attached_tags = client.tagging.TagAssociation.list_attached_tags(dynamic_id)
        except vsphere_errors:
            log.warning('Unable to list attached tags. Check user privileges and object_id (must be a string).')
    return {'Attached tags': attached_tags}

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_tag_category(name, description, cardinality, server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a category with given cardinality.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.create_tag_category\n\n    :param str name:\n        Name of tag category to create (ex. Machine, OS, Availability, etc.)\n    :param str description:\n        Given description of tag category.\n    :param str cardinality:\n        The associated cardinality (SINGLE, MULTIPLE) of the category.\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :return:\n        Identifier of the created category.\n    :rtype:\n        str\n    :raise: AlreadyExists\n        if the name` provided in the create_spec is the name of an already\n        existing category.\n    :raise: InvalidArgument\n        if any of the information in the create_spec is invalid.\n    :raise: Unauthorized\n        if you do not have the privilege to create a category.\n    '
    category_created = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        if cardinality == 'SINGLE':
            cardinality = CategoryModel.Cardinality.SINGLE
        elif cardinality == 'MULTIPLE':
            cardinality = CategoryModel.Cardinality.MULTIPLE
        else:
            cardinality = None
        create_spec = client.tagging.Category.CreateSpec()
        create_spec.name = name
        create_spec.description = description
        create_spec.cardinality = cardinality
        associable_types = set()
        create_spec.associable_types = associable_types
        try:
            category_created = client.tagging.Category.create(create_spec)
        except vsphere_errors:
            log.warning('Unable to create tag category. Check user privilege and see if category exists.')
    return {'Category created': category_created}

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def delete_tag_category(category_id, server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        while True:
            i = 10
    '\n    Delete a category.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.delete_tag_category\n\n    :param str category_id:\n        The identifier of category to be deleted.\n        The parameter must be an identifier for the resource type:\n        ``com.vmware.cis.tagging.Category``.\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :raise: NotFound\n        if the tag for the given tag_id does not exist in the system.\n    :raise: Unauthorized\n        if you do not have the privilege to delete the tag.\n    :raise: Unauthenticated\n        if the user can not be authenticated.\n    '
    category_deleted = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        try:
            category_deleted = client.tagging.Category.delete(category_id)
        except vsphere_errors:
            log.warning('Unable to delete tag category. Check user privilege and see if category exists.')
    return {'Category deleted': category_deleted}

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_tag(name, description, category_id, server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        i = 10
        return i + 15
    '\n    Create a tag under a category with given description.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.create_tag\n\n    :param basestring server:\n        Target DNS or IP of vCenter client.\n    :param basestring username:\n         Username associated with the vCenter client.\n    :param basestring password:\n        Password associated with the vCenter client.\n    :param str name:\n        Name of tag category to create (ex. Machine, OS, Availability, etc.)\n    :param str description:\n        Given description of tag category.\n    :param str category_id:\n        Value of category_id representative of the category created previously.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :return:\n        The identifier of the created tag.\n    :rtype:\n        str\n    :raise: AlreadyExists\n        if the name provided in the create_spec is the name of an already\n        existing tag in the input category.\n    :raise: InvalidArgument\n        if any of the input information in the create_spec is invalid.\n    :raise: NotFound\n        if the category for in the given create_spec does not exist in\n        the system.\n    :raise: Unauthorized\n        if you do not have the privilege to create tag.\n    '
    tag_created = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        create_spec = client.tagging.Tag.CreateSpec()
        create_spec.name = name
        create_spec.description = description
        create_spec.category_id = category_id
        try:
            tag_created = client.tagging.Tag.create(create_spec)
        except vsphere_errors:
            log.warning('Unable to create tag. Check user privilege and see if category exists.')
    return {'Tag created': tag_created}

@depends(HAS_PYVMOMI, HAS_VSPHERE_SDK)
@_supports_proxies('vcenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def delete_tag(tag_id, server=None, username=None, password=None, service_instance=None, verify_ssl=None, ca_bundle=None):
    if False:
        while True:
            i = 10
    '\n    Delete a tag.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n            salt vm_minion vsphere.delete_tag\n\n    :param str tag_id:\n        The identifier of tag to be deleted.\n        The parameter must be an identifier for the resource type:\n        ``com.vmware.cis.tagging.Tag``.\n    :param basestring server:\n        Target DNS or IP of vCenter center.\n    :param basestring username:\n        Username associated with the vCenter center.\n    :param basestring password:\n        Password associated with the vCenter center.\n    :param boolean verify_ssl:\n        Verify the SSL certificate. Default: True\n    :param basestring ca_bundle:\n        Path to the ca bundle to use when verifying SSL certificates.\n    :raise: AlreadyExists\n        if the name provided in the create_spec is the name of an already\n        existing category.\n    :raise: InvalidArgument\n        if any of the information in the create_spec is invalid.\n    :raise: Unauthorized\n        if you do not have the privilege to create a category.\n    '
    tag_deleted = None
    client = _get_client(server, username, password, verify_ssl=verify_ssl, ca_bundle=ca_bundle)
    if client:
        try:
            tag_deleted = client.tagging.Tag.delete(tag_id)
        except vsphere_errors:
            log.warning('Unable to delete category. Check user privileges and that category exists.')
    return {'Tag deleted': tag_deleted}

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def create_vm(vm_name, cpu, memory, image, version, datacenter, datastore, placement, interfaces, disks, scsi_devices, serial_ports=None, ide_controllers=None, sata_controllers=None, cd_drives=None, advanced_configs=None, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Creates a virtual machine container.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt vm_minion vsphere.create_vm vm_name=vmname cpu='{count: 2, nested: True}' ...\n\n    vm_name\n        Name of the virtual machine\n\n    cpu\n        Properties of CPUs for freshly created machines\n\n    memory\n        Memory size for freshly created machines\n\n    image\n        Virtual machine guest OS version identifier\n        VirtualMachineGuestOsIdentifier\n\n    version\n        Virtual machine container hardware version\n\n    datacenter\n        Datacenter where the virtual machine will be deployed (mandatory)\n\n    datastore\n        Datastore where the virtual machine files will be placed\n\n    placement\n        Resource pool or cluster or host or folder where the virtual machine\n        will be deployed\n\n    devices\n        interfaces\n\n        .. code-block:: bash\n\n            interfaces:\n              adapter: 'Network adapter 1'\n              name: vlan100\n              switch_type: distributed or standard\n              adapter_type: vmxnet3 or vmxnet, vmxnet2, vmxnet3, e1000, e1000e\n              mac: '00:11:22:33:44:55'\n              connectable:\n                allow_guest_control: True\n                connected: True\n                start_connected: True\n\n        disks\n\n        .. code-block:: bash\n\n            disks:\n              adapter: 'Hard disk 1'\n              size: 16\n              unit: GB\n              address: '0:0'\n              controller: 'SCSI controller 0'\n              thin_provision: False\n              eagerly_scrub: False\n              datastore: 'myshare'\n              filename: 'vm/mydisk.vmdk'\n\n        scsi_devices\n\n        .. code-block:: bash\n\n            scsi_devices:\n              controller: 'SCSI controller 0'\n              type: paravirtual\n              bus_sharing: no_sharing\n\n        serial_ports\n\n        .. code-block:: bash\n\n            serial_ports:\n              adapter: 'Serial port 1'\n              type: network\n              backing:\n                uri: 'telnet://something:port'\n                direction: <client|server>\n                filename: 'service_uri'\n              connectable:\n                allow_guest_control: True\n                connected: True\n                start_connected: True\n              yield: False\n\n        cd_drives\n\n        .. code-block:: bash\n\n            cd_drives:\n              adapter: 'CD/DVD drive 0'\n              controller: 'IDE 0'\n              device_type: datastore_iso_file\n              datastore_iso_file:\n                path: path_to_iso\n              connectable:\n                allow_guest_control: True\n                connected: True\n                start_connected: True\n\n    advanced_config\n        Advanced config parameters to be set for the virtual machine\n    "
    container_object = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    (resourcepool_object, placement_object) = salt.utils.vmware.get_placement(service_instance, datacenter, placement=placement)
    folder_object = salt.utils.vmware.get_folder(service_instance, datacenter, placement)
    config_spec = vim.vm.ConfigSpec()
    config_spec.name = vm_name
    config_spec.guestId = image
    config_spec.files = vim.vm.FileInfo()
    datastore_object = salt.utils.vmware.get_datastores(service_instance, placement_object, datastore_names=[datastore])[0]
    if not datastore_object:
        raise salt.exceptions.ArgumentValueError(f"Specified datastore: '{datastore}' does not exist.")
    try:
        ds_summary = salt.utils.vmware.get_properties_of_managed_object(datastore_object, 'summary.type')
        if 'summary.type' in ds_summary and ds_summary['summary.type'] == 'vsan':
            log.trace('The vmPathName should be the datastore name if the datastore type is vsan')
            config_spec.files.vmPathName = f'[{datastore}]'
        else:
            config_spec.files.vmPathName = '[{0}] {1}/{1}.vmx'.format(datastore, vm_name)
    except salt.exceptions.VMwareApiError:
        config_spec.files.vmPathName = '[{0}] {1}/{1}.vmx'.format(datastore, vm_name)
    cd_controllers = []
    if version:
        _apply_hardware_version(version, config_spec, 'add')
    if cpu:
        _apply_cpu_config(config_spec, cpu)
    if memory:
        _apply_memory_config(config_spec, memory)
    if scsi_devices:
        scsi_specs = _create_scsi_devices(scsi_devices)
        config_spec.deviceChange.extend(scsi_specs)
    if disks:
        scsi_controllers = [spec.device for spec in scsi_specs]
        disk_specs = _create_disks(service_instance, disks, scsi_controllers=scsi_controllers, parent=container_object)
        config_spec.deviceChange.extend(disk_specs)
    if interfaces:
        (interface_specs, nic_settings) = _create_network_adapters(interfaces, parent=container_object)
        config_spec.deviceChange.extend(interface_specs)
    if serial_ports:
        serial_port_specs = _create_serial_ports(serial_ports)
        config_spec.deviceChange.extend(serial_port_specs)
    if ide_controllers:
        ide_specs = _create_ide_controllers(ide_controllers)
        config_spec.deviceChange.extend(ide_specs)
        cd_controllers.extend(ide_specs)
    if sata_controllers:
        sata_specs = _create_sata_controllers(sata_controllers)
        config_spec.deviceChange.extend(sata_specs)
        cd_controllers.extend(sata_specs)
    if cd_drives:
        cd_drive_specs = _create_cd_drives(cd_drives, controllers=cd_controllers, parent_ref=container_object)
        config_spec.deviceChange.extend(cd_drive_specs)
    if advanced_configs:
        _apply_advanced_config(config_spec, advanced_configs)
    salt.utils.vmware.create_vm(vm_name, config_spec, folder_object, resourcepool_object, placement_object)
    return {'create_vm': True}

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def update_vm(vm_name, cpu=None, memory=None, image=None, version=None, interfaces=None, disks=None, scsi_devices=None, serial_ports=None, datacenter=None, datastore=None, cd_dvd_drives=None, sata_controllers=None, advanced_configs=None, service_instance=None):
    if False:
        return 10
    '\n    Updates the configuration of the virtual machine if the config differs\n\n    vm_name\n        Virtual Machine name to be updated\n\n    cpu\n        CPU configuration options\n\n    memory\n        Memory configuration options\n\n    version\n        Virtual machine container hardware version\n\n    image\n        Virtual machine guest OS version identifier\n        VirtualMachineGuestOsIdentifier\n\n    interfaces\n        Network interfaces configuration options\n\n    disks\n        Disks configuration options\n\n    scsi_devices\n        SCSI devices configuration options\n\n    serial_ports\n        Serial ports configuration options\n\n    datacenter\n        Datacenter where the virtual machine is available\n\n    datastore\n        Datastore where the virtual machine config files are available\n\n    cd_dvd_drives\n        CD/DVD drives configuration options\n\n    advanced_config\n        Advanced config parameters to be set for the virtual machine\n\n    service_instance\n        vCenter service instance for connection and configuration\n    '
    current_config = get_vm_config(vm_name, datacenter=datacenter, objects=True, service_instance=service_instance)
    diffs = compare_vm_configs({'name': vm_name, 'cpu': cpu, 'memory': memory, 'image': image, 'version': version, 'interfaces': interfaces, 'disks': disks, 'scsi_devices': scsi_devices, 'serial_ports': serial_ports, 'datacenter': datacenter, 'datastore': datastore, 'cd_drives': cd_dvd_drives, 'sata_controllers': sata_controllers, 'advanced_configs': advanced_configs}, current_config)
    config_spec = vim.vm.ConfigSpec()
    datacenter_ref = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    vm_ref = salt.utils.vmware.get_mor_by_property(service_instance, vim.VirtualMachine, vm_name, property_name='name', container_ref=datacenter_ref)
    difference_keys = diffs.keys()
    if 'cpu' in difference_keys:
        if diffs['cpu'].changed() != set():
            _apply_cpu_config(config_spec, diffs['cpu'].current_dict)
    if 'memory' in difference_keys:
        if diffs['memory'].changed() != set():
            _apply_memory_config(config_spec, diffs['memory'].current_dict)
    if 'advanced_configs' in difference_keys:
        _apply_advanced_config(config_spec, diffs['advanced_configs'].new_values, vm_ref.config.extraConfig)
    if 'version' in difference_keys:
        _apply_hardware_version(version, config_spec, 'edit')
    if 'image' in difference_keys:
        config_spec.guestId = image
    new_scsi_devices = []
    if 'scsi_devices' in difference_keys and 'disks' in current_config:
        scsi_changes = []
        scsi_changes.extend(_update_scsi_devices(diffs['scsi_devices'].intersect, current_config['disks']))
        for item in diffs['scsi_devices'].removed:
            scsi_changes.append(_delete_device(item['object']))
        new_scsi_devices = _create_scsi_devices(diffs['scsi_devices'].added)
        scsi_changes.extend(new_scsi_devices)
        config_spec.deviceChange.extend(scsi_changes)
    if 'disks' in difference_keys:
        disk_changes = []
        disk_changes.extend(_update_disks(diffs['disks'].intersect))
        for item in diffs['disks'].removed:
            disk_changes.append(_delete_device(item['object']))
        scsi_controllers = [dev['object'] for dev in current_config['scsi_devices']]
        scsi_controllers.extend([device_spec.device for device_spec in new_scsi_devices])
        disk_changes.extend(_create_disks(service_instance, diffs['disks'].added, scsi_controllers=scsi_controllers, parent=datacenter_ref))
        config_spec.deviceChange.extend(disk_changes)
    if 'interfaces' in difference_keys:
        network_changes = []
        network_changes.extend(_update_network_adapters(diffs['interfaces'].intersect, datacenter_ref))
        for item in diffs['interfaces'].removed:
            network_changes.append(_delete_device(item['object']))
        (adapters, nics) = _create_network_adapters(diffs['interfaces'].added, datacenter_ref)
        network_changes.extend(adapters)
        config_spec.deviceChange.extend(network_changes)
    if 'serial_ports' in difference_keys:
        serial_changes = []
        serial_changes.extend(_update_serial_ports(diffs['serial_ports'].intersect))
        for item in diffs['serial_ports'].removed:
            serial_changes.append(_delete_device(item['object']))
        serial_changes.extend(_create_serial_ports(diffs['serial_ports'].added))
        config_spec.deviceChange.extend(serial_changes)
    new_controllers = []
    if 'sata_controllers' in difference_keys:
        sata_specs = _create_sata_controllers(diffs['sata_controllers'].added)
        for item in diffs['sata_controllers'].removed:
            sata_specs.append(_delete_device(item['object']))
        new_controllers.extend(sata_specs)
        config_spec.deviceChange.extend(sata_specs)
    if 'cd_drives' in difference_keys:
        cd_changes = []
        controllers = [dev['object'] for dev in current_config['sata_controllers']]
        controllers.extend([device_spec.device for device_spec in new_controllers])
        cd_changes.extend(_update_cd_drives(diffs['cd_drives'].intersect, controllers=controllers, parent=datacenter_ref))
        for item in diffs['cd_drives'].removed:
            cd_changes.append(_delete_device(item['object']))
        cd_changes.extend(_create_cd_drives(diffs['cd_drives'].added, controllers=controllers, parent_ref=datacenter_ref))
        config_spec.deviceChange.extend(cd_changes)
    if difference_keys:
        salt.utils.vmware.update_vm(vm_ref, config_spec)
    changes = {}
    for (key, properties) in diffs.items():
        if isinstance(properties, salt.utils.listdiffer.ListDictDiffer):
            properties.remove_diff(diff_key='object', diff_list='intersect')
            properties.remove_diff(diff_key='key', diff_list='intersect')
            properties.remove_diff(diff_key='object', diff_list='removed')
            properties.remove_diff(diff_key='key', diff_list='removed')
        changes[key] = properties.diffs
    return changes

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def register_vm(name, datacenter, placement, vmx_path, service_instance=None):
    if False:
        i = 10
        return i + 15
    '\n    Registers a virtual machine to the inventory with the given vmx file.\n    Returns comments and change list\n\n    name\n        Name of the virtual machine\n\n    datacenter\n        Datacenter of the virtual machine\n\n    placement\n        Placement dictionary of the virtual machine, host or cluster\n\n    vmx_path:\n        Full path to the vmx file, datastore name should be included\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n    '
    log.trace('Registering virtual machine with properties datacenter=%s, placement=%s, vmx_path=%s', datacenter, placement, vmx_path)
    datacenter_object = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    if 'cluster' in placement:
        cluster_obj = salt.utils.vmware.get_cluster(datacenter_object, placement['cluster'])
        cluster_props = salt.utils.vmware.get_properties_of_managed_object(cluster_obj, properties=['resourcePool'])
        if 'resourcePool' in cluster_props:
            resourcepool = cluster_props['resourcePool']
        else:
            raise salt.exceptions.VMwareObjectRetrievalError("The cluster's resource pool object could not be retrieved.")
        salt.utils.vmware.register_vm(datacenter_object, name, vmx_path, resourcepool)
    elif 'host' in placement:
        hosts = salt.utils.vmware.get_hosts(service_instance, datacenter_name=datacenter, host_names=[placement['host']])
        if not hosts:
            raise salt.exceptions.VMwareObjectRetrievalError("ESXi host named '{}' wasn't found.".format(placement['host']))
        host_obj = hosts[0]
        host_props = salt.utils.vmware.get_properties_of_managed_object(host_obj, properties=['parent'])
        if 'parent' in host_props:
            host_parent = host_props['parent']
            parent = salt.utils.vmware.get_properties_of_managed_object(host_parent, properties=['parent'])
            if 'parent' in parent:
                resourcepool = parent['parent']
            else:
                raise salt.exceptions.VMwareObjectRetrievalError("The host parent's parent object could not be retrieved.")
        else:
            raise salt.exceptions.VMwareObjectRetrievalError("The host's parent object could not be retrieved.")
        salt.utils.vmware.register_vm(datacenter_object, name, vmx_path, resourcepool, host_object=host_obj)
    result = {'comment': 'Virtual machine registration action succeeded', 'changes': {'register_vm': True}}
    return result

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def power_on_vm(name, datacenter=None, service_instance=None):
    if False:
        return 10
    "\n    Powers on a virtual machine specified by its name.\n\n    name\n        Name of the virtual machine\n\n    datacenter\n        Datacenter of the virtual machine\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.power_on_vm name=my_vm\n\n    "
    log.trace('Powering on virtual machine %s', name)
    vm_properties = ['name', 'summary.runtime.powerState']
    virtual_machine = salt.utils.vmware.get_vm_by_property(service_instance, name, datacenter=datacenter, vm_properties=vm_properties)
    if virtual_machine['summary.runtime.powerState'] == 'poweredOn':
        result = {'comment': 'Virtual machine is already powered on', 'changes': {'power_on': True}}
        return result
    salt.utils.vmware.power_cycle_vm(virtual_machine['object'], action='on')
    result = {'comment': 'Virtual machine power on action succeeded', 'changes': {'power_on': True}}
    return result

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def power_off_vm(name, datacenter=None, service_instance=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Powers off a virtual machine specified by its name.\n\n    name\n        Name of the virtual machine\n\n    datacenter\n        Datacenter of the virtual machine\n\n    service_instance\n        Service instance (vim.ServiceInstance) of the vCenter.\n        Default is None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.power_off_vm name=my_vm\n\n    "
    log.trace('Powering off virtual machine %s', name)
    vm_properties = ['name', 'summary.runtime.powerState']
    virtual_machine = salt.utils.vmware.get_vm_by_property(service_instance, name, datacenter=datacenter, vm_properties=vm_properties)
    if virtual_machine['summary.runtime.powerState'] == 'poweredOff':
        result = {'comment': 'Virtual machine is already powered off', 'changes': {'power_off': True}}
        return result
    salt.utils.vmware.power_cycle_vm(virtual_machine['object'], action='off')
    result = {'comment': 'Virtual machine power off action succeeded', 'changes': {'power_off': True}}
    return result

def _remove_vm(name, datacenter, service_instance, placement=None, power_off=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to remove a virtual machine\n\n    name\n        Name of the virtual machine\n\n    service_instance\n        vCenter service instance for connection and configuration\n\n    datacenter\n        Datacenter of the virtual machine\n\n    placement\n        Placement information of the virtual machine\n    '
    results = {}
    if placement:
        (resourcepool_object, placement_object) = salt.utils.vmware.get_placement(service_instance, datacenter, placement)
    else:
        placement_object = salt.utils.vmware.get_datacenter(service_instance, datacenter)
    if power_off:
        power_off_vm(name, datacenter, service_instance)
        results['powered_off'] = True
    vm_ref = salt.utils.vmware.get_mor_by_property(service_instance, vim.VirtualMachine, name, property_name='name', container_ref=placement_object)
    if not vm_ref:
        raise salt.exceptions.VMwareObjectRetrievalError('The virtual machine object {} in datacenter {} was not found'.format(name, datacenter))
    return (results, vm_ref)

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def delete_vm(name, datacenter, placement=None, power_off=False, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Deletes a virtual machine defined by name and placement\n\n    name\n        Name of the virtual machine\n\n    datacenter\n        Datacenter of the virtual machine\n\n    placement\n        Placement information of the virtual machine\n\n    service_instance\n        vCenter service instance for connection and configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.delete_vm name=my_vm datacenter=my_datacenter\n\n    "
    results = {}
    schema = ESXVirtualMachineDeleteSchema.serialize()
    try:
        jsonschema.validate({'name': name, 'datacenter': datacenter, 'placement': placement}, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise InvalidConfigError(exc)
    (results, vm_ref) = _remove_vm(name, datacenter, service_instance=service_instance, placement=placement, power_off=power_off)
    salt.utils.vmware.delete_vm(vm_ref)
    results['deleted_vm'] = True
    return results

@depends(HAS_PYVMOMI)
@_supports_proxies('esxvm', 'esxcluster', 'esxdatacenter')
@_gets_service_instance_via_proxy
@_deprecation_message
def unregister_vm(name, datacenter, placement=None, power_off=False, service_instance=None):
    if False:
        while True:
            i = 10
    "\n    Unregisters a virtual machine defined by name and placement\n\n    name\n        Name of the virtual machine\n\n    datacenter\n        Datacenter of the virtual machine\n\n    placement\n        Placement information of the virtual machine\n\n    service_instance\n        vCenter service instance for connection and configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vsphere.unregister_vm name=my_vm datacenter=my_datacenter\n\n    "
    results = {}
    schema = ESXVirtualMachineUnregisterSchema.serialize()
    try:
        jsonschema.validate({'name': name, 'datacenter': datacenter, 'placement': placement}, schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise InvalidConfigError(exc)
    (results, vm_ref) = _remove_vm(name, datacenter, service_instance=service_instance, placement=placement, power_off=power_off)
    salt.utils.vmware.unregister_vm(vm_ref)
    results['unregistered_vm'] = True
    return results