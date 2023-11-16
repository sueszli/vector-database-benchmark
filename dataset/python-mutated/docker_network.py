"""
Management of Docker networks

.. versionadded:: 2017.7.0

:depends: docker_ Python module

.. note::
    Older releases of the Python bindings for Docker were called docker-py_ in
    PyPI. All releases of docker_, and releases of docker-py_ >= 1.6.0 are
    supported. These python bindings can easily be installed using
    :py:func:`pip.install <salt.modules.pip.install>`:

    .. code-block:: bash

        salt myminion pip.install docker

    To upgrade from docker-py_ to docker_, you must first uninstall docker-py_,
    and then install docker_:

    .. code-block:: bash

        salt myminion pip.uninstall docker-py
        salt myminion pip.install docker

.. _docker: https://pypi.python.org/pypi/docker
.. _docker-py: https://pypi.python.org/pypi/docker-py

These states were moved from the :mod:`docker <salt.states.docker>` state
module (formerly called **dockerng**) in the 2017.7.0 release.
"""
import copy
import logging
import random
import string
import salt.utils.dockermod.translate.network
from salt._compat import ipaddress
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'docker_network'
__virtual_aliases__ = ('moby_network',)
__deprecated__ = (3009, 'docker', 'https://github.com/saltstack/saltext-docker')

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the docker execution module is available\n    '
    if 'docker.version' in __salt__:
        return __virtualname__
    return (False, __salt__.missing_fun_string('docker.version'))

def _normalize_pools(existing, desired):
    if False:
        while True:
            i = 10
    pools = {'existing': {4: None, 6: None}, 'desired': {4: None, 6: None}}
    for pool in existing['Config']:
        subnet = ipaddress.ip_network(pool.get('Subnet'))
        pools['existing'][subnet.version] = pool
    for pool in desired['Config']:
        subnet = ipaddress.ip_network(pool.get('Subnet'))
        if pools['desired'][subnet.version] is not None:
            raise ValueError(f'Only one IPv{subnet.version} pool is permitted')
        else:
            pools['desired'][subnet.version] = pool
    if pools['desired'][6] and (not pools['desired'][4]):
        raise ValueError('An IPv4 pool is required when an IPv6 pool is used. See the documentation for details.')
    existing['Config'] = [pools['existing'][x] for x in (4, 6) if pools['existing'][x] is not None]
    desired['Config'] = [pools['desired'][x] for x in (4, 6) if pools['desired'][x] is not None]

def present(name, skip_translate=None, ignore_collisions=False, validate_ip_addrs=True, containers=None, reconnect=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionchanged:: 2018.3.0\n        Support added for network configuration options other than ``driver``\n        and ``driver_opts``, as well as IPAM configuration.\n\n    Ensure that a network is present\n\n    .. note::\n        This state supports all arguments for network and IPAM pool\n        configuration which are available for the release of docker-py\n        installed on the minion. For that reason, the arguments described below\n        in the :ref:`NETWORK CONFIGURATION\n        <salt-states-docker-network-present-netconf>` and :ref:`IP ADDRESS\n        MANAGEMENT (IPAM) <salt-states-docker-network-present-ipam>` sections\n        may not accurately reflect what is available on the minion. The\n        :py:func:`docker.get_client_args\n        <salt.modules.dockermod.get_client_args>` function can be used to check\n        the available arguments for the installed version of docker-py (they\n        are found in the ``network_config`` and ``ipam_config`` sections of the\n        return data), but Salt will not prevent a user from attempting to use\n        an argument which is unsupported in the release of Docker which is\n        installed. In those cases, network creation be attempted but will fail.\n\n    name\n        Network name\n\n    skip_translate\n        This function translates Salt SLS input into the format which\n        docker-py expects. However, in the event that Salt's translation logic\n        fails (due to potential changes in the Docker Remote API, or to bugs in\n        the translation code), this argument can be used to exert granular\n        control over which arguments are translated and which are not.\n\n        Pass this argument as a comma-separated list (or Python list) of\n        arguments, and translation for each passed argument name will be\n        skipped. Alternatively, pass ``True`` and *all* translation will be\n        skipped.\n\n        Skipping tranlsation allows for arguments to be formatted directly in\n        the format which docker-py expects. This allows for API changes and\n        other issues to be more easily worked around. See the following links\n        for more information:\n\n        - `docker-py Low-level API`_\n        - `Docker Engine API`_\n\n        .. versionadded:: 2018.3.0\n\n    .. _`docker-py Low-level API`: http://docker-py.readthedocs.io/en/stable/api.html#docker.api.container.ContainerApiMixin.create_container\n    .. _`Docker Engine API`: https://docs.docker.com/engine/api/v1.33/#operation/ContainerCreate\n\n    ignore_collisions : False\n        Since many of docker-py's arguments differ in name from their CLI\n        counterparts (with which most Docker users are more familiar), Salt\n        detects usage of these and aliases them to the docker-py version of\n        that argument. However, if both the alias and the docker-py version of\n        the same argument (e.g. ``options`` and ``driver_opts``) are used, an error\n        will be raised. Set this argument to ``True`` to suppress these errors\n        and keep the docker-py version of the argument.\n\n        .. versionadded:: 2018.3.0\n\n    validate_ip_addrs : True\n        For parameters which accept IP addresses/subnets as input, validation\n        will be performed. To disable, set this to ``False``.\n\n        .. versionadded:: 2018.3.0\n\n    containers\n        A list of containers which should be connected to this network.\n\n        .. note::\n            As of the 2018.3.0 release, this is not the recommended way of\n            managing a container's membership in a network, for a couple\n            reasons:\n\n            1. It does not support setting static IPs, aliases, or links in the\n               container's IP configuration.\n            2. If a :py:func:`docker_container.running\n               <salt.states.docker_container.running>` state replaces a\n               container, it will not be reconnected to the network until the\n               ``docker_network.present`` state is run again. Since containers\n               often have ``require`` requisites to ensure that the network\n               is present, this means that the ``docker_network.present`` state\n               ends up being run *before* the :py:func:`docker_container.running\n               <salt.states.docker_container.running>`, leaving the container\n               unattached at the end of the Salt run.\n\n            For these reasons, it is recommended to use\n            :ref:`docker_container.running's network management support\n            <salt-states-docker-container-network-management>`.\n\n    reconnect : True\n        If ``containers`` is not used, and the network is replaced, then Salt\n        will keep track of the containers which were connected to the network\n        and reconnect them to the network after it is replaced. Salt will first\n        attempt to reconnect using the same IP the container had before the\n        network was replaced. If that fails (for instance, if the network was\n        replaced because the subnet was modified), then the container will be\n        reconnected without an explicit IP address, and its IP will be assigned\n        by Docker.\n\n        Set this option to ``False`` to keep Salt from trying to reconnect\n        containers. This can be useful in some cases when :ref:`managing static\n        IPs in docker_container.running\n        <salt-states-docker-container-network-management>`. For instance, if a\n        network's subnet is modified, it is likely that the static IP will need\n        to be updated in the ``docker_container.running`` state as well. When\n        the network is replaced, the initial reconnect attempt would fail, and\n        the container would be reconnected with an automatically-assigned IP\n        address. Then, when the ``docker_container.running`` state executes, it\n        would disconnect the network *again* and reconnect using the new static\n        IP. Disabling the reconnect behavior in these cases would prevent the\n        unnecessary extra reconnection.\n\n        .. versionadded:: 2018.3.0\n\n    .. _salt-states-docker-network-present-netconf:\n\n    **NETWORK CONFIGURATION ARGUMENTS**\n\n    driver\n        Network driver\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - driver: macvlan\n\n    driver_opts (or *driver_opt*, or *options*)\n        Options for the network driver. Either a dictionary of option names and\n        values or a Python list of strings in the format ``varname=value``. The\n        below three examples are equivalent:\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - driver: macvlan\n                - driver_opts: macvlan_mode=bridge,parent=eth0\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - driver: macvlan\n                - driver_opts:\n                  - macvlan_mode=bridge\n                  - parent=eth0\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - driver: macvlan\n                - driver_opts:\n                  - macvlan_mode: bridge\n                  - parent: eth0\n\n        The options can also simply be passed as a dictionary, though this can\n        be error-prone due to some :ref:`idiosyncrasies <yaml-idiosyncrasies>`\n        with how PyYAML loads nested data structures:\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - driver: macvlan\n                - driver_opts:\n                    macvlan_mode: bridge\n                    parent: eth0\n\n    check_duplicate : True\n        If ``True``, checks for networks with duplicate names. Since networks\n        are primarily keyed based on a random ID and not on the name, and\n        network name is strictly a user-friendly alias to the network which is\n        uniquely identified using ID, there is no guaranteed way to check for\n        duplicates. This option providess a best effort, checking for any\n        networks which have the same name, but it is not guaranteed to catch\n        all name collisions.\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - check_duplicate: False\n\n    internal : False\n        If ``True``, restricts external access to the network\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - internal: True\n\n    labels\n        Add metadata to the network. Labels can be set both with and without\n        values, and labels with values can be passed either as ``key=value`` or\n        ``key: value`` pairs. For example, while the below would be very\n        confusing to read, it is technically valid, and demonstrates the\n        different ways in which labels can be passed:\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - labels:\n                  - foo\n                  - bar=baz\n                  - hello: world\n\n        The labels can also simply be passed as a YAML dictionary, though this\n        can be error-prone due to some :ref:`idiosyncrasies\n        <yaml-idiosyncrasies>` with how PyYAML loads nested data structures:\n\n        .. code-block:: yaml\n\n            foo:\n              docker_network.present:\n                - labels:\n                    foo: ''\n                    bar: baz\n                    hello: world\n\n        .. versionchanged:: 2018.3.0\n            Methods for specifying labels can now be mixed. Earlier releases\n            required either labels with or without values.\n\n    enable_ipv6 (or *ipv6*) : False\n        Enable IPv6 on the network\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - enable_ipv6: True\n\n        .. note::\n            While it should go without saying, this argument must be set to\n            ``True`` to :ref:`configure an IPv6 subnet\n            <salt-states-docker-network-present-ipam>`. Also, if this option is\n            turned on without an IPv6 subnet explicitly configured, you will\n            get an error unless you have set up a fixed IPv6 subnet. Consult\n            the `Docker IPv6 docs`_ for information on how to do this.\n\n            .. _`Docker IPv6 docs`: https://docs.docker.com/v17.09/engine/userguide/networking/default_network/ipv6/\n\n    attachable : False\n        If ``True``, and the network is in the global scope, non-service\n        containers on worker nodes will be able to connect to the network.\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - attachable: True\n\n        .. note::\n            This option cannot be reliably managed on CentOS 7. This is because\n            while support for this option was added in API version 1.24, its\n            value was not added to the inpsect results until API version 1.26.\n            The version of Docker which is available for CentOS 7 runs API\n            version 1.24, meaning that while Salt can pass this argument to the\n            API, it has no way of knowing the value of this config option in an\n            existing Docker network.\n\n    scope\n        Specify the network's scope (``local``, ``global`` or ``swarm``)\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - scope: local\n\n    ingress : False\n        If ``True``, create an ingress network which provides the routing-mesh in\n        swarm mode\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - ingress: True\n\n    .. _salt-states-docker-network-present-ipam:\n\n    **IP ADDRESS MANAGEMENT (IPAM)**\n\n    This state supports networks with either IPv4, or both IPv4 and IPv6. If\n    configuring IPv4, then you can pass the :ref:`IPAM pool arguments\n    <salt-states-docker-network-present-ipam-pool-arguments>` below as\n    individual arguments. However, if configuring IPv4 and IPv6, the arguments\n    must be passed as a list of dictionaries, in the ``ipam_pools`` argument\n    (click :ref:`here <salt-states-docker-network-present-ipam-examples>` for\n    some examples). `These docs`_ also have more information on these\n    arguments.\n\n    .. _`These docs`: http://docker-py.readthedocs.io/en/stable/api.html#docker.types.IPAMPool\n\n    *IPAM ARGUMENTS*\n\n    ipam_driver\n        IPAM driver to use, if different from the default one\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - ipam_driver: foo\n\n    ipam_opts\n        Options for the IPAM driver. Either a dictionary of option names and\n        values or a Python list of strings in the format ``varname=value``. The\n        below three examples are equivalent:\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - ipam_driver: foo\n                - ipam_opts: foo=bar,baz=qux\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - ipam_driver: foo\n                - ipam_opts:\n                  - foo=bar\n                  - baz=qux\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - ipam_driver: foo\n                - ipam_opts:\n                  - foo: bar\n                  - baz: qux\n\n        The options can also simply be passed as a dictionary, though this can\n        be error-prone due to some :ref:`idiosyncrasies <yaml-idiosyncrasies>`\n        with how PyYAML loads nested data structures:\n\n        .. code-block:: yaml\n\n            mynet:\n              docker_network.present:\n                - ipam_driver: macvlan\n                - ipam_opts:\n                    foo: bar\n                    baz: qux\n\n    .. _salt-states-docker-network-present-ipam-pool-arguments:\n\n    *IPAM POOL ARGUMENTS*\n\n    subnet\n        Subnet in CIDR format that represents a network segment\n\n    iprange (or *ip_range*)\n        Allocate container IP from a sub-range within the subnet\n\n        Subnet in CIDR format that represents a network segment\n\n    gateway\n        IPv4 or IPv6 gateway for the master subnet\n\n    aux_addresses (or *aux_address*)\n        A dictionary of mapping container names to IP addresses which should be\n        allocated for them should they connect to the network. Either a\n        dictionary of option names and values or a Python list of strings in\n        the format ``host=ipaddr``.\n\n    .. _salt-states-docker-network-present-ipam-examples:\n\n    *IPAM CONFIGURATION EXAMPLES*\n\n    Below is an example of an IPv4-only network (keep in mind that ``subnet``\n    is the only required argument).\n\n    .. code-block:: yaml\n\n        mynet:\n          docker_network.present:\n            - subnet: 10.0.20.0/24\n            - iprange: 10.0.20.128/25\n            - gateway: 10.0.20.254\n            - aux_addresses:\n              - foo.bar.tld: 10.0.20.50\n              - hello.world.tld: 10.0.20.51\n\n    .. note::\n        The ``aux_addresses`` can be passed differently, in the same way that\n        ``driver_opts`` and ``ipam_opts`` can.\n\n    This same network could also be configured this way:\n\n    .. code-block:: yaml\n\n        mynet:\n          docker_network.present:\n            - ipam_pools:\n              - subnet: 10.0.20.0/24\n                iprange: 10.0.20.128/25\n                gateway: 10.0.20.254\n                aux_addresses:\n                  foo.bar.tld: 10.0.20.50\n                  hello.world.tld: 10.0.20.51\n\n    Here is an example of a mixed IPv4/IPv6 subnet.\n\n    .. code-block:: yaml\n\n        mynet:\n          docker_network.present:\n            - ipam_pools:\n              - subnet: 10.0.20.0/24\n                gateway: 10.0.20.1\n              - subnet: fe3f:2180:26:1::/123\n                gateway: fe3f:2180:26:1::1\n    "
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    try:
        network = __salt__['docker.inspect_network'](name)
    except CommandExecutionError as exc:
        msg = exc.__str__()
        if '404' in msg:
            network = None
        else:
            ret['comment'] = msg
            return ret
    to_connect = {}
    missing_containers = []
    stopped_containers = []
    for cname in __utils__['args.split_input'](containers or []):
        try:
            cinfo = __salt__['docker.inspect_container'](cname)
        except CommandExecutionError:
            missing_containers.append(cname)
        else:
            try:
                cid = cinfo['Id']
            except KeyError:
                missing_containers.append(cname)
            else:
                if not cinfo.get('State', {}).get('Running', False):
                    stopped_containers.append(cname)
                else:
                    to_connect[cid] = {'Name': cname}
    if missing_containers:
        ret.setdefault('warnings', []).append('The following containers do not exist: {}.'.format(', '.join(missing_containers)))
    if stopped_containers:
        ret.setdefault('warnings', []).append('The following containers are not running: {}.'.format(', '.join(stopped_containers)))
    disconnected_containers = {}
    try:
        kwargs = __utils__['docker.translate_input'](salt.utils.dockermod.translate.network, skip_translate=skip_translate, ignore_collisions=ignore_collisions, validate_ip_addrs=validate_ip_addrs, **__utils__['args.clean_kwargs'](**kwargs))
    except Exception as exc:
        ret['comment'] = exc.__str__()
        return ret
    ipam_kwargs = {}
    ipam_kwarg_names = ['ipam', 'ipam_driver', 'ipam_opts', 'ipam_pools']
    ipam_kwarg_names.extend(__salt__['docker.get_client_args']('ipam_config')['ipam_config'])
    for key in ipam_kwarg_names:
        try:
            ipam_kwargs[key] = kwargs.pop(key)
        except KeyError:
            pass
    if 'ipam' in ipam_kwargs:
        if len(ipam_kwargs) > 1:
            ret['comment'] = "Cannot mix the 'ipam' argument with any of the IPAM config arguments. See documentation for details."
            return ret
        ipam_config = ipam_kwargs['ipam']
    else:
        ipam_pools = ipam_kwargs.pop('ipam_pools', ())
        try:
            ipam_config = __utils__['docker.create_ipam_config'](*ipam_pools, **ipam_kwargs)
        except Exception as exc:
            ret['comment'] = exc.__str__()
            return ret
    create_network = True
    if network is not None:
        log.debug("Docker network '%s' already exists", name)
        ret['comment'] = f"Network '{name}' already exists, and is configured as specified"
        log.trace("Details of docker network '%s': %s", name, network)
        temp_net_name = ''.join((random.choice(string.ascii_lowercase) for _ in range(20)))
        try:
            enable_ipv6 = kwargs.pop('enable_ipv6', None)
            kwargs_tmp = kwargs
            driver = kwargs.get('driver')
            driver_opts = kwargs.get('options', {})
            bridge_name = driver_opts.get('com.docker.network.bridge.name', None)
            if driver == 'bridge' and bridge_name is not None:
                tmp_name = str(bridge_name) + 'comp'
                kwargs_tmp['options']['com.docker.network.bridge.name'] = tmp_name[-14:]
            __salt__['docker.create_network'](temp_net_name, skip_translate=True, enable_ipv6=False, **kwargs_tmp)
        except CommandExecutionError as exc:
            ret['comment'] = 'Failed to create temp network for comparison: {}'.format(exc.__str__())
            return ret
        else:
            if enable_ipv6 is not None:
                kwargs['enable_ipv6'] = enable_ipv6
        try:
            try:
                temp_net_info = __salt__['docker.inspect_network'](temp_net_name)
            except CommandExecutionError as exc:
                ret['comment'] = 'Failed to inspect temp network: {}'.format(exc.__str__())
                return ret
            else:
                temp_net_info['EnableIPv6'] = bool(enable_ipv6)
            temp_net_info['IPAM'] = ipam_config
            existing_pool_count = len(network['IPAM']['Config'])
            desired_pool_count = len(temp_net_info['IPAM']['Config'])
            is_default_pool = lambda x: True if sorted(x) == ['Gateway', 'Subnet'] else False
            if desired_pool_count == 0 and existing_pool_count == 1 and is_default_pool(network['IPAM']['Config'][0]):
                network['IPAM']['Config'] = []
            changes = __salt__['docker.compare_networks'](network, temp_net_info, ignore='Name,Id,Created,Containers')
            if not changes:
                create_network = False
            else:
                ret['changes'][name] = changes
                if __opts__['test']:
                    ret['result'] = None
                    ret['comment'] = 'Network would be recreated with new config'
                    return ret
                if network['Containers']:
                    disconnected_containers = copy.deepcopy(network['Containers'])
                    if not containers and reconnect:
                        for cid in disconnected_containers:
                            try:
                                cinfo = __salt__['docker.inspect_container'](cid)
                                netinfo = cinfo['NetworkSettings']['Networks'][name]
                                net_links = netinfo.get('Links') or []
                                net_aliases = netinfo.get('Aliases') or []
                                if net_links:
                                    disconnected_containers[cid]['Links'] = net_links
                                if net_aliases:
                                    disconnected_containers[cid]['Aliases'] = net_aliases
                            except (CommandExecutionError, KeyError, ValueError):
                                continue
                remove_result = _remove_network(network)
                if not remove_result['result']:
                    return remove_result
                network['Containers'] = {}
        finally:
            try:
                __salt__['docker.remove_network'](temp_net_name)
            except CommandExecutionError as exc:
                ret.setdefault('warnings', []).append("Failed to remove temp network '{}': {}.".format(temp_net_name, exc.__str__()))
    if create_network:
        log.debug("Network '%s' will be created", name)
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Network will be created'
            return ret
        kwargs['ipam'] = ipam_config
        try:
            __salt__['docker.create_network'](name, skip_translate=True, **kwargs)
        except Exception as exc:
            ret['comment'] = "Failed to create network '{}': {}".format(name, exc.__str__())
            return ret
        else:
            action = 'recreated' if network is not None else 'created'
            ret['changes'][action] = True
            ret['comment'] = "Network '{}' {}".format(name, 'created' if network is None else 'was replaced with updated config')
            network = {'Containers': {}}
    if containers is None and reconnect and disconnected_containers:
        to_connect = disconnected_containers
    for cid in list(to_connect):
        if cid in network['Containers']:
            del to_connect[cid]
    errors = []
    if to_connect:
        for (cid, connect_info) in to_connect.items():
            connect_kwargs = {}
            if cid in disconnected_containers:
                for (key_name, arg_name) in (('IPv4Address', 'ipv4_address'), ('IPV6Address', 'ipv6_address'), ('Links', 'links'), ('Aliases', 'aliases')):
                    try:
                        connect_kwargs[arg_name] = connect_info[key_name]
                    except (KeyError, AttributeError):
                        continue
                    else:
                        if key_name.endswith('Address'):
                            connect_kwargs[arg_name] = connect_kwargs[arg_name].rsplit('/', 1)[0]
            try:
                __salt__['docker.connect_container_to_network'](cid, name, **connect_kwargs)
            except CommandExecutionError as exc:
                if not connect_kwargs:
                    errors.append(exc.__str__())
                else:
                    try:
                        __salt__['docker.connect_container_to_network'](cid, name)
                    except CommandExecutionError as exc:
                        errors.append(exc.__str__())
                    else:
                        ret['changes'].setdefault('reconnected' if cid in disconnected_containers else 'connected', []).append(connect_info['Name'])
            else:
                ret['changes'].setdefault('reconnected' if cid in disconnected_containers else 'connected', []).append(connect_info['Name'])
    if errors:
        if ret['comment']:
            ret['comment'] += '. '
        ret['comment'] += '. '.join(errors) + '.'
    else:
        ret['result'] = True
    for (cid, c_info) in disconnected_containers.items():
        if cid not in to_connect:
            ret['changes'].setdefault('disconnected', []).append(c_info['Name'])
    return ret

def absent(name):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that a network is absent.\n\n    name\n        Name of the network\n\n    Usage Example:\n\n    .. code-block:: yaml\n\n        network_foo:\n          docker_network.absent\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    try:
        network = __salt__['docker.inspect_network'](name)
    except CommandExecutionError as exc:
        msg = exc.__str__()
        if '404' in msg:
            network = None
        else:
            ret['comment'] = msg
            return ret
    if network is None:
        ret['result'] = True
        ret['comment'] = f"Network '{name}' already absent"
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = f"Network '{name}' will be removed"
        return ret
    return _remove_network(network)

def _remove_network(network):
    if False:
        while True:
            i = 10
    '\n    Remove network, including all connected containers\n    '
    ret = {'name': network['Name'], 'changes': {}, 'result': False, 'comment': ''}
    errors = []
    for cid in network['Containers']:
        try:
            cinfo = __salt__['docker.inspect_container'](cid)
        except CommandExecutionError:
            cname = cid
        else:
            cname = cinfo.get('Name', '').lstrip('/')
        try:
            __salt__['docker.disconnect_container_from_network'](cid, network['Name'])
        except CommandExecutionError as exc:
            errors = f"Failed to disconnect container '{cname}' : {exc}"
        else:
            ret['changes'].setdefault('disconnected', []).append(cname)
    if errors:
        ret['comment'] = '\n'.join(errors)
        return ret
    try:
        __salt__['docker.remove_network'](network['Name'])
    except CommandExecutionError as exc:
        ret['comment'] = f'Failed to remove network: {exc}'
    else:
        ret['changes']['removed'] = True
        ret['result'] = True
        ret['comment'] = "Removed network '{}'".format(network['Name'])
    return ret