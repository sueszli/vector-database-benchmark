"""
NAPALM helpers
==============

Helpers for the NAPALM modules.

.. versionadded:: 2017.7.0
"""
import logging
import salt.utils.napalm
from salt.exceptions import CommandExecutionError
from salt.utils.decorators import depends
from salt.utils.napalm import proxy_napalm_wrap
try:
    from netmiko import BaseConnection
    HAS_NETMIKO = True
except ImportError:
    HAS_NETMIKO = False
try:
    import napalm.base.netmiko_helpers
    HAS_NETMIKO_HELPERS = True
except ImportError:
    HAS_NETMIKO_HELPERS = False
try:
    import jxmlease
    HAS_JXMLEASE = True
except ImportError:
    HAS_JXMLEASE = False
try:
    import ciscoconfparse
    HAS_CISCOCONFPARSE = True
except ImportError:
    HAS_CISCOCONFPARSE = False
try:
    import scp
    HAS_SCP = True
except ImportError:
    HAS_SCP = False
__virtualname__ = 'napalm'
__proxyenabled__ = ['*']
log = logging.getLogger(__file__)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def _get_netmiko_args(optional_args):
    if False:
        i = 10
        return i + 15
    '\n    Check for Netmiko arguments that were passed in as NAPALM optional arguments.\n\n    Return a dictionary of these optional args that will be passed into the\n    Netmiko ConnectHandler call.\n\n    .. note::\n\n        This is a port of the NAPALM helper for backwards compatibility with\n        older versions of NAPALM, and stability across Salt features.\n        If the netmiko helpers module is available however, it will prefer that\n        implementation nevertheless.\n    '
    if HAS_NETMIKO_HELPERS:
        return napalm.base.netmiko_helpers.netmiko_args(optional_args)
    (netmiko_args, _, _, netmiko_defaults) = __utils__['args.get_function_argspec'](BaseConnection.__init__)
    check_self = netmiko_args.pop(0)
    if check_self != 'self':
        raise ValueError('Error processing Netmiko arguments')
    netmiko_argument_map = dict(zip(netmiko_args, netmiko_defaults))
    netmiko_filter = ['ip', 'host', 'username', 'password', 'device_type', 'timeout']
    for k in netmiko_filter:
        netmiko_argument_map.pop(k)
    netmiko_optional_args = {}
    for (k, v) in netmiko_argument_map.items():
        try:
            netmiko_optional_args[k] = optional_args[k]
        except KeyError:
            pass
    return netmiko_optional_args

def _inject_junos_proxy(napalm_device):
    if False:
        for i in range(10):
            print('nop')
    '\n    Inject the junos.conn key into the __proxy__, reusing the existing NAPALM\n    connection to the Junos device.\n    '

    def _ret_device():
        if False:
            for i in range(10):
                print('nop')
        return napalm_device['DRIVER'].device
    __proxy__['junos.conn'] = _ret_device

def _junos_prep_fun(napalm_device):
    if False:
        i = 10
        return i + 15
    '\n    Prepare the Junos function.\n    '
    if __grains__['os'] != 'junos':
        return {'out': None, 'result': False, 'comment': 'This function is only available on Junos'}
    if not HAS_JXMLEASE:
        return {'out': None, 'result': False, 'comment': 'Please install jxmlease (``pip install jxmlease``) to be able to use this function.'}
    _inject_junos_proxy(napalm_device)
    return {'result': True}

@proxy_napalm_wrap
def _netmiko_conn(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Return the connection object with the network device, over Netmiko, passing\n    the authentication details from the existing NAPALM connection.\n\n    .. warning::\n\n        This function is not suitable for CLI usage, more rather to be used\n        in various Salt modules.\n\n    USAGE Example:\n\n    .. code-block:: python\n\n        conn = __salt__['napalm.netmiko_conn']()\n        res = conn.send_command('show interfaces')\n        conn.disconnect()\n    "
    netmiko_kwargs = netmiko_args()
    kwargs.update(netmiko_kwargs)
    return __salt__['netmiko.get_connection'](**kwargs)

@proxy_napalm_wrap
def _pyeapi_conn(**kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Return the connection object with the Arista switch, over ``pyeapi``,\n    passing the authentication details from the existing NAPALM connection.\n\n    .. warning::\n        This function is not suitable for CLI usage, more rather to be used in\n        various Salt modules, to reusing the established connection, as in\n        opposite to opening a new connection for each task.\n\n    Usage example:\n\n    .. code-block:: python\n\n        conn = __salt__['napalm.pyeapi_conn']()\n        res1 = conn.run_commands('show version')\n        res2 = conn.get_config(as_string=True)\n    "
    pyeapi_kwargs = pyeapi_nxos_api_args(**kwargs)
    return __salt__['pyeapi.get_connection'](**pyeapi_kwargs)

@proxy_napalm_wrap
def alive(**kwargs):
    if False:
        return 10
    "\n    Returns the alive status of the connection layer.\n    The output is a dictionary under the usual dictionary\n    output of the NAPALM modules.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.alive\n\n    Output Example:\n\n    .. code-block:: yaml\n\n        result: True\n        out:\n            is_alive: False\n        comment: ''\n    "
    return salt.utils.napalm.call(napalm_device, 'is_alive', **{})

@proxy_napalm_wrap
def reconnect(force=False, **kwargs):
    if False:
        return 10
    "\n    Reconnect the NAPALM proxy when the connection\n    is dropped by the network device.\n    The connection can be forced to be restarted\n    using the ``force`` argument.\n\n    .. note::\n\n        This function can be used only when running proxy minions.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.reconnect\n        salt '*' napalm.reconnect force=True\n    "
    default_ret = {'out': None, 'result': True, 'comment': 'Already alive.'}
    if not salt.utils.napalm.is_proxy(__opts__):
        return default_ret
    is_alive = alive()
    log.debug('Is alive fetch:')
    log.debug(is_alive)
    if not is_alive.get('result', False) or not is_alive.get('out', False) or (not is_alive.get('out', {}).get('is_alive', False)) or force:
        proxyid = __opts__.get('proxyid') or __opts__.get('id')
        log.info('Closing the NAPALM proxy connection with %s', proxyid)
        salt.utils.napalm.call(napalm_device, 'close', **{})
        log.info('Re-opening the NAPALM proxy connection with %s', proxyid)
        salt.utils.napalm.call(napalm_device, 'open', **{})
        default_ret.update({'comment': 'Connection restarted!'})
        return default_ret
    return default_ret

@proxy_napalm_wrap
def call(method, *args, **kwargs):
    if False:
        return 10
    "\n    Execute arbitrary methods from the NAPALM library.\n    To see the expected output, please consult the NAPALM documentation.\n\n    .. note::\n\n        This feature is not recommended to be used in production.\n        It should be used for testing only!\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.call get_lldp_neighbors\n        salt '*' napalm.call get_firewall_policies\n        salt '*' napalm.call get_bgp_config group='my-group'\n    "
    clean_kwargs = {}
    for (karg, warg) in kwargs.items():
        if not karg.startswith('__pub_'):
            clean_kwargs[karg] = warg
    return salt.utils.napalm.call(napalm_device, method, *args, **clean_kwargs)

@proxy_napalm_wrap
def compliance_report(filepath=None, string=None, renderer='jinja|yaml', **kwargs):
    if False:
        print('Hello World!')
    '\n    Return the compliance report.\n\n    filepath\n        The absolute path to the validation file.\n\n        .. versionchanged:: 2019.2.0\n\n        Beginning with release codename ``2019.2.0``, this function has been\n        enhanced, to be able to leverage the multi-engine template rendering\n        of Salt, besides the possibility to retrieve the file source from\n        remote systems, the URL schemes supported being:\n\n        - ``salt://``\n        - ``http://`` and ``https://``\n        - ``ftp://``\n        - ``s3://``\n        - ``swift:/``\n\n        Or on the local file system (on the Minion).\n\n        .. note::\n\n            The rendering result does not necessarily need to be YAML, instead\n            it can be any format interpreted by Salt\'s rendering pipeline\n            (including pure Python).\n\n    string\n        .. versionadded:: 2019.2.0\n\n        The compliance report send as inline string, to be used as the file to\n        send through the renderer system. Note, not all renderer modules can\n        work with strings; the \'py\' renderer requires a file, for example.\n\n    renderer: ``jinja|yaml``\n        .. versionadded:: 2019.2.0\n\n        The renderer pipe to send the file through; this is overridden by a\n        "she-bang" at the top of the file.\n\n    kwargs\n        .. versionchanged:: 2019.2.0\n\n        Keyword args to pass to Salt\'s compile_template() function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm.compliance_report ~/validate.yml\n        salt \'*\' napalm.compliance_report salt://path/to/validator.sls\n\n    Validation File Example (pure YAML):\n\n    .. code-block:: yaml\n\n        - get_facts:\n            os_version: 4.17\n\n        - get_interfaces_ip:\n            Management1:\n              ipv4:\n                10.0.2.14:\n                  prefix_length: 24\n                _mode: strict\n\n    Validation File Example (as Jinja + YAML):\n\n    .. code-block:: yaml\n\n        - get_facts:\n            os_version: {{ grains.version }}\n        - get_interfaces_ip:\n            Loopback0:\n              ipv4:\n                {{ grains.lo0.ipv4 }}:\n                  prefix_length: 24\n                _mode: strict\n        - get_bgp_neighbors: {{ pillar.bgp.neighbors }}\n\n    Output Example:\n\n    .. code-block:: yaml\n\n        device1:\n            ----------\n            comment:\n            out:\n                ----------\n                complies:\n                    False\n                get_facts:\n                    ----------\n                    complies:\n                        False\n                    extra:\n                    missing:\n                    present:\n                        ----------\n                        os_version:\n                            ----------\n                            actual_value:\n                                15.1F6-S1.4\n                            complies:\n                                False\n                            nested:\n                                False\n                get_interfaces_ip:\n                    ----------\n                    complies:\n                        False\n                    extra:\n                    missing:\n                        - Management1\n                    present:\n                        ----------\n                skipped:\n            result:\n                True\n    '
    validation_string = __salt__['slsutil.renderer'](path=filepath, string=string, default_renderer=renderer, **kwargs)
    return salt.utils.napalm.call(napalm_device, 'compliance_report', validation_source=validation_string)

@proxy_napalm_wrap
def netmiko_args(**kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2019.2.0\n\n    Return the key-value arguments used for the authentication arguments for\n    the netmiko module.\n\n    When running in a non-native NAPALM driver (e.g., ``panos``, `f5``, ``mos`` -\n    either from https://github.com/napalm-automation-community or defined in\n    user's own environment, one can specify the Netmiko device type (the\n    ``device_type`` argument) via the ``netmiko_device_type_map`` configuration\n    option / Pillar key, e.g.,\n\n    .. code-block:: yaml\n\n        netmiko_device_type_map:\n          f5: f5_ltm\n          dellos10: dell_os10\n\n    The configuration above defines the mapping between the NAPALM ``os`` Grain\n    and the Netmiko ``device_type``, e.g., when the NAPALM Grain is ``f5``, it\n    would use the ``f5_ltm`` SSH Netmiko driver to execute commands over SSH on\n    the remote network device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.netmiko_args\n    "
    if not HAS_NETMIKO:
        raise CommandExecutionError('Please install netmiko to be able to use this feature.')
    kwargs = {}
    napalm_opts = salt.utils.napalm.get_device_opts(__opts__, salt_obj=__salt__)
    optional_args = napalm_opts['OPTIONAL_ARGS']
    netmiko_args = _get_netmiko_args(optional_args)
    kwargs['host'] = napalm_opts['HOSTNAME']
    kwargs['username'] = napalm_opts['USERNAME']
    kwargs['password'] = napalm_opts['PASSWORD']
    kwargs['timeout'] = napalm_opts['TIMEOUT']
    kwargs.update(netmiko_args)
    netmiko_device_type_map = {'junos': 'juniper_junos', 'ios': 'cisco_ios', 'iosxr': 'cisco_xr', 'eos': 'arista_eos', 'nxos_ssh': 'cisco_nxos', 'asa': 'cisco_asa', 'fortios': 'fortinet', 'panos': 'paloalto_panos', 'aos': 'alcatel_aos', 'vyos': 'vyos', 'f5': 'f5_ltm', 'ce': 'huawei', 's350': 'cisco_s300'}
    netmiko_device_type_map.update(__salt__['config.get']('netmiko_device_type_map', {}))
    kwargs['device_type'] = netmiko_device_type_map[__grains__['os']]
    return kwargs

@proxy_napalm_wrap
def netmiko_fun(fun, *args, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2019.2.0\n\n    Call an arbitrary function from the :mod:`Netmiko<salt.modules.netmiko_mod>`\n    module, passing the authentication details from the existing NAPALM\n    connection.\n\n    fun\n        The name of the function from the :mod:`Netmiko<salt.modules.netmiko_mod>`\n        to invoke.\n\n    args\n        List of arguments to send to the execution function specified in\n        ``fun``.\n\n    kwargs\n        Key-value arguments to send to the execution function specified in\n        ``fun``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.netmiko_fun send_command 'show version'\n    "
    if 'netmiko.' not in fun:
        fun = f'netmiko.{fun}'
    netmiko_kwargs = netmiko_args()
    kwargs.update(netmiko_kwargs)
    return __salt__[fun](*args, **kwargs)

@proxy_napalm_wrap
def netmiko_call(method, *args, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Execute an arbitrary Netmiko method, passing the authentication details from\n    the existing NAPALM connection.\n\n    method\n        The name of the Netmiko method to execute.\n\n    args\n        List of arguments to send to the Netmiko method specified in ``method``.\n\n    kwargs\n        Key-value arguments to send to the execution function specified in\n        ``method``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.netmiko_call send_command 'show version'\n    "
    netmiko_kwargs = netmiko_args()
    kwargs.update(netmiko_kwargs)
    return __salt__['netmiko.call'](method, *args, **kwargs)

@proxy_napalm_wrap
def netmiko_multi_call(*methods, **kwargs):
    if False:
        return 10
    '\n    .. versionadded:: 2019.2.0\n\n    Execute a list of arbitrary Netmiko methods, passing the authentication\n    details from the existing NAPALM connection.\n\n    methods\n        List of dictionaries with the following keys:\n\n        - ``name``: the name of the Netmiko function to invoke.\n        - ``args``: list of arguments to send to the ``name`` method.\n        - ``kwargs``: key-value arguments to send to the ``name`` method.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm.netmiko_multi_call "{\'name\': \'send_command\', \'args\': [\'show version\']}" "{\'name\': \'send_command\', \'args\': [\'show interfaces\']}"\n    '
    netmiko_kwargs = netmiko_args()
    kwargs.update(netmiko_kwargs)
    return __salt__['netmiko.multi_call'](*methods, **kwargs)

@proxy_napalm_wrap
def netmiko_commands(*commands, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Invoke one or more commands to be executed on the remote device, via Netmiko.\n    Returns a list of strings, with the output from each command.\n\n    commands\n        A list of commands to be executed.\n\n    expect_string\n        Regular expression pattern to use for determining end of output.\n        If left blank will default to being based on router prompt.\n\n    delay_factor: ``1``\n        Multiplying factor used to adjust delays (default: ``1``).\n\n    max_loops: ``500``\n        Controls wait time in conjunction with delay_factor. Will default to be\n        based upon self.timeout.\n\n    auto_find_prompt: ``True``\n        Whether it should try to auto-detect the prompt (default: ``True``).\n\n    strip_prompt: ``True``\n        Remove the trailing router prompt from the output (default: ``True``).\n\n    strip_command: ``True``\n        Remove the echo of the command from the output (default: ``True``).\n\n    normalize: ``True``\n        Ensure the proper enter is sent at end of command (default: ``True``).\n\n    use_textfsm: ``False``\n        Process command output through TextFSM template (default: ``False``).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.netmiko_commands 'show version' 'show interfaces'\n    "
    conn = _netmiko_conn(**kwargs)
    ret = []
    for cmd in commands:
        ret.append(conn.send_command(cmd))
    return ret

@proxy_napalm_wrap
def netmiko_config(*config_commands, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Load a list of configuration commands on the remote device, via Netmiko.\n\n    .. warning::\n\n        Please remember that ``netmiko`` does not have any rollback safeguards\n        and any configuration change will be directly loaded into the running\n        config if the platform doesn't have the concept of ``candidate`` config.\n\n        On Junos, or other platforms that have this capability, the changes will\n        not be loaded into the running config, and the user must set the\n        ``commit`` argument to ``True`` to transfer the changes from the\n        candidate into the running config before exiting.\n\n    config_commands\n        A list of configuration commands to be loaded on the remote device.\n\n    config_file\n        Read the configuration commands from a file. The file can equally be a\n        template that can be rendered using the engine of choice (see\n        ``template_engine``).\n\n        This can be specified using the absolute path to the file, or using one\n        of the following URL schemes:\n\n        - ``salt://``, to fetch the file from the Salt fileserver.\n        - ``http://`` or ``https://``\n        - ``ftp://``\n        - ``s3://``\n        - ``swift://``\n\n    exit_config_mode: ``True``\n        Determines whether or not to exit config mode after complete.\n\n    delay_factor: ``1``\n        Factor to adjust delays.\n\n    max_loops: ``150``\n        Controls wait time in conjunction with delay_factor (default: ``150``).\n\n    strip_prompt: ``False``\n        Determines whether or not to strip the prompt (default: ``False``).\n\n    strip_command: ``False``\n        Determines whether or not to strip the command (default: ``False``).\n\n    config_mode_command\n        The command to enter into config mode.\n\n    commit: ``False``\n        Commit the configuration changes before exiting the config mode. This\n        option is by default disabled, as many platforms don't have this\n        capability natively.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.netmiko_config 'set system ntp peer 1.2.3.4' commit=True\n        salt '*' napalm.netmiko_config https://bit.ly/2sgljCB\n    "
    netmiko_kwargs = netmiko_args()
    kwargs.update(netmiko_kwargs)
    return __salt__['netmiko.send_config'](config_commands=config_commands, **kwargs)

@proxy_napalm_wrap
def junos_rpc(cmd=None, dest=None, format=None, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Execute an RPC request on the remote Junos device.\n\n    cmd\n        The RPC request to the executed. To determine the RPC request, you can\n        check the from the command line of the device, by executing the usual\n        command followed by ``| display xml rpc``, e.g.,\n        ``show lldp neighbors | display xml rpc``.\n\n    dest\n        Destination file where the RPC output is stored. Note that the file will\n        be stored on the Proxy Minion. To push the files to the Master, use\n        :mod:`cp.push <salt.modules.cp.push>` Execution function.\n\n    format: ``xml``\n        The format in which the RPC reply is received from the device.\n\n    dev_timeout: ``30``\n        The NETCONF RPC timeout.\n\n    filter\n        Used with the ``get-config`` RPC request to filter out the config tree.\n\n    terse: ``False``\n        Whether to return terse output.\n\n        .. note::\n\n            Some RPC requests may not support this argument.\n\n    interface_name\n        Name of the interface to query.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.junos_rpc get-lldp-neighbors-information\n        salt '*' napalm.junos_rpc get-config <configuration><system><ntp/></system></configuration>\n    "
    prep = _junos_prep_fun(napalm_device)
    if not prep['result']:
        return prep
    if not format:
        format = 'xml'
    rpc_ret = __salt__['junos.rpc'](cmd=cmd, dest=dest, format=format, **kwargs)
    rpc_ret['comment'] = rpc_ret.pop('message', '')
    rpc_ret['result'] = rpc_ret.pop('out', False)
    rpc_ret['out'] = rpc_ret.pop('rpc_reply', None)
    return rpc_ret

@proxy_napalm_wrap
def junos_commit(**kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Commit the changes loaded in the candidate configuration.\n\n    dev_timeout: ``30``\n        The NETCONF RPC timeout (in seconds).\n\n    comment\n      Provide a comment for the commit.\n\n    confirm\n      Provide time in minutes for commit confirmation. If this option is\n      specified, the commit will be rolled back in the specified amount of time\n      unless the commit is confirmed.\n\n    sync: ``False``\n      When ``True``, on dual control plane systems, requests that the candidate\n      configuration on one control plane be copied to the other control plane,\n      checked for correct syntax, and committed on both Routing Engines.\n\n    force_sync: ``False``\n      When ``True``, on dual control plane systems, force the candidate\n      configuration on one control plane to be copied to the other control\n      plane.\n\n    full\n      When ``True``, requires all the daemons to check and evaluate the new\n      configuration.\n\n    detail\n      When ``True``, return commit detail.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' napalm.junos_commit comment='Commitiing via Salt' detail=True\n        salt '*' napalm.junos_commit dev_timeout=60 confirm=10\n        salt '*' napalm.junos_commit sync=True dev_timeout=90\n    "
    prep = _junos_prep_fun(napalm_device)
    if not prep['result']:
        return prep
    return __salt__['junos.commit'](**kwargs)

@proxy_napalm_wrap
def junos_install_os(path=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2019.2.0\n\n    Installs the given image on the device.\n\n    path\n        The image file source. This argument supports the following URIs:\n\n        - Absolute path on the Minion.\n        - ``salt://`` to fetch from the Salt fileserver.\n        - ``http://`` and ``https://``\n        - ``ftp://``\n        - ``swift:/``\n        - ``s3://``\n\n    dev_timeout: ``30``\n        The NETCONF RPC timeout (in seconds)\n\n    reboot: ``False``\n        Whether to reboot the device after the installation is complete.\n\n    no_copy: ``False``\n        If ``True`` the software package will not be copied to the remote\n        device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.junos_install_os salt://images/junos_16_1.tgz reboot=True\n    "
    prep = _junos_prep_fun(napalm_device)
    if not prep['result']:
        return prep
    return __salt__['junos.install_os'](path=path, **kwargs)

@proxy_napalm_wrap
def junos_facts(**kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    The complete list of Junos facts collected by ``junos-eznc``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.junos_facts\n    "
    prep = _junos_prep_fun(napalm_device)
    if not prep['result']:
        return prep
    facts = dict(napalm_device['DRIVER'].device.facts)
    if 'version_info' in facts:
        facts['version_info'] = dict(facts['version_info'])
    if 'junos_info' in facts:
        for re in facts['junos_info']:
            facts['junos_info'][re]['object'] = dict(facts['junos_info'][re]['object'])
    return facts

@proxy_napalm_wrap
def junos_cli(command, format=None, dev_timeout=None, dest=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2019.2.0\n\n    Execute a CLI command and return the output in the specified format.\n\n    command\n        The command to execute on the Junos CLI.\n\n    format: ``text``\n        Format in which to get the CLI output (either ``text`` or ``xml``).\n\n    dev_timeout: ``30``\n        The NETCONF RPC timeout (in seconds).\n\n    dest\n        Destination file where the RPC output is stored. Note that the file will\n        be stored on the Proxy Minion. To push the files to the Master, use\n        :mod:`cp.push <salt.modules.cp.push>`.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.junos_cli 'show lldp neighbors'\n    "
    prep = _junos_prep_fun(napalm_device)
    if not prep['result']:
        return prep
    return __salt__['junos.cli'](command, format=format, dev_timeout=dev_timeout, dest=dest, **kwargs)

@proxy_napalm_wrap
def junos_copy_file(src, dst, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Copies the file on the remote Junos device.\n\n    src\n        The source file path. This argument accepts the usual Salt URIs (e.g.,\n        ``salt://``, ``http://``, ``https://``, ``s3://``, ``ftp://``, etc.).\n\n    dst\n        The destination path on the device where to copy the file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.junos_copy_file https://example.com/junos.cfg /var/tmp/myjunos.cfg\n    "
    prep = _junos_prep_fun(napalm_device)
    if not prep['result']:
        return prep
    cached_src = __salt__['cp.cache_file'](src)
    return __salt__['junos.file_copy'](cached_src, dst)

@proxy_napalm_wrap
def junos_call(fun, *args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Execute an arbitrary function from the\n    :mod:`junos execution module <salt.module.junos>`. To check what ``args``\n    and ``kwargs`` you must send to the function, please consult the appropriate\n    documentation.\n\n    fun\n        The name of the function. E.g., ``set_hostname``.\n\n    args\n        List of arguments to send to the ``junos`` function invoked.\n\n    kwargs\n        Dictionary of key-value arguments to send to the ``juno`` function\n        invoked.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.junos_fun cli 'show system commit'\n    "
    prep = _junos_prep_fun(napalm_device)
    if not prep['result']:
        return prep
    if 'junos.' not in fun:
        mod_fun = f'junos.{fun}'
    else:
        mod_fun = fun
    if mod_fun not in __salt__:
        return {'out': None, 'result': False, 'comment': f'{fun} is not a valid function'}
    return __salt__[mod_fun](*args, **kwargs)

def pyeapi_nxos_api_args(**prev_kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Return the key-value arguments used for the authentication arguments for the\n    :mod:`pyeapi execution module <salt.module.arista_pyeapi>`.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.pyeapi_nxos_api_args\n    "
    kwargs = {}
    napalm_opts = salt.utils.napalm.get_device_opts(__opts__, salt_obj=__salt__)
    optional_args = napalm_opts['OPTIONAL_ARGS']
    kwargs['host'] = napalm_opts['HOSTNAME']
    kwargs['username'] = napalm_opts['USERNAME']
    kwargs['password'] = napalm_opts['PASSWORD']
    kwargs['timeout'] = napalm_opts['TIMEOUT']
    if 'transport' in optional_args and optional_args['transport']:
        kwargs['transport'] = optional_args['transport']
    else:
        kwargs['transport'] = 'https'
    if 'port' in optional_args and optional_args['port']:
        kwargs['port'] = optional_args['port']
    else:
        kwargs['port'] = 80 if kwargs['transport'] == 'http' else 443
    kwargs['verify'] = optional_args.get('verify')
    prev_kwargs.update(kwargs)
    return prev_kwargs

@proxy_napalm_wrap
def pyeapi_run_commands(*commands, **kwargs):
    if False:
        return 10
    "\n    Execute a list of commands on the Arista switch, via the ``pyeapi`` library.\n    This function forwards the existing connection details to the\n    :mod:`pyeapi.run_commands <salt.module.arista_pyeapi.run_commands>`\n    execution function.\n\n    commands\n        A list of commands to execute.\n\n    encoding: ``json``\n        The requested encoding of the command output. Valid values for encoding\n        are ``json`` (default) or ``text``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.pyeapi_run_commands 'show version' encoding=text\n        salt '*' napalm.pyeapi_run_commands 'show ip bgp neighbors'\n    "
    pyeapi_kwargs = pyeapi_nxos_api_args(**kwargs)
    return __salt__['pyeapi.run_commands'](*commands, **pyeapi_kwargs)

@proxy_napalm_wrap
def pyeapi_call(method, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Invoke an arbitrary method from the ``pyeapi`` library.\n    This function forwards the existing connection details to the\n    :mod:`pyeapi.run_commands <salt.module.arista_pyeapi.run_commands>`\n    execution function.\n\n    method\n        The name of the ``pyeapi`` method to invoke.\n\n    kwargs\n        Key-value arguments to send to the ``pyeapi`` method.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.pyeapi_call run_commands 'show version' encoding=text\n        salt '*' napalm.pyeapi_call get_config as_string=True\n    "
    pyeapi_kwargs = pyeapi_nxos_api_args(**kwargs)
    return __salt__['pyeapi.call'](method, *args, **pyeapi_kwargs)

@proxy_napalm_wrap
def pyeapi_config(commands=None, config_file=None, template_engine='jinja', context=None, defaults=None, saltenv='base', **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Configures the Arista switch with the specified commands, via the ``pyeapi``\n    library. This function forwards the existing connection details to the\n    :mod:`pyeapi.run_commands <salt.module.arista_pyeapi.run_commands>`\n    execution function.\n\n    commands\n        The list of configuration commands to load on the Arista switch.\n\n        .. note::\n            This argument is ignored when ``config_file`` is specified.\n\n    config_file\n        The source file with the configuration commands to be sent to the device.\n\n        The file can also be a template that can be rendered using the template\n        engine of choice. This can be specified using the absolute path to the\n        file, or using one of the following URL schemes:\n\n        - ``salt://``\n        - ``https://``\n        - ``ftp:/``\n        - ``s3:/``\n        - ``swift://``\n\n    template_engine: ``jinja``\n        The template engine to use when rendering the source file. Default:\n        ``jinja``. To simply fetch the file without attempting to render, set\n        this argument to ``None``.\n\n    context: ``None``\n        Variables to add to the template context.\n\n    defaults: ``None``\n        Default values of the ``context`` dict.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. Ignored if\n        ``config_file`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.pyeapi_config 'ntp server 1.2.3.4'\n    "
    pyeapi_kwargs = pyeapi_nxos_api_args(**kwargs)
    return __salt__['pyeapi.config'](commands=commands, config_file=config_file, template_engine=template_engine, context=context, defaults=defaults, saltenv=saltenv, **pyeapi_kwargs)

@proxy_napalm_wrap
def nxos_api_rpc(commands, method='cli', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Execute an arbitrary RPC request via the Nexus API.\n\n    commands\n        The RPC commands to be executed.\n\n    method: ``cli``\n        The type of the response, i.e., raw text (``cli_ascii``) or structured\n        document (``cli``). Defaults to ``cli`` (structured data).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.nxos_api_rpc 'show version'\n    "
    nxos_api_kwargs = pyeapi_nxos_api_args(**kwargs)
    return __salt__['nxos_api.rpc'](commands, method=method, **nxos_api_kwargs)

@proxy_napalm_wrap
def nxos_api_config(commands=None, config_file=None, template_engine='jinja', context=None, defaults=None, saltenv='base', **kwargs):
    if False:
        print('Hello World!')
    '\n     .. versionadded:: 2019.2.0\n\n    Configures the Nexus switch with the specified commands, via the NX-API.\n\n    commands\n        The list of configuration commands to load on the Nexus switch.\n\n        .. note::\n            This argument is ignored when ``config_file`` is specified.\n\n    config_file\n        The source file with the configuration commands to be sent to the device.\n\n        The file can also be a template that can be rendered using the template\n        engine of choice. This can be specified using the absolute path to the\n        file, or using one of the following URL schemes:\n\n        - ``salt://``\n        - ``https://``\n        - ``ftp:/``\n        - ``s3:/``\n        - ``swift://``\n\n    template_engine: ``jinja``\n        The template engine to use when rendering the source file. Default:\n        ``jinja``. To simply fetch the file without attempting to render, set\n        this argument to ``None``.\n\n    context: ``None``\n        Variables to add to the template context.\n\n    defaults: ``None``\n        Default values of the ``context`` dict.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. Ignored if\n        ``config_file`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm.nxos_api_config \'spanning-tree mode mstp\'\n        salt \'*\' napalm.nxos_api_config config_file=https://bit.ly/2LGLcDy context="{\'servers\': [\'1.2.3.4\']}"\n    '
    nxos_api_kwargs = pyeapi_nxos_api_args(**kwargs)
    return __salt__['nxos_api.config'](commands=commands, config_file=config_file, template_engine=template_engine, context=context, defaults=defaults, saltenv=saltenv, **nxos_api_kwargs)

@proxy_napalm_wrap
def nxos_api_show(commands, raw_text=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Execute one or more show (non-configuration) commands.\n\n    commands\n        The commands to be executed.\n\n    raw_text: ``True``\n        Whether to return raw text or structured data.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.nxos_api_show 'show version'\n        salt '*' napalm.nxos_api_show 'show bgp sessions' 'show processes' raw_text=False\n    "
    nxos_api_kwargs = pyeapi_nxos_api_args(**kwargs)
    return __salt__['nxos_api.show'](commands, raw_text=raw_text, **nxos_api_kwargs)

@proxy_napalm_wrap
def rpc(command, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    This is a wrapper to execute RPC requests on various network operating\n    systems supported by NAPALM, invoking the following functions for the NAPALM\n    native drivers:\n\n    - :py:func:`napalm.junos_rpc <salt.modules.napalm_mod.junos_rpc>` for ``junos``\n    - :py:func:`napalm.pyeapi_run_commands <salt.modules.napalm_mod.pyeapi_run_commands>`\n      for ``eos``\n    - :py:func:`napalm.nxos_api_rpc <salt.modules.napalm_mod.nxos_api_rpc>` for\n      ``nxos``\n    - :py:func:`napalm.netmiko_commands <salt.modules.napalm_mod.netmiko_commands>`\n      for ``ios``, ``iosxr``, and ``nxos_ssh``\n\n    command\n        The RPC command to execute. This depends on the nature of the operating\n        system.\n\n    kwargs\n        Key-value arguments to be sent to the underlying Execution function.\n\n    The function capabilities are extensible in the user environment via the\n    ``napalm_rpc_map`` configuration option / Pillar, e.g.,\n\n    .. code-block:: yaml\n\n        napalm_rpc_map:\n          f5: napalm.netmiko_commands\n          panos: panos.call\n\n    The mapping above reads: when the NAPALM ``os`` Grain is ``f5``, then call\n    ``napalm.netmiko_commands`` for RPC requests.\n\n    By default, if the user does not specify any map, non-native NAPALM drivers\n    will invoke the ``napalm.netmiko_commands`` Execution function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.rpc 'show version'\n        salt '*' napalm.rpc get-interfaces\n    "
    default_map = {'junos': 'napalm.junos_rpc', 'eos': 'napalm.pyeapi_run_commands', 'nxos': 'napalm.nxos_api_rpc'}
    napalm_map = __salt__['config.get']('napalm_rpc_map', {})
    napalm_map.update(default_map)
    fun = napalm_map.get(__grains__['os'], 'napalm.netmiko_commands')
    return __salt__[fun](command, **kwargs)

@depends(HAS_CISCOCONFPARSE)
def config_find_lines(regex, source='running'):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Return the configuration lines that match the regular expressions from the\n    ``regex`` argument. The configuration is read from the network device\n    interrogated.\n\n    regex\n        The regular expression to match the configuration lines against.\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_find_lines '^interface Ethernet1\\d'\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['ciscoconfparse.find_lines'](config=config_txt, regex=regex)

@depends(HAS_CISCOCONFPARSE)
def config_lines_w_child(parent_regex, child_regex, source='running'):
    if False:
        i = 10
        return i + 15
    "\n     .. versionadded:: 2019.2.0\n\n    Return the configuration lines that match the regular expressions from the\n    ``parent_regex`` argument, having child lines matching ``child_regex``.\n    The configuration is read from the network device interrogated.\n\n    .. note::\n        This function is only available only when the underlying library\n        `ciscoconfparse <http://www.pennington.net/py/ciscoconfparse/index.html>`_\n        is installed. See\n        :py:func:`ciscoconfparse module <salt.modules.ciscoconfparse_mod>` for\n        more details.\n\n    parent_regex\n        The regular expression to match the parent configuration lines against.\n\n    child_regex\n        The regular expression to match the child configuration lines against.\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_lines_w_child '^interface' 'ip address'\n        salt '*' napalm.config_lines_w_child '^interface' 'shutdown' source=candidate\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['ciscoconfparse.find_lines_w_child'](config=config_txt, parent_regex=parent_regex, child_regex=child_regex)

@depends(HAS_CISCOCONFPARSE)
def config_lines_wo_child(parent_regex, child_regex, source='running'):
    if False:
        for i in range(10):
            print('nop')
    "\n      .. versionadded:: 2019.2.0\n\n    Return the configuration lines that match the regular expressions from the\n    ``parent_regex`` argument, having the child lines *not* matching\n    ``child_regex``.\n    The configuration is read from the network device interrogated.\n\n    .. note::\n        This function is only available only when the underlying library\n        `ciscoconfparse <http://www.pennington.net/py/ciscoconfparse/index.html>`_\n        is installed. See\n        :py:func:`ciscoconfparse module <salt.modules.ciscoconfparse_mod>` for\n        more details.\n\n    parent_regex\n        The regular expression to match the parent configuration lines against.\n\n    child_regex\n        The regular expression to match the child configuration lines against.\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_lines_wo_child '^interface' 'ip address'\n        salt '*' napalm.config_lines_wo_child '^interface' 'shutdown' source=candidate\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['ciscoconfparse.find_lines_wo_child'](config=config_txt, parent_regex=parent_regex, child_regex=child_regex)

@depends(HAS_CISCOCONFPARSE)
def config_filter_lines(parent_regex, child_regex, source='running'):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Return a list of detailed matches, for the configuration blocks (parent-child\n    relationship) whose parent respects the regular expressions configured via\n    the ``parent_regex`` argument, and the child matches the ``child_regex``\n    regular expression. The result is a list of dictionaries with the following\n    keys:\n\n    - ``match``: a boolean value that tells whether ``child_regex`` matched any\n      children lines.\n    - ``parent``: the parent line (as text).\n    - ``child``: the child line (as text). If no child line matched, this field\n      will be ``None``.\n\n    .. note::\n        This function is only available only when the underlying library\n        `ciscoconfparse <http://www.pennington.net/py/ciscoconfparse/index.html>`_\n        is installed. See\n        :py:func:`ciscoconfparse module <salt.modules.ciscoconfparse_mod>` for\n        more details.\n\n    parent_regex\n        The regular expression to match the parent configuration lines against.\n\n    child_regex\n        The regular expression to match the child configuration lines against.\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_filter_lines '^interface' 'ip address'\n        salt '*' napalm.config_filter_lines '^interface' 'shutdown' source=candidate\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['ciscoconfparse.filter_lines'](config=config_txt, parent_regex=parent_regex, child_regex=child_regex)

def config_tree(source='running', with_tags=False):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2019.2.0\n\n    Transform Cisco IOS style configuration to structured Python dictionary.\n    Depending on the value of the ``with_tags`` argument, this function may\n    provide different views, valuable in different situations.\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    with_tags: ``False``\n        Whether this function should return a detailed view, with tags.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_tree\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['iosconfig.tree'](config=config_txt)

def config_merge_tree(source='running', merge_config=None, merge_path=None, saltenv='base'):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Return the merge tree of the ``initial_config`` with the ``merge_config``,\n    as a Python dictionary.\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    merge_config\n        The config to be merged into the initial config, sent as text. This\n        argument is ignored when ``merge_path`` is set.\n\n    merge_path\n        Absolute or remote path from where to load the merge configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``merge_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_merge_tree merge_path=salt://path/to/merge.cfg\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['iosconfig.merge_tree'](initial_config=config_txt, merge_config=merge_config, merge_path=merge_path, saltenv=saltenv)

def config_merge_text(source='running', merge_config=None, merge_path=None, saltenv='base'):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2019.2.0\n\n    Return the merge result of the configuration from ``source`` with the\n    merge configuration, as plain text (without loading the config on the\n    device).\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    merge_config\n        The config to be merged into the initial config, sent as text. This\n        argument is ignored when ``merge_path`` is set.\n\n    merge_path\n        Absolute or remote path from where to load the merge configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``merge_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_merge_text merge_path=salt://path/to/merge.cfg\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['iosconfig.merge_text'](initial_config=config_txt, merge_config=merge_config, merge_path=merge_path, saltenv=saltenv)

def config_merge_diff(source='running', merge_config=None, merge_path=None, saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Return the merge diff, as text, after merging the merge config into the\n    configuration source requested (without loading the config on the device).\n\n    source: ``running``\n        The configuration type to retrieve from the network device. Default:\n        ``running``. Available options: ``running``, ``startup``, ``candidate``.\n\n    merge_config\n        The config to be merged into the initial config, sent as text. This\n        argument is ignored when ``merge_path`` is set.\n\n    merge_path\n        Absolute or remote path from where to load the merge configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``merge_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_merge_diff merge_path=salt://path/to/merge.cfg\n    "
    config_txt = __salt__['net.config'](source=source)['out'][source]
    return __salt__['iosconfig.merge_diff'](initial_config=config_txt, merge_config=merge_config, merge_path=merge_path, saltenv=saltenv)

def config_diff_tree(source1='candidate', candidate_path=None, source2='running', running_path=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Return the diff, as Python dictionary, between two different sources.\n    The sources can be either specified using the ``source1`` and ``source2``\n    arguments when retrieving from the managed network device.\n\n    source1: ``candidate``\n        The source from where to retrieve the configuration to be compared with.\n        Available options: ``candidate``, ``running``, ``startup``. Default:\n        ``candidate``.\n\n    candidate_path\n        Absolute or remote path from where to load the candidate configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    source2: ``running``\n        The source from where to retrieve the configuration to compare with.\n        Available options: ``candidate``, ``running``, ``startup``. Default:\n        ``running``.\n\n    running_path\n        Absolute or remote path from where to load the running configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``candidate_path`` or ``running_path`` is not a\n        ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_diff_text\n        salt '*' napalm.config_diff_text candidate_path=https://bit.ly/2mAdq7z\n        # Would compare the running config with the configuration available at\n        # https://bit.ly/2mAdq7z\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_diff_tree\n        salt '*' napalm.config_diff_tree running startup\n    "
    get_config = __salt__['net.config']()['out']
    candidate_cfg = get_config[source1]
    running_cfg = get_config[source2]
    return __salt__['iosconfig.diff_tree'](candidate_config=candidate_cfg, candidate_path=candidate_path, running_config=running_cfg, running_path=running_path)

def config_diff_text(source1='candidate', candidate_path=None, source2='running', running_path=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Return the diff, as text, between the two different configuration sources.\n    The sources can be either specified using the ``source1`` and ``source2``\n    arguments when retrieving from the managed network device.\n\n    source1: ``candidate``\n        The source from where to retrieve the configuration to be compared with.\n        Available options: ``candidate``, ``running``, ``startup``. Default:\n        ``candidate``.\n\n    candidate_path\n        Absolute or remote path from where to load the candidate configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    source2: ``running``\n        The source from where to retrieve the configuration to compare with.\n        Available options: ``candidate``, ``running``, ``startup``. Default:\n        ``running``.\n\n    running_path\n        Absolute or remote path from where to load the running configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``candidate_path`` or ``running_path`` is not a\n        ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.config_diff_text\n        salt '*' napalm.config_diff_text candidate_path=https://bit.ly/2mAdq7z\n        # Would compare the running config with the configuration available at\n        # https://bit.ly/2mAdq7z\n    "
    get_config = __salt__['net.config']()['out']
    candidate_cfg = get_config[source1]
    running_cfg = get_config[source2]
    return __salt__['iosconfig.diff_text'](candidate_config=candidate_cfg, candidate_path=candidate_path, running_config=running_cfg, running_path=running_path)

@depends(HAS_SCP)
def scp_get(remote_path, local_path='', recursive=False, preserve_times=False, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2019.2.0\n\n    Transfer files and directories from remote network device to the localhost\n    of the Minion.\n\n    .. note::\n        This function is only available only when the underlying library\n        `scp <https://github.com/jbardin/scp.py>`_\n        is installed. See\n        :mod:`scp module <salt.modules.scp_mod>` for\n        more details.\n\n    remote_path\n        Path to retrieve from remote host. Since this is evaluated by scp on the\n        remote host, shell wildcards and environment variables may be used.\n\n    recursive: ``False``\n        Transfer files and directories recursively.\n\n    preserve_times: ``False``\n        Preserve ``mtime`` and ``atime`` of transferred files and directories.\n\n    passphrase\n        Used for decrypting private keys.\n\n    pkey\n        An optional private key to use for authentication.\n\n    key_filename\n        The filename, or list of filenames, of optional private key(s) and/or\n        certificates to try for authentication.\n\n    timeout\n        An optional timeout (in seconds) for the TCP connect.\n\n    socket_timeout: ``10``\n        The channel socket timeout in seconds.\n\n    buff_size: ``16384``\n        The size of the SCP send buffer.\n\n    allow_agent: ``True``\n        Set to ``False`` to disable connecting to the SSH agent.\n\n    look_for_keys: ``True``\n        Set to ``False`` to disable searching for discoverable private key\n        files in ``~/.ssh/``\n\n    banner_timeout\n        An optional timeout (in seconds) to wait for the SSH banner to be\n        presented.\n\n    auth_timeout\n        An optional timeout (in seconds) to wait for an authentication\n        response.\n\n    auto_add_policy: ``False``\n        Automatically add the host to the ``known_hosts``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.scp_get /var/tmp/file /tmp/file auto_add_policy=True\n    "
    conn_args = netmiko_args(**kwargs)
    conn_args['hostname'] = conn_args['host']
    kwargs.update(conn_args)
    return __salt__['scp.get'](remote_path, local_path=local_path, recursive=recursive, preserve_times=preserve_times, **kwargs)

@depends(HAS_SCP)
def scp_put(files, remote_path=None, recursive=False, preserve_times=False, saltenv='base', **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Transfer files and directories to remote network device.\n\n    .. note::\n        This function is only available only when the underlying library\n        `scp <https://github.com/jbardin/scp.py>`_\n        is installed. See\n        :mod:`scp module <salt.modules.scp_mod>` for\n        more details.\n\n    files\n        A single path or a list of paths to be transferred.\n\n    remote_path\n        The path on the remote device where to store the files.\n\n    recursive: ``True``\n        Transfer files and directories recursively.\n\n    preserve_times: ``False``\n        Preserve ``mtime`` and ``atime`` of transferred files and directories.\n\n    saltenv: ``base``\n        The name of the Salt environment. Ignored when ``files`` is not a\n        ``salt://`` URL.\n\n    hostname\n        The hostname of the remote device.\n\n    port: ``22``\n        The port of the remote device.\n\n    username\n        The username required for SSH authentication on the device.\n\n    password\n        Used for password authentication. It is also used for private key\n        decryption if ``passphrase`` is not given.\n\n    passphrase\n        Used for decrypting private keys.\n\n    pkey\n        An optional private key to use for authentication.\n\n    key_filename\n        The filename, or list of filenames, of optional private key(s) and/or\n        certificates to try for authentication.\n\n    timeout\n        An optional timeout (in seconds) for the TCP connect.\n\n    socket_timeout: ``10``\n        The channel socket timeout in seconds.\n\n    buff_size: ``16384``\n        The size of the SCP send buffer.\n\n    allow_agent: ``True``\n        Set to ``False`` to disable connecting to the SSH agent.\n\n    look_for_keys: ``True``\n        Set to ``False`` to disable searching for discoverable private key\n        files in ``~/.ssh/``\n\n    banner_timeout\n        An optional timeout (in seconds) to wait for the SSH banner to be\n        presented.\n\n    auth_timeout\n        An optional timeout (in seconds) to wait for an authentication\n        response.\n\n    auto_add_policy: ``False``\n        Automatically add the host to the ``known_hosts``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' napalm.scp_put /path/to/file /var/tmp/file auto_add_policy=True\n    "
    conn_args = netmiko_args(**kwargs)
    conn_args['hostname'] = conn_args['host']
    kwargs.update(conn_args)
    return __salt__['scp.put'](files, remote_path=remote_path, recursive=recursive, preserve_times=preserve_times, saltenv=saltenv, **kwargs)