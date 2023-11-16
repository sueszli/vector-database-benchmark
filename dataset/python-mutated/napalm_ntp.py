"""
NAPALM NTP
==========

Manages NTP on network devices.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net> & Jerome Fleury <jf@cloudflare.com>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------
- :mod:`NAPALM proxy minion <salt.proxy.napalm>`
- :mod:`NET basic features <salt.modules.napalm_network>`

.. seealso::
    :mod:`NTP peers management state <salt.states.netntp>`

.. versionadded:: 2016.11.0
"""
import logging
import salt.utils.napalm
from salt.utils.napalm import proxy_napalm_wrap
log = logging.getLogger(__file__)
__virtualname__ = 'ntp'
__proxyenabled__ = ['napalm']
__virtual_aliases__ = ('napalm_ntp',)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

@proxy_napalm_wrap
def peers(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a list the NTP peers configured on the network device.\n\n    :return: configured NTP peers as list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ntp.peers\n\n    Example output:\n\n    .. code-block:: python\n\n        [\n            '192.168.0.1',\n            '172.17.17.1',\n            '172.17.17.2',\n            '2400:cb00:6:1024::c71b:840a'\n        ]\n\n    "
    ntp_peers = salt.utils.napalm.call(napalm_device, 'get_ntp_peers', **{})
    if not ntp_peers.get('result'):
        return ntp_peers
    ntp_peers_list = list(ntp_peers.get('out', {}).keys())
    ntp_peers['out'] = ntp_peers_list
    return ntp_peers

@proxy_napalm_wrap
def servers(**kwargs):
    if False:
        return 10
    "\n    Returns a list of the configured NTP servers on the device.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ntp.servers\n\n    Example output:\n\n    .. code-block:: python\n\n        [\n            '192.168.0.1',\n            '172.17.17.1',\n            '172.17.17.2',\n            '2400:cb00:6:1024::c71b:840a'\n        ]\n    "
    ntp_servers = salt.utils.napalm.call(napalm_device, 'get_ntp_servers', **{})
    if not ntp_servers.get('result'):
        return ntp_servers
    ntp_servers_list = list(ntp_servers.get('out', {}).keys())
    ntp_servers['out'] = ntp_servers_list
    return ntp_servers

@proxy_napalm_wrap
def stats(peer=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a dictionary containing synchronization details of the NTP peers.\n\n    :param peer: Returns only the details of a specific NTP peer.\n    :return: a list of dictionaries, with the following keys:\n\n        * remote\n        * referenceid\n        * synchronized\n        * stratum\n        * type\n        * when\n        * hostpoll\n        * reachability\n        * delay\n        * offset\n        * jitter\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ntp.stats\n\n    Example output:\n\n    .. code-block:: python\n\n        [\n            {\n                'remote'        : '188.114.101.4',\n                'referenceid'   : '188.114.100.1',\n                'synchronized'  : True,\n                'stratum'       : 4,\n                'type'          : '-',\n                'when'          : '107',\n                'hostpoll'      : 256,\n                'reachability'  : 377,\n                'delay'         : 164.228,\n                'offset'        : -13.866,\n                'jitter'        : 2.695\n            }\n        ]\n    "
    proxy_output = salt.utils.napalm.call(napalm_device, 'get_ntp_stats', **{})
    if not proxy_output.get('result'):
        return proxy_output
    ntp_peers = proxy_output.get('out')
    if peer:
        ntp_peers = [ntp_peer for ntp_peer in ntp_peers if ntp_peer.get('remote', '') == peer]
    proxy_output.update({'out': ntp_peers})
    return proxy_output

@proxy_napalm_wrap
def set_peers(*peers, **options):
    if False:
        while True:
            i = 10
    "\n    Configures a list of NTP peers on the device.\n\n    :param peers: list of IP Addresses/Domain Names\n    :param test (bool): discard loaded config. By default ``test`` is False\n        (will not dicard the changes)\n    :commit commit (bool): commit loaded config. By default ``commit`` is True\n        (will commit the changes). Useful when the user does not want to commit\n        after each change, but after a couple.\n\n    By default this function will commit the config changes (if any). To load without committing, use the `commit`\n    option. For dry run use the `test` argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ntp.set_peers 192.168.0.1 172.17.17.1 time.apple.com\n        salt '*' ntp.set_peers 172.17.17.1 test=True  # only displays the diff\n        salt '*' ntp.set_peers 192.168.0.1 commit=False  # preserves the changes, but does not commit\n    "
    test = options.pop('test', False)
    commit = options.pop('commit', True)
    return __salt__['net.load_template']('set_ntp_peers', peers=peers, test=test, commit=commit, inherit_napalm_device=napalm_device)

@proxy_napalm_wrap
def set_servers(*servers, **options):
    if False:
        i = 10
        return i + 15
    "\n    Configures a list of NTP servers on the device.\n\n    :param servers: list of IP Addresses/Domain Names\n    :param test (bool): discard loaded config. By default ``test`` is False\n        (will not dicard the changes)\n    :commit commit (bool): commit loaded config. By default ``commit`` is True\n        (will commit the changes). Useful when the user does not want to commit\n        after each change, but after a couple.\n\n    By default this function will commit the config changes (if any). To load without committing, use the `commit`\n    option. For dry run use the `test` argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ntp.set_servers 192.168.0.1 172.17.17.1 time.apple.com\n        salt '*' ntp.set_servers 172.17.17.1 test=True  # only displays the diff\n        salt '*' ntp.set_servers 192.168.0.1 commit=False  # preserves the changes, but does not commit\n    "
    test = options.pop('test', False)
    commit = options.pop('commit', True)
    return __salt__['net.load_template']('set_ntp_servers', servers=servers, test=test, commit=commit, inherit_napalm_device=napalm_device)

@proxy_napalm_wrap
def delete_peers(*peers, **options):
    if False:
        print('Hello World!')
    "\n    Removes NTP peers configured on the device.\n\n    :param peers: list of IP Addresses/Domain Names to be removed as NTP peers\n    :param test (bool): discard loaded config. By default ``test`` is False\n        (will not dicard the changes)\n    :param commit (bool): commit loaded config. By default ``commit`` is True\n        (will commit the changes). Useful when the user does not want to commit\n        after each change, but after a couple.\n\n    By default this function will commit the config changes (if any). To load\n    without committing, use the ``commit`` option. For a dry run, use the\n    ``test`` argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ntp.delete_peers 8.8.8.8 time.apple.com\n        salt '*' ntp.delete_peers 172.17.17.1 test=True  # only displays the diff\n        salt '*' ntp.delete_peers 192.168.0.1 commit=False  # preserves the changes, but does not commit\n    "
    test = options.pop('test', False)
    commit = options.pop('commit', True)
    return __salt__['net.load_template']('delete_ntp_peers', peers=peers, test=test, commit=commit, inherit_napalm_device=napalm_device)

@proxy_napalm_wrap
def delete_servers(*servers, **options):
    if False:
        return 10
    "\n    Removes NTP servers configured on the device.\n\n    :param servers: list of IP Addresses/Domain Names to be removed as NTP\n        servers\n    :param test (bool): discard loaded config. By default ``test`` is False\n        (will not dicard the changes)\n    :param commit (bool): commit loaded config. By default ``commit`` is True\n        (will commit the changes). Useful when the user does not want to commit\n        after each change, but after a couple.\n\n    By default this function will commit the config changes (if any). To load\n    without committing, use the ``commit`` option. For dry run use the ``test``\n    argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ntp.delete_servers 8.8.8.8 time.apple.com\n        salt '*' ntp.delete_servers 172.17.17.1 test=True  # only displays the diff\n        salt '*' ntp.delete_servers 192.168.0.1 commit=False  # preserves the changes, but does not commit\n    "
    test = options.pop('test', False)
    commit = options.pop('commit', True)
    return __salt__['net.load_template']('delete_ntp_servers', servers=servers, test=test, commit=commit, inherit_napalm_device=napalm_device)