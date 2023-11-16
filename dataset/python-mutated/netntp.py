"""
Network NTP
===========

.. versionadded:: 2016.11.0

Manage the configuration of NTP peers and servers on the network devices through the NAPALM proxy.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net> & Jerome Fleury <jf@cloudflare.com>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------
- Requires netaddr_ to be installed: `pip install netaddr` to check if IP
  Addresses are correctly specified
- Requires dnspython_ to be installed: `pip install dnspython` to resolve the
  nameserver entities (in case the user does not configure the peers/servers
  using their IP addresses)
- :mod:`NAPALM proxy minion <salt.proxy.napalm>`
- :mod:`NTP operational and configuration management module <salt.modules.napalm_ntp>`

.. _netaddr: https://pythonhosted.org/netaddr/
.. _dnspython: http://www.dnspython.org/
"""
import logging
import salt.utils.napalm
try:
    from netaddr import IPAddress
    from netaddr.core import AddrFormatError
    HAS_NETADDR = True
except ImportError:
    HAS_NETADDR = False
try:
    import dns.resolver
    HAS_DNSRESOLVER = True
except ImportError:
    HAS_DNSRESOLVER = False
__virtualname__ = 'netntp'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def _default_ret(name):
    if False:
        i = 10
        return i + 15
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    return ret

def _retrieve_ntp_peers():
    if False:
        print('Hello World!')
    'Retrieves configured NTP peers'
    return __salt__['ntp.peers']()

def _retrieve_ntp_servers():
    if False:
        while True:
            i = 10
    'Retrieves configured NTP servers'
    return __salt__['ntp.servers']()

def _check(peers):
    if False:
        while True:
            i = 10
    'Checks whether the input is a valid list of peers and transforms domain names into IP Addresses'
    if not isinstance(peers, list):
        return False
    for peer in peers:
        if not isinstance(peer, str):
            return False
    if not HAS_NETADDR:
        return True
    ip_only_peers = []
    for peer in peers:
        try:
            ip_only_peers.append(str(IPAddress(peer)))
        except AddrFormatError:
            if not HAS_DNSRESOLVER:
                continue
            dns_reply = []
            try:
                dns_reply = dns.resolver.query(peer)
            except dns.resolver.NoAnswer:
                return False
            for dns_ip in dns_reply:
                ip_only_peers.append(str(dns_ip))
    peers = ip_only_peers
    return True

def _clean(lst):
    if False:
        return 10
    return [elem for elem in lst if elem]

def _set_ntp_peers(peers):
    if False:
        return 10
    'Calls ntp.set_peers.'
    return __salt__['ntp.set_peers'](*peers, commit=False)

def _set_ntp_servers(servers):
    if False:
        return 10
    'Calls ntp.set_servers.'
    return __salt__['ntp.set_servers'](*servers, commit=False)

def _delete_ntp_peers(peers):
    if False:
        print('Hello World!')
    'Calls ntp.delete_peers.'
    return __salt__['ntp.delete_peers'](*peers, commit=False)

def _delete_ntp_servers(servers):
    if False:
        i = 10
        return i + 15
    'Calls ntp.delete_servers.'
    return __salt__['ntp.delete_servers'](*servers, commit=False)

def _exec_fun(name, *kargs):
    if False:
        while True:
            i = 10
    if name in list(globals().keys()):
        return globals().get(name)(*kargs)
    return None

def _check_diff_and_configure(fun_name, peers_servers, name='peers'):
    if False:
        return 10
    _ret = _default_ret(fun_name)
    _options = ['peers', 'servers']
    if name not in _options:
        return _ret
    _retrieve_fun = '_retrieve_ntp_{what}'.format(what=name)
    ntp_list_output = _exec_fun(_retrieve_fun)
    if ntp_list_output.get('result', False) is False:
        _ret['comment'] = 'Cannot retrieve NTP {what} from the device: {reason}'.format(what=name, reason=ntp_list_output.get('comment'))
        return _ret
    configured_ntp_list = set(ntp_list_output.get('out', {}))
    desired_ntp_list = set(peers_servers)
    if configured_ntp_list == desired_ntp_list:
        _ret.update({'comment': 'NTP {what} already configured as needed.'.format(what=name), 'result': True})
        return _ret
    list_to_set = list(desired_ntp_list - configured_ntp_list)
    list_to_delete = list(configured_ntp_list - desired_ntp_list)
    list_to_set = _clean(list_to_set)
    list_to_delete = _clean(list_to_delete)
    changes = {}
    if list_to_set:
        changes['added'] = list_to_set
    if list_to_delete:
        changes['removed'] = list_to_delete
    _ret.update({'changes': changes})
    if __opts__['test'] is True:
        _ret.update({'result': None, 'comment': 'Testing mode: configuration was not changed!'})
        return _ret
    expected_config_change = False
    successfully_changed = True
    comment = ''
    if list_to_set:
        _set_fun = '_set_ntp_{what}'.format(what=name)
        _set = _exec_fun(_set_fun, list_to_set)
        if _set.get('result'):
            expected_config_change = True
        else:
            successfully_changed = False
            comment += 'Cannot set NTP {what}: {reason}'.format(what=name, reason=_set.get('comment'))
    if list_to_delete:
        _delete_fun = '_delete_ntp_{what}'.format(what=name)
        _removed = _exec_fun(_delete_fun, list_to_delete)
        if _removed.get('result'):
            expected_config_change = True
        else:
            successfully_changed = False
            comment += 'Cannot remove NTP {what}: {reason}'.format(what=name, reason=_removed.get('comment'))
    _ret.update({'successfully_changed': successfully_changed, 'expected_config_change': expected_config_change, 'comment': comment})
    return _ret

def managed(name, peers=None, servers=None):
    if False:
        return 10
    "\n    Manages the configuration of NTP peers and servers on the device, as specified in the state SLS file.\n    NTP entities not specified in these lists will be removed whilst entities not configured on the device will be set.\n\n    SLS Example:\n\n    .. code-block:: yaml\n\n        netntp_example:\n            netntp.managed:\n                 - peers:\n                    - 192.168.0.1\n                    - 172.17.17.1\n                 - servers:\n                    - 24.124.0.251\n                    - 138.236.128.36\n\n    Output example:\n\n    .. code-block:: python\n\n        {\n            'edge01.nrt04': {\n                'netntp_|-netntp_example_|-netntp_example_|-managed': {\n                    'comment': 'NTP servers already configured as needed.',\n                    'name': 'netntp_example',\n                    'start_time': '12:45:24.056659',\n                    'duration': 2938.857,\n                    'changes': {\n                        'peers': {\n                            'removed': [\n                                '192.168.0.2',\n                                '192.168.0.3'\n                            ],\n                            'added': [\n                                '192.168.0.1',\n                                '172.17.17.1'\n                            ]\n                        }\n                    },\n                    'result': None\n                }\n            }\n        }\n    "
    ret = _default_ret(name)
    result = ret.get('result', False)
    comment = ret.get('comment', '')
    changes = ret.get('changes', {})
    if not (isinstance(peers, list) or isinstance(servers, list)):
        return ret
    if isinstance(peers, list) and (not _check(peers)):
        ret['comment'] = 'NTP peers must be a list of valid IP Addresses or Domain Names'
        return ret
    if isinstance(servers, list) and (not _check(servers)):
        ret['comment'] = 'NTP servers must be a list of valid IP Addresses or Domain Names'
        return ret
    successfully_changed = True
    expected_config_change = False
    if isinstance(peers, list):
        _peers_ret = _check_diff_and_configure(name, peers, name='peers')
        expected_config_change = _peers_ret.get('expected_config_change', False)
        successfully_changed = _peers_ret.get('successfully_changed', True)
        result = result and _peers_ret.get('result', False)
        comment += '\n' + _peers_ret.get('comment', '')
        _changed_peers = _peers_ret.get('changes', {})
        if _changed_peers:
            changes['peers'] = _changed_peers
    if isinstance(servers, list):
        _servers_ret = _check_diff_and_configure(name, servers, name='servers')
        expected_config_change = expected_config_change or _servers_ret.get('expected_config_change', False)
        successfully_changed = successfully_changed and _servers_ret.get('successfully_changed', True)
        result = result and _servers_ret.get('result', False)
        comment += '\n' + _servers_ret.get('comment', '')
        _changed_servers = _servers_ret.get('changes', {})
        if _changed_servers:
            changes['servers'] = _changed_servers
    ret.update({'changes': changes})
    if not (changes or expected_config_change):
        ret.update({'result': True, 'comment': 'Device configured properly.'})
        return ret
    if __opts__['test'] is True:
        ret.update({'result': None, 'comment': 'This is in testing mode, the device configuration was not changed!'})
        return ret
    if expected_config_change:
        (config_result, config_comment) = __salt__['net.config_control']()
        result = config_result and successfully_changed
        comment += config_comment
    ret.update({'result': result, 'comment': comment})
    return ret