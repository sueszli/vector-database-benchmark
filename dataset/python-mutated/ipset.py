"""
Support for ipset
"""
import logging
import salt.utils.path
from salt._compat import ipaddress
log = logging.getLogger(__name__)
_IPSET_FAMILIES = {'ipv4': 'inet', 'ip4': 'inet', 'ipv6': 'inet6', 'ip6': 'inet6'}
_IPSET_SET_TYPES = {'bitmap:ip', 'bitmap:ip,mac', 'bitmap:port', 'hash:ip', 'hash:mac', 'hash:ip,port', 'hash:ip,port,ip', 'hash:ip,port,net', 'hash:net', 'hash:net,net', 'hash:net,iface', 'hash:net,port', 'hash:net,port,net', 'hash:ip,mark', 'list:set'}
_CREATE_OPTIONS = {'bitmap:ip': {'range', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'bitmap:ip,mac': {'range', 'timeout', 'counters', 'comment', 'skbinfo'}, 'bitmap:port': {'range', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:ip': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:mac': {'hashsize', 'maxelem', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:net': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:net,net': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:net,port': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:net,port,net': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:ip,port,ip': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:ip,port,net': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:ip,port': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:ip,mark': {'family', 'markmask', 'hashsize', 'maxelem', 'timeout', 'counters', 'comment', 'skbinfo'}, 'hash:net,iface': {'family', 'hashsize', 'maxelem', 'netmask', 'timeout', 'counters', 'comment', 'skbinfo'}, 'list:set': {'size', 'timeout', 'counters', 'comment'}}
_CREATE_OPTIONS_WITHOUT_VALUE = {'comment', 'counters', 'skbinfo'}
_CREATE_OPTIONS_REQUIRED = {'bitmap:ip': ['range'], 'bitmap:ip,mac': ['range'], 'bitmap:port': ['range'], 'hash:ip': [], 'hash:mac': [], 'hash:net': [], 'hash:net,net': [], 'hash:ip,port': [], 'hash:net,port': [], 'hash:ip,port,ip': [], 'hash:ip,port,net': [], 'hash:net,port,net': [], 'hash:net,iface': [], 'hash:ip,mark': [], 'list:set': []}
_ADD_OPTIONS = {'bitmap:ip': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'bitmap:ip,mac': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'bitmap:port': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio'}, 'hash:ip': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:mac': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:net': {'timeout', 'nomatch', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:net,net': {'timeout', 'nomatch', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:net,port': {'timeout', 'nomatch', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:net,port,net': {'timeout', 'nomatch', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:ip,port,ip': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:ip,port,net': {'timeout', 'nomatch', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:ip,port': {'timeout', 'nomatch', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:net,iface': {'timeout', 'nomatch', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'hash:ip,mark': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}, 'list:set': {'timeout', 'packets', 'bytes', 'skbmark', 'skbprio', 'skbqueue'}}
__virtualname__ = 'ipset'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load the module if ipset is installed\n    '
    if salt.utils.path.which('ipset'):
        return True
    return (False, 'The ipset execution modules cannot be loaded: ipset binary not in path.')

def _ipset_cmd():
    if False:
        i = 10
        return i + 15
    '\n    Return correct command\n    '
    return salt.utils.path.which('ipset')

def version():
    if False:
        return 10
    "\n    Return version from ipset --version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.version\n\n    "
    cmd = [_ipset_cmd(), '--version']
    out = __salt__['cmd.run'](cmd, python_shell=False).split()
    return out[1]

def new_set(name=None, set_type=None, family='ipv4', comment=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2014.7.0\n\n    Create new custom set\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.new_set custom_set list:set\n\n        salt '*' ipset.new_set custom_set list:set comment=True\n\n        IPv6:\n        salt '*' ipset.new_set custom_set list:set family=ipv6\n    "
    ipset_family = _IPSET_FAMILIES[family]
    if not name:
        return 'Error: Set Name needs to be specified'
    if not set_type:
        return 'Error: Set Type needs to be specified'
    if set_type not in _IPSET_SET_TYPES:
        return 'Error: Set Type is invalid'
    for item in _CREATE_OPTIONS_REQUIRED[set_type]:
        if item not in kwargs:
            return f'Error: {item} is a required argument'
    cmd = [_ipset_cmd(), 'create', name, set_type]
    for item in _CREATE_OPTIONS[set_type]:
        if item in kwargs:
            if item in _CREATE_OPTIONS_WITHOUT_VALUE:
                cmd.append(item)
            else:
                cmd.extend([item, kwargs[item]])
    if 'family' in _CREATE_OPTIONS[set_type]:
        cmd.extend(['family', ipset_family])
    if comment:
        cmd.append('comment')
    out = __salt__['cmd.run'](cmd, python_shell=False)
    if not out:
        out = True
    return out

def delete_set(name=None, family='ipv4'):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2014.7.0\n\n    Delete ipset set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.delete_set custom_set\n\n        IPv6:\n        salt '*' ipset.delete_set custom_set family=ipv6\n    "
    if not name:
        return 'Error: Set needs to be specified'
    cmd = [_ipset_cmd(), 'destroy', name]
    out = __salt__['cmd.run'](cmd, python_shell=False)
    if not out:
        out = True
    return out

def rename_set(name=None, new_set=None, family='ipv4'):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2014.7.0\n\n    Delete ipset set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.rename_set custom_set new_set=new_set_name\n\n        IPv6:\n        salt '*' ipset.rename_set custom_set new_set=new_set_name family=ipv6\n    "
    if not name:
        return 'Error: Set needs to be specified'
    if not new_set:
        return 'Error: New name for set needs to be specified'
    settype = _find_set_type(name)
    if not settype:
        return 'Error: Set does not exist'
    settype = _find_set_type(new_set)
    if settype:
        return 'Error: New Set already exists'
    cmd = [_ipset_cmd(), 'rename', name, new_set]
    out = __salt__['cmd.run'](cmd, python_shell=False)
    if not out:
        out = True
    return out

def list_sets(family='ipv4'):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2014.7.0\n\n    List all ipset sets.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.list_sets\n\n    "
    cmd = [_ipset_cmd(), 'list', '-t']
    out = __salt__['cmd.run'](cmd, python_shell=False)
    _tmp = out.split('\n')
    count = 0
    sets = []
    sets.append({})
    for item in _tmp:
        if not item:
            count = count + 1
            sets.append({})
            continue
        (key, value) = item.split(':', 1)
        sets[count][key] = value[1:]
    return sets

def check_set(name=None, family='ipv4'):
    if False:
        return 10
    "\n    Check that given ipset set exists.\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.check_set name\n\n    "
    if not name:
        return 'Error: Set needs to be specified'
    setinfo = _find_set_info(name)
    if not setinfo:
        return False
    return True

def add(name=None, entry=None, family='ipv4', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Append an entry to the specified set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.add name 192.168.1.26\n\n        salt '*' ipset.add name 192.168.0.3,AA:BB:CC:DD:EE:FF\n\n    "
    if not name:
        return 'Error: Set needs to be specified'
    if not entry:
        return 'Error: Entry needs to be specified'
    setinfo = _find_set_info(name)
    if not setinfo:
        return f'Error: Set {name} does not exist'
    settype = setinfo['Type']
    cmd = [_ipset_cmd(), 'add', '-exist', name] + entry.split()
    if 'timeout' in kwargs:
        if 'timeout' not in setinfo['Header']:
            return f'Error: Set {name} not created with timeout support'
    if 'packets' in kwargs or 'bytes' in kwargs:
        if 'counters' not in setinfo['Header']:
            return f'Error: Set {name} not created with counters support'
    if 'comment' in kwargs:
        if 'comment' not in setinfo['Header']:
            return f'Error: Set {name} not created with comment support'
        if 'comment' not in entry:
            cmd = cmd + ['comment', f"{kwargs['comment']}"]
    if {'skbmark', 'skbprio', 'skbqueue'} & set(kwargs.keys()):
        if 'skbinfo' not in setinfo['Header']:
            return f'Error: Set {name} not created with skbinfo support'
    for item in _ADD_OPTIONS[settype]:
        if item in kwargs:
            cmd.extend([item, kwargs[item]])
    current_members = _find_set_members(name)
    if entry in current_members:
        return f'Warn: Entry {entry} already exists in set {name}'
    out = __salt__['cmd.run'](cmd, python_shell=False)
    if not out:
        return 'Success'
    return f'Error: {out}'

def delete(name=None, entry=None, family='ipv4', **kwargs):
    if False:
        return 10
    "\n    Delete an entry from the specified set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.delete name 192.168.0.3,AA:BB:CC:DD:EE:FF\n\n    "
    if not name:
        return 'Error: Set needs to be specified'
    if not entry:
        return 'Error: Entry needs to be specified'
    settype = _find_set_type(name)
    if not settype:
        return f'Error: Set {name} does not exist'
    cmd = [_ipset_cmd(), 'del', name, entry]
    out = __salt__['cmd.run'](cmd, python_shell=False)
    if not out:
        return 'Success'
    return f'Error: {out}'

def check(name=None, entry=None, family='ipv4'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that an entry exists in the specified set.\n\n    name\n        The ipset name\n\n    entry\n        An entry in the ipset.  This parameter can be a single IP address, a\n        range of IP addresses, or a subnet block.  Example:\n\n        .. code-block:: cfg\n\n            192.168.0.1\n            192.168.0.2-192.168.0.19\n            192.168.0.0/25\n\n    family\n        IP protocol version: ipv4 or ipv6\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ipset.check name \'192.168.0.1 comment "Hello"\'\n\n    '
    if not name:
        return 'Error: Set needs to be specified'
    if not entry:
        return 'Error: Entry needs to be specified'
    settype = _find_set_type(name)
    if not settype:
        return f'Error: Set {name} does not exist'
    current_members = _parse_members(settype, _find_set_members(name))
    if not current_members:
        return False
    if isinstance(entry, list):
        entries = _parse_members(settype, entry)
    else:
        entries = [_parse_member(settype, entry)]
    for current_member in current_members:
        for entry in entries:
            if _member_contains(current_member, entry):
                return True
    return False

def test(name=None, entry=None, family='ipv4', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Test if an entry is in the specified set.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.test name 192.168.0.2\n\n        IPv6:\n        salt '*' ipset.test name fd81:fc56:9ac7::/48\n    "
    if not name:
        return 'Error: Set needs to be specified'
    if not entry:
        return 'Error: Entry needs to be specified'
    settype = _find_set_type(name)
    if not settype:
        return f'Error: Set {name} does not exist'
    cmd = [_ipset_cmd(), 'test', name, entry]
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode'] > 0:
        return False
    return True

def flush(name=None, family='ipv4'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Flush entries in the specified set,\n    Flush all sets if set is not specified.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ipset.flush\n\n        salt '*' ipset.flush set\n\n        IPv6:\n        salt '*' ipset.flush\n\n        salt '*' ipset.flush set\n    "
    cmd = [_ipset_cmd(), 'flush']
    if name:
        cmd.append(name)
    out = __salt__['cmd.run'](cmd, python_shell=False)
    return not out

def _find_set_members(name):
    if False:
        while True:
            i = 10
    '\n    Return list of members for a set\n    '
    cmd = [_ipset_cmd(), 'list', name]
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode'] > 0:
        return False
    _tmp = out['stdout'].split('\n')
    members = []
    startMembers = False
    for i in _tmp:
        if startMembers:
            members.append(i)
        if 'Members:' in i:
            startMembers = True
    return members

def _find_set_info(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return information about the set\n    '
    cmd = [_ipset_cmd(), 'list', '-t', name]
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode'] > 0:
        return False
    setinfo = {}
    _tmp = out['stdout'].split('\n')
    for item in _tmp:
        if ':' in item:
            (key, value) = item.split(':', 1)
            setinfo[key] = value[1:]
    return setinfo

def _find_set_type(name):
    if False:
        print('Hello World!')
    '\n    Find the type of the set\n    '
    setinfo = _find_set_info(name)
    if setinfo:
        return setinfo['Type']
    else:
        return False

def _parse_members(settype, members):
    if False:
        while True:
            i = 10
    if isinstance(members, str):
        return [_parse_member(settype, members)]
    return [_parse_member(settype, member) for member in members]

def _parse_member(settype, member, strict=False):
    if False:
        while True:
            i = 10
    subtypes = settype.split(':')[1].split(',')
    all_parts = member.split(' ', 1)
    parts = all_parts[0].split(',')
    parsed_member = []
    for (i, subtype) in enumerate(subtypes):
        part = parts[i]
        if subtype in ['ip', 'net']:
            try:
                if '/' in part:
                    part = ipaddress.ip_network(part, strict=strict)
                elif '-' in part:
                    (start, end) = list(map(ipaddress.ip_address, part.split('-')))
                    part = list(ipaddress.summarize_address_range(start, end))
                else:
                    part = ipaddress.ip_address(part)
            except ValueError:
                pass
        elif subtype == 'port':
            part = int(part)
        parsed_member.append(part)
    if len(all_parts) > 1:
        parsed_member.append(all_parts[1])
    return parsed_member

def _members_contain(members, entry):
    if False:
        while True:
            i = 10
    pass

def _member_contains(member, entry):
    if False:
        print('Hello World!')
    if len(member) < len(entry):
        return False
    for (i, _entry) in enumerate(entry):
        if not _compare_member_parts(member[i], _entry):
            return False
    return True

def _compare_member_parts(member_part, entry_part):
    if False:
        while True:
            i = 10
    if member_part == entry_part:
        return True
    if isinstance(entry_part, list):
        for entry_part_item in entry_part:
            if not _compare_member_parts(member_part, entry_part_item):
                return False
        return True
    if _is_address(member_part):
        if _is_network(entry_part):
            return member_part in entry_part
    elif _is_network(member_part):
        if _is_address(entry_part):
            return entry_part in member_part
        return False
    return False

def _is_network(o):
    if False:
        i = 10
        return i + 15
    return isinstance(o, (ipaddress.IPv4Network, ipaddress.IPv6Network))

def _is_address(o):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(o, (ipaddress.IPv4Address, ipaddress.IPv6Address))