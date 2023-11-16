"""
Support for Config Server Firewall (CSF)
========================================
:maintainer: Mostafa Hussein <mostafa.hussein91@gmail.com>
:maturity: new
:platform: Linux
"""
import re
import salt.utils.path
from salt.exceptions import CommandExecutionError, SaltInvocationError

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if csf exists on the system\n    '
    if salt.utils.path.which('csf') is None:
        return (False, 'The csf execution module cannot be loaded: csf unavailable.')
    else:
        return True

def _temp_exists(method, ip):
    if False:
        i = 10
        return i + 15
    '\n    Checks if the ip exists as a temporary rule based\n    on the method supplied, (tempallow, tempdeny).\n    '
    _type = method.replace('temp', '').upper()
    cmd = "csf -t | awk -v code=1 -v type=_type -v ip=ip '$1==type && $2==ip {{code=0}} END {{exit code}}'".format(_type=_type, ip=ip)
    exists = __salt__['cmd.run_all'](cmd)
    return not bool(exists['retcode'])

def _exists_with_port(method, rule):
    if False:
        return 10
    path = '/etc/csf/csf.{}'.format(method)
    return __salt__['file.contains'](path, rule)

def exists(method, ip, port=None, proto='tcp', direction='in', port_origin='d', ip_origin='d', ttl=None, comment=''):
    if False:
        return 10
    "\n    Returns true a rule for the ip already exists\n    based on the method supplied. Returns false if\n    not found.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.exists allow 1.2.3.4\n        salt '*' csf.exists tempdeny 1.2.3.4\n    "
    if method.startswith('temp'):
        return _temp_exists(method, ip)
    if port:
        rule = _build_port_rule(ip, port, proto, direction, port_origin, ip_origin, comment)
        return _exists_with_port(method, rule)
    exists = __salt__['cmd.run_all']("egrep ^'{} +' /etc/csf/csf.{}".format(ip, method))
    return not bool(exists['retcode'])

def __csf_cmd(cmd):
    if False:
        i = 10
        return i + 15
    '\n    Execute csf command\n    '
    csf_cmd = '{} {}'.format(salt.utils.path.which('csf'), cmd)
    out = __salt__['cmd.run_all'](csf_cmd)
    if out['retcode'] != 0:
        if not out['stderr']:
            ret = out['stdout']
        else:
            ret = out['stderr']
        raise CommandExecutionError('csf failed: {}'.format(ret))
    else:
        ret = out['stdout']
    return ret

def _status_csf():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return True if csf is running otherwise return False\n    '
    cmd = 'test -e /etc/csf/csf.disable'
    out = __salt__['cmd.run_all'](cmd)
    return bool(out['retcode'])

def _get_opt(method):
    if False:
        return 10
    '\n    Returns the cmd option based on a long form argument.\n    '
    opts = {'allow': '-a', 'deny': '-d', 'unallow': '-ar', 'undeny': '-dr', 'tempallow': '-ta', 'tempdeny': '-td', 'temprm': '-tr'}
    return opts[method]

def _build_args(method, ip, comment):
    if False:
        print('Hello World!')
    '\n    Returns the cmd args for csf basic allow/deny commands.\n    '
    opt = _get_opt(method)
    args = '{} {}'.format(opt, ip)
    if comment:
        args += ' {}'.format(comment)
    return args

def _access_rule(method, ip=None, port=None, proto='tcp', direction='in', port_origin='d', ip_origin='d', comment=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handles the cmd execution for allow and deny commands.\n    '
    if _status_csf():
        if ip is None:
            return {'error': 'You must supply an ip address or CIDR.'}
        if port is None:
            args = _build_args(method, ip, comment)
            return __csf_cmd(args)
        else:
            if method not in ['allow', 'deny']:
                return {'error': 'Only allow and deny rules are allowed when specifying a port.'}
            return _access_rule_with_port(method=method, ip=ip, port=port, proto=proto, direction=direction, port_origin=port_origin, ip_origin=ip_origin, comment=comment)

def _build_port_rule(ip, port, proto, direction, port_origin, ip_origin, comment):
    if False:
        while True:
            i = 10
    kwargs = {'ip': ip, 'port': port, 'proto': proto, 'direction': direction, 'port_origin': port_origin, 'ip_origin': ip_origin}
    rule = '{proto}|{direction}|{port_origin}={port}|{ip_origin}={ip}'.format(**kwargs)
    if comment:
        rule += ' #{}'.format(comment)
    return rule

def _remove_access_rule_with_port(method, ip, port, proto='tcp', direction='in', port_origin='d', ip_origin='d', ttl=None):
    if False:
        print('Hello World!')
    rule = _build_port_rule(ip, port=port, proto=proto, direction=direction, port_origin=port_origin, ip_origin=ip_origin, comment='')
    rule = rule.replace('|', '[|]')
    rule = rule.replace('.', '[.]')
    result = __salt__['file.replace']('/etc/csf/csf.{}'.format(method), pattern='^{}(( +)?\\#.*)?$\n'.format(rule), repl='')
    return result

def _csf_to_list(option):
    if False:
        return 10
    '\n    Extract comma-separated values from a csf.conf\n    option and return a list.\n    '
    result = []
    line = get_option(option)
    if line:
        csv = line.split('=')[1].replace(' ', '').replace('"', '')
        result = csv.split(',')
    return result

def split_option(option):
    if False:
        return 10
    return re.split('(?: +)?\\=(?: +)?', option)

def get_option(option):
    if False:
        return 10
    pattern = '^{}(\\ +)?\\=(\\ +)?".*"$'.format(option)
    grep = __salt__['file.grep']('/etc/csf/csf.conf', pattern, '-E')
    if 'stdout' in grep and grep['stdout']:
        line = grep['stdout']
        return line
    return None

def set_option(option, value):
    if False:
        print('Hello World!')
    current_option = get_option(option)
    if not current_option:
        return {'error': 'No such option exists in csf.conf'}
    result = __salt__['file.replace']('/etc/csf/csf.conf', pattern='^{}(\\ +)?\\=(\\ +)?".*"'.format(option), repl='{} = "{}"'.format(option, value))
    return result

def get_skipped_nics(ipv6=False):
    if False:
        i = 10
        return i + 15
    if ipv6:
        option = 'ETH6_DEVICE_SKIP'
    else:
        option = 'ETH_DEVICE_SKIP'
    skipped_nics = _csf_to_list(option)
    return skipped_nics

def skip_nic(nic, ipv6=False):
    if False:
        while True:
            i = 10
    nics = get_skipped_nics(ipv6=ipv6)
    nics.append(nic)
    return skip_nics(nics, ipv6)

def skip_nics(nics, ipv6=False):
    if False:
        while True:
            i = 10
    if ipv6:
        ipv6 = '6'
    else:
        ipv6 = ''
    nics_csv = ','.join(map(str, nics))
    result = __salt__['file.replace']('/etc/csf/csf.conf', pattern='^ETH{}_DEVICE_SKIP(\\ +)?\\=(\\ +)?".*"'.format(ipv6), repl='ETH{}_DEVICE_SKIP = "{}"'.format(ipv6, nics_csv))
    return result

def _access_rule_with_port(method, ip, port, proto='tcp', direction='in', port_origin='d', ip_origin='d', ttl=None, comment=''):
    if False:
        return 10
    results = {}
    if direction == 'both':
        directions = ['in', 'out']
    else:
        directions = [direction]
    for direction in directions:
        _exists = exists(method, ip, port=port, proto=proto, direction=direction, port_origin=port_origin, ip_origin=ip_origin, ttl=ttl, comment=comment)
        if not _exists:
            rule = _build_port_rule(ip, port=port, proto=proto, direction=direction, port_origin=port_origin, ip_origin=ip_origin, comment=comment)
            path = '/etc/csf/csf.{}'.format(method)
            results[direction] = __salt__['file.append'](path, rule)
    return results

def _tmp_access_rule(method, ip=None, ttl=None, port=None, direction='in', port_origin='d', ip_origin='d', comment=''):
    if False:
        while True:
            i = 10
    '\n    Handles the cmd execution for tempdeny and tempallow commands.\n    '
    if _status_csf():
        if ip is None:
            return {'error': 'You must supply an ip address or CIDR.'}
        if ttl is None:
            return {'error': 'You must supply a ttl.'}
        args = _build_tmp_access_args(method, ip, ttl, port, direction, comment)
        return __csf_cmd(args)

def _build_tmp_access_args(method, ip, ttl, port, direction, comment):
    if False:
        for i in range(10):
            print('nop')
    '\n    Builds the cmd args for temporary access/deny opts.\n    '
    opt = _get_opt(method)
    args = '{} {} {}'.format(opt, ip, ttl)
    if port:
        args += ' -p {}'.format(port)
    if direction:
        args += ' -d {}'.format(direction)
    if comment:
        args += ' #{}'.format(comment)
    return args

def running():
    if False:
        i = 10
        return i + 15
    "\n    Check csf status\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.running\n    "
    return _status_csf()

def disable():
    if False:
        return 10
    "\n    Disable csf permanently\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.disable\n    "
    if _status_csf():
        return __csf_cmd('-x')

def enable():
    if False:
        print('Hello World!')
    "\n    Activate csf if not running\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.enable\n    "
    if not _status_csf():
        return __csf_cmd('-e')

def reload():
    if False:
        return 10
    "\n    Restart csf\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.reload\n    "
    return __csf_cmd('-r')

def tempallow(ip=None, ttl=None, port=None, direction=None, comment=''):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add an rule to the temporary ip allow list.\n    See :func:`_access_rule`.\n    1- Add an IP:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.tempallow 127.0.0.1 3600 port=22 direction='in' comment='# Temp dev ssh access'\n    "
    return _tmp_access_rule('tempallow', ip, ttl, port, direction, comment)

def tempdeny(ip=None, ttl=None, port=None, direction=None, comment=''):
    if False:
        print('Hello World!')
    "\n    Add a rule to the temporary ip deny list.\n    See :func:`_access_rule`.\n    1- Add an IP:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.tempdeny 127.0.0.1 300 port=22 direction='in' comment='# Brute force attempt'\n    "
    return _tmp_access_rule('tempdeny', ip, ttl, port, direction, comment)

def allow(ip, port=None, proto='tcp', direction='in', port_origin='d', ip_origin='s', ttl=None, comment=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add an rule to csf allowed hosts\n    See :func:`_access_rule`.\n    1- Add an IP:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' csf.allow 127.0.0.1\n        salt \'*\' csf.allow 127.0.0.1 comment="Allow localhost"\n    '
    return _access_rule('allow', ip, port=port, proto=proto, direction=direction, port_origin=port_origin, ip_origin=ip_origin, comment=comment)

def deny(ip, port=None, proto='tcp', direction='in', port_origin='d', ip_origin='d', ttl=None, comment=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add an rule to csf denied hosts\n    See :func:`_access_rule`.\n    1- Deny an IP:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' csf.deny 127.0.0.1\n        salt \'*\' csf.deny 127.0.0.1 comment="Too localhosty"\n    '
    return _access_rule('deny', ip, port, proto, direction, port_origin, ip_origin, comment)

def remove_temp_rule(ip):
    if False:
        print('Hello World!')
    opt = _get_opt('temprm')
    args = '{} {}'.format(opt, ip)
    return __csf_cmd(args)

def unallow(ip):
    if False:
        i = 10
        return i + 15
    "\n    Remove a rule from the csf denied hosts\n    See :func:`_access_rule`.\n    1- Deny an IP:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.unallow 127.0.0.1\n    "
    return _access_rule('unallow', ip)

def undeny(ip):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove a rule from the csf denied hosts\n    See :func:`_access_rule`.\n    1- Deny an IP:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.undeny 127.0.0.1\n    "
    return _access_rule('undeny', ip)

def remove_rule(method, ip, port=None, proto='tcp', direction='in', port_origin='d', ip_origin='s', ttl=None, comment=''):
    if False:
        print('Hello World!')
    if method.startswith('temp') or ttl:
        return remove_temp_rule(ip)
    if not port:
        if method == 'allow':
            return unallow(ip)
        elif method == 'deny':
            return undeny(ip)
    if port:
        return _remove_access_rule_with_port(method=method, ip=ip, port=port, proto=proto, direction=direction, port_origin=port_origin, ip_origin=ip_origin)

def allow_ports(ports, proto='tcp', direction='in'):
    if False:
        return 10
    '\n    Fully replace the incoming or outgoing ports\n    line in the csf.conf file - e.g. TCP_IN, TCP_OUT,\n    UDP_IN, UDP_OUT, etc.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' csf.allow_ports ports="[22,80,443,4505,4506]" proto=\'tcp\' direction=\'in\'\n    '
    results = []
    ports = set(ports)
    ports = list(ports)
    proto = proto.upper()
    direction = direction.upper()
    _validate_direction_and_proto(direction, proto)
    ports_csv = ','.join(map(str, ports))
    directions = build_directions(direction)
    for direction in directions:
        result = __salt__['file.replace']('/etc/csf/csf.conf', pattern='^{}_{}(\\ +)?\\=(\\ +)?".*"$'.format(proto, direction), repl='{}_{} = "{}"'.format(proto, direction, ports_csv))
        results.append(result)
    return results

def get_ports(proto='tcp', direction='in'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Lists ports from csf.conf based on direction and protocol.\n    e.g. - TCP_IN, TCP_OUT, UDP_IN, UDP_OUT, etc..\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.allow_port 22 proto='tcp' direction='in'\n    "
    proto = proto.upper()
    direction = direction.upper()
    results = {}
    _validate_direction_and_proto(direction, proto)
    directions = build_directions(direction)
    for direction in directions:
        option = '{}_{}'.format(proto, direction)
        results[direction] = _csf_to_list(option)
    return results

def _validate_direction_and_proto(direction, proto):
    if False:
        i = 10
        return i + 15
    if direction.upper() not in ['IN', 'OUT', 'BOTH']:
        raise SaltInvocationError('You must supply a direction of in, out, or both')
    if proto.upper() not in ['TCP', 'UDP', 'TCP6', 'UDP6']:
        raise SaltInvocationError('You must supply tcp, udp, tcp6, or udp6 for the proto keyword')
    return

def build_directions(direction):
    if False:
        return 10
    direction = direction.upper()
    if direction == 'BOTH':
        directions = ['IN', 'OUT']
    else:
        directions = [direction]
    return directions

def allow_port(port, proto='tcp', direction='both'):
    if False:
        while True:
            i = 10
    "\n    Like allow_ports, but it will append to the\n    existing entry instead of replacing it.\n    Takes a single port instead of a list of ports.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' csf.allow_port 22 proto='tcp' direction='in'\n    "
    ports = get_ports(proto=proto, direction=direction)
    direction = direction.upper()
    _validate_direction_and_proto(direction, proto)
    directions = build_directions(direction)
    results = []
    for direction in directions:
        _ports = ports[direction]
        _ports.append(port)
        results += allow_ports(_ports, proto=proto, direction=direction)
    return results

def get_testing_status():
    if False:
        print('Hello World!')
    testing = _csf_to_list('TESTING')[0]
    return testing

def _toggle_testing(val):
    if False:
        return 10
    if val == 'on':
        val = '1'
    elif val == 'off':
        val = '0'
    else:
        raise SaltInvocationError("Only valid arg is 'on' or 'off' here.")
    result = __salt__['file.replace']('/etc/csf/csf.conf', pattern='^TESTING(\\ +)?\\=(\\ +)?".*"', repl='TESTING = "{}"'.format(val))
    return result

def enable_testing_mode():
    if False:
        return 10
    return _toggle_testing('on')

def disable_testing_mode():
    if False:
        while True:
            i = 10
    return _toggle_testing('off')