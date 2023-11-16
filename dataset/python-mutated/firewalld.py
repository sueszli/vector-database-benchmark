"""
Support for firewalld.

.. versionadded:: 2015.2.0
"""
import logging
import re
import salt.utils.path
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Check to see if firewall-cmd exists\n    '
    if salt.utils.path.which('firewall-cmd'):
        return True
    return (False, 'The firewalld execution module cannot be loaded: the firewall-cmd binary is not in the path.')

def __firewall_cmd(cmd):
    if False:
        i = 10
        return i + 15
    '\n    Return the firewall-cmd location\n    '
    firewall_cmd = '{} {}'.format(salt.utils.path.which('firewall-cmd'), cmd)
    out = __salt__['cmd.run_all'](firewall_cmd)
    if out['retcode'] != 0:
        if not out['stderr']:
            msg = out['stdout']
        else:
            msg = out['stderr']
        raise CommandExecutionError('firewall-cmd failed: {}'.format(msg))
    return out['stdout']

def __mgmt(name, _type, action):
    if False:
        return 10
    '\n    Perform zone management\n    '
    cmd = '--{}-{}={} --permanent'.format(action, _type, name)
    return __firewall_cmd(cmd)

def __parse_zone(cmd):
    if False:
        return 10
    '\n    Return zone information in a dictionary\n    '
    _zone = {}
    id_ = ''
    for i in __firewall_cmd(cmd).splitlines():
        if i.strip():
            if re.match('^[a-z0-9]', i, re.I):
                zone_name = i.rstrip()
            else:
                if i.startswith('\t'):
                    _zone[zone_name][id_].append(i.strip())
                    continue
                (id_, val) = i.split(':', 1)
                id_ = id_.strip()
                if _zone.get(zone_name, None):
                    _zone[zone_name].update({id_: [val.strip()]})
                else:
                    _zone[zone_name] = {id_: [val.strip()]}
    return _zone

def version():
    if False:
        while True:
            i = 10
    "\n    Return version from firewall-cmd\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.version\n    "
    return __firewall_cmd('--version')

def reload_rules():
    if False:
        return 10
    "\n    Reload the firewall rules, which makes the permanent configuration the new\n    runtime configuration without losing state information.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.reload_rules\n    "
    return __firewall_cmd('--reload')

def default_zone():
    if False:
        for i in range(10):
            print('nop')
    "\n    Print default zone for connections and interfaces\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.default_zone\n    "
    return __firewall_cmd('--get-default-zone')

def list_zones(permanent=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    List everything added for or enabled in all zones\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.list_zones\n    "
    cmd = '--list-all-zones'
    if permanent:
        cmd += ' --permanent'
    return __parse_zone(cmd)

def get_zones(permanent=True):
    if False:
        return 10
    "\n    Print predefined zones\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_zones\n    "
    cmd = '--get-zones'
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def get_services(permanent=True):
    if False:
        i = 10
        return i + 15
    "\n    Print predefined services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_services\n    "
    cmd = '--get-services'
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def get_icmp_types(permanent=True):
    if False:
        while True:
            i = 10
    "\n    Print predefined icmptypes\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_icmp_types\n    "
    cmd = '--get-icmptypes'
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def new_zone(zone, restart=True):
    if False:
        return 10
    "\n    Add a new zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.new_zone my_zone\n\n    By default firewalld will be reloaded. However, to avoid reloading\n    you need to specify the restart as False\n\n    .. code-block:: bash\n\n        salt '*' firewalld.new_zone my_zone False\n    "
    out = __mgmt(zone, 'zone', 'new')
    if restart:
        if out == 'success':
            return __firewall_cmd('--reload')
    return out

def delete_zone(zone, restart=True):
    if False:
        print('Hello World!')
    "\n    Delete an existing zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.delete_zone my_zone\n\n    By default firewalld will be reloaded. However, to avoid reloading\n    you need to specify the restart as False\n\n    .. code-block:: bash\n\n        salt '*' firewalld.delete_zone my_zone False\n    "
    out = __mgmt(zone, 'zone', 'delete')
    if restart:
        if out == 'success':
            return __firewall_cmd('--reload')
    return out

def set_default_zone(zone):
    if False:
        i = 10
        return i + 15
    "\n    Set default zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.set_default_zone damian\n    "
    return __firewall_cmd('--set-default-zone={}'.format(zone))

def new_service(name, restart=True):
    if False:
        return 10
    "\n    Add a new service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.new_service my_service\n\n    By default firewalld will be reloaded. However, to avoid reloading\n    you need to specify the restart as False\n\n    .. code-block:: bash\n\n        salt '*' firewalld.new_service my_service False\n    "
    out = __mgmt(name, 'service', 'new')
    if restart:
        if out == 'success':
            return __firewall_cmd('--reload')
    return out

def delete_service(name, restart=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete an existing service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.delete_service my_service\n\n    By default firewalld will be reloaded. However, to avoid reloading\n    you need to specify the restart as False\n\n    .. code-block:: bash\n\n        salt '*' firewalld.delete_service my_service False\n    "
    out = __mgmt(name, 'service', 'delete')
    if restart:
        if out == 'success':
            return __firewall_cmd('--reload')
    return out

def list_all(zone=None, permanent=True):
    if False:
        return 10
    "\n    List everything added for or enabled in a zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.list_all\n\n    List a specific zone\n\n    .. code-block:: bash\n\n        salt '*' firewalld.list_all my_zone\n    "
    if zone:
        cmd = '--zone={} --list-all'.format(zone)
    else:
        cmd = '--list-all'
    if permanent:
        cmd += ' --permanent'
    return __parse_zone(cmd)

def list_services(zone=None, permanent=True):
    if False:
        print('Hello World!')
    "\n    List services added for zone as a space separated list.\n    If zone is omitted, default zone will be used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.list_services\n\n    List a specific zone\n\n    .. code-block:: bash\n\n        salt '*' firewalld.list_services my_zone\n    "
    if zone:
        cmd = '--zone={} --list-services'.format(zone)
    else:
        cmd = '--list-services'
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def add_service(service, zone=None, permanent=True):
    if False:
        return 10
    "\n    Add a service for zone. If zone is omitted, default zone will be used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_service ssh\n\n    To assign a service to a specific zone:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_service ssh my_zone\n    "
    if zone:
        cmd = '--zone={} --add-service={}'.format(zone, service)
    else:
        cmd = '--add-service={}'.format(service)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def remove_service(service, zone=None, permanent=True):
    if False:
        return 10
    "\n    Remove a service from zone. This option can be specified multiple times.\n    If zone is omitted, default zone will be used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_service ssh\n\n    To remove a service from a specific zone\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_service ssh dmz\n    "
    if zone:
        cmd = '--zone={} --remove-service={}'.format(zone, service)
    else:
        cmd = '--remove-service={}'.format(service)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def add_service_port(service, port):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add a new port to the specified service.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_service_port zone 80\n    "
    if service not in get_services(permanent=True):
        raise CommandExecutionError('The service does not exist.')
    cmd = '--permanent --service={} --add-port={}'.format(service, port)
    return __firewall_cmd(cmd)

def remove_service_port(service, port):
    if False:
        return 10
    "\n    Remove a port from the specified service.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_service_port zone 80\n    "
    if service not in get_services(permanent=True):
        raise CommandExecutionError('The service does not exist.')
    cmd = '--permanent --service={} --remove-port={}'.format(service, port)
    return __firewall_cmd(cmd)

def get_service_ports(service):
    if False:
        while True:
            i = 10
    "\n    List ports of a service.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_service_ports zone\n    "
    cmd = '--permanent --service={} --get-ports'.format(service)
    return __firewall_cmd(cmd).split()

def add_service_protocol(service, protocol):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add a new protocol to the specified service.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_service_protocol zone ssh\n    "
    cmd = '--permanent --service={} --add-protocol={}'.format(service, protocol)
    return __firewall_cmd(cmd)

def remove_service_protocol(service, protocol):
    if False:
        return 10
    "\n    Remove a protocol from the specified service.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_service_protocol zone ssh\n    "
    cmd = '--permanent --service={} --remove-protocol={}'.format(service, protocol)
    return __firewall_cmd(cmd)

def get_service_protocols(service):
    if False:
        print('Hello World!')
    "\n    List protocols of a service.\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_service_protocols zone\n    "
    cmd = '--permanent --service={} --get-protocols'.format(service)
    return __firewall_cmd(cmd).split()

def get_masquerade(zone=None, permanent=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Show if masquerading is enabled on a zone.\n    If zone is omitted, default zone will be used.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_masquerade zone\n    "
    zone_info = list_all(zone, permanent)
    if 'no' in [zone_info[i]['masquerade'][0] for i in zone_info]:
        return False
    return True

def add_masquerade(zone=None, permanent=True):
    if False:
        while True:
            i = 10
    "\n    Enable masquerade on a zone.\n    If zone is omitted, default zone will be used.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_masquerade\n\n    To enable masquerade on a specific zone\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_masquerade dmz\n    "
    if zone:
        cmd = '--zone={} --add-masquerade'.format(zone)
    else:
        cmd = '--add-masquerade'
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def remove_masquerade(zone=None, permanent=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove masquerade on a zone.\n    If zone is omitted, default zone will be used.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_masquerade\n\n    To remove masquerade on a specific zone\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_masquerade dmz\n    "
    if zone:
        cmd = '--zone={} --remove-masquerade'.format(zone)
    else:
        cmd = '--remove-masquerade'
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def add_port(zone, port, permanent=True, force_masquerade=False):
    if False:
        i = 10
        return i + 15
    "\n    Allow specific ports in a zone.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_port internal 443/tcp\n\n    force_masquerade\n        when a zone is created ensure masquerade is also enabled\n        on that zone.\n    "
    if force_masquerade and (not get_masquerade(zone)):
        add_masquerade(zone)
    cmd = '--zone={} --add-port={}'.format(zone, port)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def remove_port(zone, port, permanent=True):
    if False:
        return 10
    "\n    Remove a specific port from a zone.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_port internal 443/tcp\n    "
    cmd = '--zone={} --remove-port={}'.format(zone, port)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def list_ports(zone, permanent=True):
    if False:
        while True:
            i = 10
    "\n    List all ports in a zone.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.list_ports\n    "
    cmd = '--zone={} --list-ports'.format(zone)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def add_port_fwd(zone, src, dest, proto='tcp', dstaddr='', permanent=True, force_masquerade=False):
    if False:
        return 10
    "\n    Add port forwarding.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_port_fwd public 80 443 tcp\n\n    force_masquerade\n        when a zone is created ensure masquerade is also enabled\n        on that zone.\n    "
    if force_masquerade and (not get_masquerade(zone)):
        add_masquerade(zone)
    cmd = '--zone={} --add-forward-port=port={}:proto={}:toport={}:toaddr={}'.format(zone, src, proto, dest, dstaddr)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def remove_port_fwd(zone, src, dest, proto='tcp', dstaddr='', permanent=True):
    if False:
        i = 10
        return i + 15
    "\n    Remove Port Forwarding.\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_port_fwd public 80 443 tcp\n    "
    cmd = '--zone={} --remove-forward-port=port={}:proto={}:toport={}:toaddr={}'.format(zone, src, proto, dest, dstaddr)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def list_port_fwd(zone, permanent=True):
    if False:
        print('Hello World!')
    "\n    List port forwarding\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.list_port_fwd public\n    "
    ret = []
    cmd = '--zone={} --list-forward-ports'.format(zone)
    if permanent:
        cmd += ' --permanent'
    for i in __firewall_cmd(cmd).splitlines():
        (src, proto, dest, addr) = i.split(':')
        ret.append({'Source port': src.split('=')[1], 'Protocol': proto.split('=')[1], 'Destination port': dest.split('=')[1], 'Destination address': addr.split('=')[1]})
    return ret

def block_icmp(zone, icmp, permanent=True):
    if False:
        i = 10
        return i + 15
    "\n    Block a specific ICMP type on a zone\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.block_icmp zone echo-reply\n    "
    if icmp not in get_icmp_types(permanent):
        log.error('Invalid ICMP type')
        return False
    if icmp in list_icmp_block(zone, permanent):
        log.info('ICMP block already exists')
        return 'success'
    cmd = '--zone={} --add-icmp-block={}'.format(zone, icmp)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def allow_icmp(zone, icmp, permanent=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Allow a specific ICMP type on a zone\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.allow_icmp zone echo-reply\n    "
    if icmp not in get_icmp_types(permanent):
        log.error('Invalid ICMP type')
        return False
    if icmp not in list_icmp_block(zone, permanent):
        log.info('ICMP Type is already permitted')
        return 'success'
    cmd = '--zone={} --remove-icmp-block={}'.format(zone, icmp)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def list_icmp_block(zone, permanent=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    List ICMP blocks on a zone\n\n    .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewlld.list_icmp_block zone\n    "
    cmd = '--zone={} --list-icmp-blocks'.format(zone)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def make_permanent():
    if False:
        while True:
            i = 10
    "\n    Make current runtime configuration permanent.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.make_permanent\n    "
    return __firewall_cmd('--runtime-to-permanent')

def get_interfaces(zone, permanent=True):
    if False:
        print('Hello World!')
    "\n    List interfaces bound to a zone\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_interfaces zone\n    "
    cmd = '--zone={} --list-interfaces'.format(zone)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def add_interface(zone, interface, permanent=True):
    if False:
        print('Hello World!')
    "\n    Bind an interface to a zone\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_interface zone eth0\n    "
    if interface in get_interfaces(zone, permanent):
        log.info('Interface is already bound to zone.')
    cmd = '--zone={} --add-interface={}'.format(zone, interface)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def remove_interface(zone, interface, permanent=True):
    if False:
        while True:
            i = 10
    "\n    Remove an interface bound to a zone\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_interface zone eth0\n    "
    if interface not in get_interfaces(zone, permanent):
        log.info('Interface is not bound to zone.')
    cmd = '--zone={} --remove-interface={}'.format(zone, interface)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def get_sources(zone, permanent=True):
    if False:
        print('Hello World!')
    "\n    List sources bound to a zone\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_sources zone\n    "
    cmd = '--zone={} --list-sources'.format(zone)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).split()

def add_source(zone, source, permanent=True):
    if False:
        while True:
            i = 10
    "\n    Bind a source to a zone\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_source zone 192.168.1.0/24\n    "
    if source in get_sources(zone, permanent):
        log.info('Source is already bound to zone.')
    cmd = '--zone={} --add-source={}'.format(zone, source)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def remove_source(zone, source, permanent=True):
    if False:
        i = 10
        return i + 15
    "\n    Remove a source bound to a zone\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_source zone 192.168.1.0/24\n    "
    if source not in get_sources(zone, permanent):
        log.info('Source is not bound to zone.')
    cmd = '--zone={} --remove-source={}'.format(zone, source)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def get_rich_rules(zone, permanent=True):
    if False:
        return 10
    "\n    List rich rules bound to a zone\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.get_rich_rules zone\n    "
    cmd = '--zone={} --list-rich-rules'.format(zone)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd).splitlines()

def add_rich_rule(zone, rule, permanent=True):
    if False:
        return 10
    "\n    Add a rich rule to a zone\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.add_rich_rule zone 'rule'\n    "
    cmd = "--zone={} --add-rich-rule='{}'".format(zone, rule)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)

def remove_rich_rule(zone, rule, permanent=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add a rich rule to a zone\n\n    .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' firewalld.remove_rich_rule zone 'rule'\n    "
    cmd = "--zone={} --remove-rich-rule='{}'".format(zone, rule)
    if permanent:
        cmd += ' --permanent'
    return __firewall_cmd(cmd)