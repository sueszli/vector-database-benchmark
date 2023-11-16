"""
Support for LVS (Linux Virtual Server)
"""
import salt.utils.decorators as decorators
import salt.utils.path
from salt.exceptions import SaltException
__func_alias__ = {'list_': 'list'}

@decorators.memoize
def __detect_os():
    if False:
        return 10
    return salt.utils.path.which('ipvsadm')

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if ipvsadm command exists on the system.\n    '
    if not __detect_os():
        return (False, 'The lvs execution module cannot be loaded: the ipvsadm binary is not in the path.')
    return 'lvs'

def _build_cmd(**kwargs):
    if False:
        while True:
            i = 10
    '\n\n    Build a well-formatted ipvsadm command based on kwargs.\n    '
    cmd = ''
    if 'service_address' in kwargs:
        if kwargs['service_address']:
            if 'protocol' in kwargs:
                if kwargs['protocol'] == 'tcp':
                    cmd += ' -t {}'.format(kwargs['service_address'])
                elif kwargs['protocol'] == 'udp':
                    cmd += ' -u {}'.format(kwargs['service_address'])
                elif kwargs['protocol'] == 'fwmark':
                    cmd += ' -f {}'.format(kwargs['service_address'])
                else:
                    raise SaltException('Error: Only support tcp, udp and fwmark service protocol')
                del kwargs['protocol']
            else:
                raise SaltException('Error: protocol should specified')
            if 'scheduler' in kwargs:
                if kwargs['scheduler']:
                    cmd += ' -s {}'.format(kwargs['scheduler'])
                    del kwargs['scheduler']
        else:
            raise SaltException('Error: service_address should specified')
        del kwargs['service_address']
    if 'server_address' in kwargs:
        if kwargs['server_address']:
            cmd += ' -r {}'.format(kwargs['server_address'])
            if 'packet_forward_method' in kwargs and kwargs['packet_forward_method']:
                if kwargs['packet_forward_method'] == 'dr':
                    cmd += ' -g'
                elif kwargs['packet_forward_method'] == 'tunnel':
                    cmd += ' -i'
                elif kwargs['packet_forward_method'] == 'nat':
                    cmd += ' -m'
                else:
                    raise SaltException('Error: only support dr, tunnel and nat')
                del kwargs['packet_forward_method']
            if 'weight' in kwargs and kwargs['weight']:
                cmd += ' -w {}'.format(kwargs['weight'])
                del kwargs['weight']
        else:
            raise SaltException('Error: server_address should specified')
        del kwargs['server_address']
    return cmd

def add_service(protocol=None, service_address=None, scheduler='wlc'):
    if False:
        print('Hello World!')
    "\n    Add a virtual service.\n\n    protocol\n        The service protocol(only support tcp, udp and fwmark service).\n\n    service_address\n        The LVS service address.\n\n    scheduler\n        Algorithm for allocating TCP connections and UDP datagrams to real servers.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.add_service tcp 1.1.1.1:80 rr\n    "
    cmd = '{} -A {}'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address, scheduler=scheduler))
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def edit_service(protocol=None, service_address=None, scheduler=None):
    if False:
        while True:
            i = 10
    "\n    Edit the virtual service.\n\n    protocol\n        The service protocol(only support tcp, udp and fwmark service).\n\n    service_address\n        The LVS service address.\n\n    scheduler\n        Algorithm for allocating TCP connections and UDP datagrams to real servers.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.edit_service tcp 1.1.1.1:80 rr\n    "
    cmd = '{} -E {}'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address, scheduler=scheduler))
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def delete_service(protocol=None, service_address=None):
    if False:
        i = 10
        return i + 15
    "\n\n    Delete the virtual service.\n\n    protocol\n        The service protocol(only support tcp, udp and fwmark service).\n\n    service_address\n        The LVS service address.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.delete_service tcp 1.1.1.1:80\n    "
    cmd = '{} -D {}'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address))
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def add_server(protocol=None, service_address=None, server_address=None, packet_forward_method='dr', weight=1, **kwargs):
    if False:
        return 10
    "\n\n    Add a real server to a virtual service.\n\n    protocol\n        The service protocol(only support ``tcp``, ``udp`` and ``fwmark`` service).\n\n    service_address\n        The LVS service address.\n\n    server_address\n        The real server address.\n\n    packet_forward_method\n        The LVS packet forwarding method(``dr`` for direct routing, ``tunnel`` for tunneling, ``nat`` for network access translation).\n\n    weight\n        The capacity  of a server relative to the others in the pool.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.add_server tcp 1.1.1.1:80 192.168.0.11:8080 nat 1\n    "
    cmd = '{} -a {}'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address, server_address=server_address, packet_forward_method=packet_forward_method, weight=weight, **kwargs))
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def edit_server(protocol=None, service_address=None, server_address=None, packet_forward_method=None, weight=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n\n    Edit a real server to a virtual service.\n\n    protocol\n        The service protocol(only support ``tcp``, ``udp`` and ``fwmark`` service).\n\n    service_address\n        The LVS service address.\n\n    server_address\n        The real server address.\n\n    packet_forward_method\n        The LVS packet forwarding method(``dr`` for direct routing, ``tunnel`` for tunneling, ``nat`` for network access translation).\n\n    weight\n        The capacity  of a server relative to the others in the pool.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.edit_server tcp 1.1.1.1:80 192.168.0.11:8080 nat 1\n    "
    cmd = '{} -e {}'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address, server_address=server_address, packet_forward_method=packet_forward_method, weight=weight, **kwargs))
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def delete_server(protocol=None, service_address=None, server_address=None):
    if False:
        i = 10
        return i + 15
    "\n\n    Delete the realserver from the virtual service.\n\n    protocol\n        The service protocol(only support ``tcp``, ``udp`` and ``fwmark`` service).\n\n    service_address\n        The LVS service address.\n\n    server_address\n        The real server address.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.delete_server tcp 1.1.1.1:80 192.168.0.11:8080\n    "
    cmd = '{} -d {}'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address, server_address=server_address))
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def clear():
    if False:
        print('Hello World!')
    "\n\n    Clear the virtual server table\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.clear\n    "
    cmd = '{} -C'.format(__detect_os())
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def get_rules():
    if False:
        while True:
            i = 10
    "\n\n    Get the virtual server rules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.get_rules\n    "
    cmd = '{} -S -n'.format(__detect_os())
    ret = __salt__['cmd.run'](cmd, python_shell=False)
    return ret

def list_(protocol=None, service_address=None):
    if False:
        while True:
            i = 10
    "\n\n    List the virtual server table if service_address is not specified. If a service_address is selected, list this service only.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.list\n    "
    if service_address:
        cmd = '{} -L {} -n'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address))
    else:
        cmd = '{} -L -n'.format(__detect_os())
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = out['stdout'].strip()
    return ret

def zero(protocol=None, service_address=None):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Zero the packet, byte and rate counters in a service or all services.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.zero\n    "
    if service_address:
        cmd = '{} -Z {}'.format(__detect_os(), _build_cmd(protocol=protocol, service_address=service_address))
    else:
        cmd = '{} -Z'.format(__detect_os())
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode']:
        ret = out['stderr'].strip()
    else:
        ret = True
    return ret

def check_service(protocol=None, service_address=None, **kwargs):
    if False:
        return 10
    "\n\n    Check the virtual service exists.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvs.check_service tcp 1.1.1.1:80\n    "
    cmd = '{}'.format(_build_cmd(protocol=protocol, service_address=service_address, **kwargs))
    if not kwargs:
        cmd += ' '
    all_rules = get_rules()
    out = all_rules.find(cmd)
    if out != -1:
        ret = True
    else:
        ret = 'Error: service not exists'
    return ret

def check_server(protocol=None, service_address=None, server_address=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n\n    Check the real server exists in the specified service.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' lvs.check_server tcp 1.1.1.1:80 192.168.0.11:8080\n    "
    cmd = '{}'.format(_build_cmd(protocol=protocol, service_address=service_address, server_address=server_address, **kwargs))
    if not kwargs:
        cmd += ' '
    all_rules = get_rules()
    out = all_rules.find(cmd)
    if out != -1:
        ret = True
    else:
        ret = 'Error: server not exists'
    return ret