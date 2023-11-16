"""
Support for Open vSwitch - module with basic Open vSwitch commands.

Suitable for setting up Openstack Neutron.

:codeauthor: Jiri Kotlin <jiri.kotlin@ultimum.io>
"""
import logging
import salt.utils.path
from salt.exceptions import ArgumentValueError, CommandExecutionError
from salt.utils import json
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load the module if Open vSwitch is installed\n    '
    if salt.utils.path.which('ovs-vsctl'):
        return 'openvswitch'
    return (False, 'Missing dependency: ovs-vsctl')

def _param_may_exist(may_exist):
    if False:
        i = 10
        return i + 15
    "\n    Returns --may-exist parameter for Open vSwitch command.\n\n    Args:\n        may_exist: Boolean whether to use this parameter.\n\n    Returns:\n        String '--may-exist ' or empty string.\n    "
    if may_exist:
        return '--may-exist '
    else:
        return ''

def _param_if_exists(if_exists):
    if False:
        i = 10
        return i + 15
    "\n    Returns --if-exist parameter for Open vSwitch command.\n\n    Args:\n        if_exists: Boolean whether to use this parameter.\n\n    Returns:\n        String '--if-exist ' or empty string.\n    "
    if if_exists:
        return '--if-exists '
    else:
        return ''

def _retcode_to_bool(retcode):
    if False:
        return 10
    '\n    Evaulates Open vSwitch command`s retcode value.\n\n    Args:\n        retcode: Value of retcode field from response, should be 0, 1 or 2.\n\n    Returns:\n        True on 0, else False\n    '
    if retcode == 0:
        return True
    else:
        return False

def _stdout_list_split(retcode, stdout='', splitstring='\n'):
    if False:
        print('Hello World!')
    '\n    Evaulates Open vSwitch command`s retcode value.\n\n    Args:\n        retcode: Value of retcode field from response, should be 0, 1 or 2.\n        stdout: Value of stdout filed from response.\n        splitstring: String used to split the stdout default new line.\n\n    Returns:\n        List or False.\n    '
    if retcode == 0:
        ret = stdout.split(splitstring)
        return ret
    else:
        return False

def _convert_json(obj):
    if False:
        while True:
            i = 10
    '\n    Converts from the JSON output provided by ovs-vsctl into a usable Python\n    object tree. In particular, sets and maps are converted from lists to\n    actual sets or maps.\n\n    Args:\n        obj: Object that shall be recursively converted.\n\n    Returns:\n        Converted version of object.\n    '
    if isinstance(obj, dict):
        return {_convert_json(key): _convert_json(val) for (key, val) in obj.items()}
    elif isinstance(obj, list) and len(obj) == 2:
        first = obj[0]
        second = obj[1]
        if first == 'set' and isinstance(second, list):
            return [_convert_json(elem) for elem in second]
        elif first == 'map' and isinstance(second, list):
            for elem in second:
                if not isinstance(elem, list) or len(elem) != 2:
                    return obj
            return {elem[0]: _convert_json(elem[1]) for elem in second}
        else:
            return obj
    elif isinstance(obj, list):
        return [_convert_json(elem) for elem in obj]
    else:
        return obj

def _stdout_parse_json(stdout):
    if False:
        while True:
            i = 10
    '\n    Parses JSON output from ovs-vsctl and returns the corresponding object\n    tree.\n\n    Args:\n        stdout: Output that shall be parsed.\n\n    Returns:\n        Object represented by the output.\n    '
    obj = json.loads(stdout)
    return _convert_json(obj)

def bridge_list():
    if False:
        print('Hello World!')
    "\n    Lists all existing real and fake bridges.\n\n    Returns:\n        List of bridges (or empty list), False on failure.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.bridge_list\n    "
    cmd = 'ovs-vsctl list-br'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    stdout = result['stdout']
    return _stdout_list_split(retcode, stdout)

def bridge_exists(br):
    if False:
        print('Hello World!')
    "\n    Tests whether bridge exists as a real or fake  bridge.\n\n    Returns:\n        True if Bridge exists, else False.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.bridge_exists br0\n    "
    cmd = f'ovs-vsctl br-exists {br}'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    return _retcode_to_bool(retcode)

def bridge_create(br, may_exist=True, parent=None, vlan=None):
    if False:
        i = 10
        return i + 15
    "\n    Creates a new bridge.\n\n    Args:\n        br : string\n            bridge name\n        may_exist : bool\n            if False - attempting to create a bridge that exists returns False.\n        parent : string\n            name of the parent bridge (if the bridge shall be created as a fake\n            bridge). If specified, vlan must also be specified.\n        .. versionadded:: 3006.0\n        vlan : int\n            VLAN ID of the bridge (if the bridge shall be created as a fake\n            bridge). If specified, parent must also be specified.\n        .. versionadded:: 3006.0\n\n    Returns:\n        True on success, else False.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.bridge_create br0\n    "
    param_may_exist = _param_may_exist(may_exist)
    if parent is not None and vlan is None:
        raise ArgumentValueError('If parent is specified, vlan must also be specified.')
    if vlan is not None and parent is None:
        raise ArgumentValueError('If vlan is specified, parent must also be specified.')
    param_parent = '' if parent is None else f' {parent}'
    param_vlan = '' if vlan is None else f' {vlan}'
    cmd = 'ovs-vsctl {1}add-br {0}{2}{3}'.format(br, param_may_exist, param_parent, param_vlan)
    result = __salt__['cmd.run_all'](cmd)
    return _retcode_to_bool(result['retcode'])

def bridge_delete(br, if_exists=True):
    if False:
        print('Hello World!')
    "\n    Deletes bridge and all of  its  ports.\n\n    Args:\n        br: A string - bridge name\n        if_exists: Bool, if False - attempting to delete a bridge that does not exist returns False.\n\n    Returns:\n        True on success, else False.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.bridge_delete br0\n    "
    param_if_exists = _param_if_exists(if_exists)
    cmd = f'ovs-vsctl {param_if_exists}del-br {br}'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    return _retcode_to_bool(retcode)

def bridge_to_parent(br):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 3006.0\n\n    Returns the parent bridge of a bridge.\n\n    Args:\n        br : string\n            bridge name\n\n    Returns:\n        Name of the parent bridge. This is the same as the bridge name if the\n        bridge is not a fake bridge. If the bridge does not exist, False is\n        returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.bridge_to_parent br0\n    "
    cmd = f'ovs-vsctl br-to-parent {br}'
    result = __salt__['cmd.run_all'](cmd)
    if result['retcode'] != 0:
        return False
    return result['stdout'].strip()

def bridge_to_vlan(br):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 3006.0\n\n    Returns the VLAN ID of a bridge.\n\n    Args:\n        br : string\n            bridge name\n\n    Returns:\n        VLAN ID of the bridge. The VLAN ID is 0 if the bridge is not a fake\n        bridge.  If the bridge does not exist, False is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.bridge_to_parent br0\n    "
    cmd = f'ovs-vsctl br-to-vlan {br}'
    result = __salt__['cmd.run_all'](cmd)
    if result['retcode'] != 0:
        return False
    return int(result['stdout'])

def port_add(br, port, may_exist=False, internal=False):
    if False:
        i = 10
        return i + 15
    "\n    Creates on bridge a new port named port.\n\n    Returns:\n        True on success, else False.\n\n    Args:\n        br: A string - bridge name\n        port: A string - port name\n        may_exist: Bool, if False - attempting to create a port that exists returns False.\n        internal: A boolean to create an internal interface if one does not exist.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.port_add br0 8080\n    "
    param_may_exist = _param_may_exist(may_exist)
    cmd = f'ovs-vsctl {param_may_exist}add-port {br} {port}'
    if internal:
        cmd += f' -- set interface {port} type=internal'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    return _retcode_to_bool(retcode)

def port_remove(br, port, if_exists=True):
    if False:
        print('Hello World!')
    "\n     Deletes port.\n\n    Args:\n        br: A string - bridge name (If bridge is None, port is removed from  whatever bridge contains it)\n        port: A string - port name.\n        if_exists: Bool, if False - attempting to delete a por that  does  not exist returns False. (Default True)\n\n    Returns:\n        True on success, else False.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.port_remove br0 8080\n    "
    param_if_exists = _param_if_exists(if_exists)
    if port and (not br):
        cmd = f'ovs-vsctl {param_if_exists}del-port {port}'
    else:
        cmd = f'ovs-vsctl {param_if_exists}del-port {br} {port}'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    return _retcode_to_bool(retcode)

def port_list(br):
    if False:
        print('Hello World!')
    "\n    Lists all of the ports within bridge.\n\n    Args:\n        br: A string - bridge name.\n\n    Returns:\n        List of bridges (or empty list), False on failure.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.port_list br0\n    "
    cmd = f'ovs-vsctl list-ports {br}'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    stdout = result['stdout']
    return _stdout_list_split(retcode, stdout)

def port_get_tag(port):
    if False:
        return 10
    "\n    Lists tags of the port.\n\n    Args:\n        port: A string - port name.\n\n    Returns:\n        List of tags (or empty list), False on failure.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.port_get_tag tap0\n    "
    cmd = f'ovs-vsctl get port {port} tag'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    stdout = result['stdout']
    return _stdout_list_split(retcode, stdout)

def interface_get_options(port):
    if False:
        return 10
    "\n    Port's interface's optional parameters.\n\n    Args:\n        port: A string - port name.\n\n    Returns:\n        String containing optional parameters of port's interface, False on failure.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.interface_get_options tap0\n    "
    cmd = f'ovs-vsctl get interface {port} options'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    stdout = result['stdout']
    return _stdout_list_split(retcode, stdout)

def interface_get_type(port):
    if False:
        i = 10
        return i + 15
    "\n    Type of port's interface.\n\n    Args:\n        port: A string - port name.\n\n    Returns:\n        String - type of interface or empty string, False on failure.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' openvswitch.interface_get_type tap0\n    "
    cmd = f'ovs-vsctl get interface {port} type'
    result = __salt__['cmd.run_all'](cmd)
    retcode = result['retcode']
    stdout = result['stdout']
    return _stdout_list_split(retcode, stdout)

def port_create_vlan(br, port, id, internal=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Isolate VM traffic using VLANs.\n\n    Args:\n        br: A string - bridge name.\n        port: A string - port name.\n        id: An integer in the valid range 0 to 4095 (inclusive), name of VLAN.\n        internal: A boolean to create an internal interface if one does not exist.\n\n    Returns:\n        True on success, else False.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' openvswitch.port_create_vlan br0 tap0 100\n    "
    interfaces = __salt__['network.interfaces']()
    if not 0 <= id <= 4095:
        return False
    elif not bridge_exists(br):
        return False
    elif not internal and port not in interfaces:
        return False
    elif port in port_list(br):
        cmd = f'ovs-vsctl set port {port} tag={id}'
        if internal:
            cmd += f' -- set interface {port} type=internal'
        result = __salt__['cmd.run_all'](cmd)
        return _retcode_to_bool(result['retcode'])
    else:
        cmd = f'ovs-vsctl add-port {br} {port} tag={id}'
        if internal:
            cmd += f' -- set interface {port} type=internal'
        result = __salt__['cmd.run_all'](cmd)
        return _retcode_to_bool(result['retcode'])

def port_create_gre(br, port, id, remote):
    if False:
        print('Hello World!')
    "\n    Generic Routing Encapsulation - creates GRE tunnel between endpoints.\n\n    Args:\n        br: A string - bridge name.\n        port: A string - port name.\n        id: An integer - unsigned 32-bit number, tunnel's key.\n        remote: A string - remote endpoint's IP address.\n\n    Returns:\n        True on success, else False.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' openvswitch.port_create_gre br0 gre1 5001 192.168.1.10\n    "
    if not 0 <= id < 2 ** 32:
        return False
    elif not __salt__['dig.check_ip'](remote):
        return False
    elif not bridge_exists(br):
        return False
    elif port in port_list(br):
        cmd = 'ovs-vsctl set interface {} type=gre options:remote_ip={} options:key={}'.format(port, remote, id)
        result = __salt__['cmd.run_all'](cmd)
        return _retcode_to_bool(result['retcode'])
    else:
        cmd = 'ovs-vsctl add-port {0} {1} -- set interface {1} type=gre options:remote_ip={2} options:key={3}'.format(br, port, remote, id)
        result = __salt__['cmd.run_all'](cmd)
        return _retcode_to_bool(result['retcode'])

def port_create_vxlan(br, port, id, remote, dst_port=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Virtual eXtensible Local Area Network - creates VXLAN tunnel between endpoints.\n\n    Args:\n        br: A string - bridge name.\n        port: A string - port name.\n        id: An integer - unsigned 64-bit number, tunnel's key.\n        remote: A string - remote endpoint's IP address.\n        dst_port: An integer - port to use when creating tunnelport in the switch.\n\n    Returns:\n        True on success, else False.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' openvswitch.port_create_vxlan br0 vx1 5001 192.168.1.10 8472\n    "
    dst_port = ' options:dst_port=' + str(dst_port) if 0 < dst_port <= 65535 else ''
    if not 0 <= id < 2 ** 64:
        return False
    elif not __salt__['dig.check_ip'](remote):
        return False
    elif not bridge_exists(br):
        return False
    elif port in port_list(br):
        cmd = 'ovs-vsctl set interface {} type=vxlan options:remote_ip={} options:key={}{}'.format(port, remote, id, dst_port)
        result = __salt__['cmd.run_all'](cmd)
        return _retcode_to_bool(result['retcode'])
    else:
        cmd = 'ovs-vsctl add-port {0} {1} -- set interface {1} type=vxlan options:remote_ip={2} options:key={3}{4}'.format(br, port, remote, id, dst_port)
        result = __salt__['cmd.run_all'](cmd)
        return _retcode_to_bool(result['retcode'])

def db_get(table, record, column, if_exists=False):
    if False:
        return 10
    "\n    .. versionadded:: 3006.0\n\n    Gets a column's value for a specific record.\n\n    Args:\n        table : string\n            name of the database table\n        record : string\n            identifier of the record\n        column : string\n            name of the column\n        if_exists : boolean\n            if True, it is not an error if the record does not exist.\n\n    Returns:\n        The column's value.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' openvswitch.db_get Port br0 vlan_mode\n    "
    cmd = ['ovs-vsctl', '--format=json', f'--columns={column}']
    if if_exists:
        cmd += ['--if-exists']
    cmd += ['list', table, record]
    result = __salt__['cmd.run_all'](cmd)
    if result['retcode'] != 0:
        raise CommandExecutionError(result['stderr'])
    output = _stdout_parse_json(result['stdout'])
    if output['data'] and output['data'][0]:
        return output['data'][0][0]
    else:
        return None

def db_set(table, record, column, value, if_exists=False):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3006.0\n\n    Sets a column's value for a specific record.\n\n    Args:\n        table : string\n            name of the database table\n        record : string\n            identifier of the record\n        column : string\n            name of the column\n        value : string\n            the value to be set\n        if_exists : boolean\n            if True, it is not an error if the record does not exist.\n\n    Returns:\n        None on success and an error message on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' openvswitch.db_set Interface br0 mac 02:03:04:05:06:07\n    "
    cmd = ['ovs-vsctl']
    if if_exists:
        cmd += ['--if-exists']
    cmd += ['set', table, record, f'{column}={json.dumps(value)}']
    result = __salt__['cmd.run_all'](cmd)
    if result['retcode'] != 0:
        return result['stderr']
    else:
        return None