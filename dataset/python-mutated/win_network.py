"""
This salt util uses WMI to gather network information on Windows 7 and .NET 4.0+
on newer systems.
The reason for this is that calls to WMI tend to be slower. Especially if the
query has not been optimized. For example, timing to gather NIC info from WMI
and .NET were as follows in testing:
WMI: 3.4169998168945312 seconds
NET: 1.0390000343322754 seconds
Since this is used to generate grain information we want to avoid using WMI as
much as possible.
There are 3 functions in this salt util.
- get_interface_info_dot_net
- get_interface_info_wmi
- get_interface_info
The ``get_interface_info`` function will call one of the other two functions
depending on the version of Windows this is run on. Once support for Windows
7 is dropped we can remove the WMI stuff and just use .NET.
:depends: - pythonnet
          - wmi
"""
import logging
import platform
import salt.utils.win_reg
from salt._compat import ipaddress
IS_WINDOWS = platform.system() == 'Windows'
log = logging.getLogger(__name__)
__virtualname__ = 'win_network'
if IS_WINDOWS:
    net_release = salt.utils.win_reg.read_value(hive='HKLM', key='SOFTWARE\\Microsoft\\NET Framework Setup\\NDP\\v4\\Full', vname='Release')
    if not net_release['success'] or net_release['vdata'] < 461808:
        USE_WMI = True
    else:
        USE_WMI = False
    if USE_WMI:
        import wmi
        import salt.utils.winapi
    else:
        try:
            import clr
            from System.Net import NetworkInformation
        except RuntimeError:
            log.debug('Failed to load pythonnet. Falling back to WMI')
            USE_WMI = True
            import wmi
            import salt.utils.winapi
enum_adapter_types = {1: 'Unknown', 6: 'Ethernet', 9: 'TokenRing', 15: 'FDDI', 20: 'BasicISDN', 21: 'PrimaryISDN', 23: 'PPP', 24: 'Loopback', 26: 'Ethernet3Megabit', 28: 'Slip', 37: 'ATM', 48: 'GenericModem', 53: 'TAPAdapter', 62: 'FastEthernetT', 63: 'ISDN', 69: 'FastEthernetFx', 71: 'Wireless802.11', 94: 'AsymmetricDSL', 95: 'RateAdaptDSL', 96: 'SymmetricDSL', 97: 'VeryHighSpeedDSL', 114: 'IPOverATM', 117: 'GigabitEthernet', 131: 'Tunnel', 143: 'MultiRateSymmetricDSL', 144: 'HighPerformanceSerialBus', 237: 'WMAN', 243: 'WWANPP', 244: 'WWANPP2'}
enum_operational_status = {1: 'Up', 2: 'Down', 3: 'Testing', 4: 'Unknown', 5: 'Dormant', 6: 'NotPresent', 7: 'LayerDown'}
enum_prefix_suffix = {0: 'Other', 1: 'Manual', 2: 'WellKnown', 3: 'DHCP', 4: 'Router', 5: 'Random'}
af_inet = 2
af_inet6 = 23

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if windows\n    '
    if not IS_WINDOWS:
        return (False, 'This utility will only run on Windows')
    return __virtualname__

def _get_base_properties(i_face):
    if False:
        print('Hello World!')
    raw_mac = i_face.GetPhysicalAddress().ToString()
    try:
        i_face_type = i_face.NetworkInterfaceType.ToString()
    except AttributeError:
        try:
            i_face_type = enum_adapter_types[i_face.NetworkInterfaceType]
        except KeyError:
            i_face_type = i_face.Description
    ret = {'alias': i_face.Name, 'description': i_face.Description, 'id': i_face.Id, 'receive_only': i_face.IsReceiveOnly, 'type': i_face_type, 'physical_address': ':'.join((raw_mac[i:i + 2] for i in range(0, 12, 2)))}
    try:
        ret['status'] = i_face.OperationalStatus.ToString()
    except AttributeError:
        ret['status'] = enum_operational_status[i_face.OperationalStatus]
    return ret

def _get_ip_base_properties(i_face):
    if False:
        while True:
            i = 10
    ip_properties = i_face.GetIPProperties()
    return {'dns_suffix': ip_properties.DnsSuffix, 'dns_enabled': ip_properties.IsDnsEnabled, 'dynamic_dns_enabled': ip_properties.IsDynamicDnsEnabled}

def _get_ip_unicast_info(i_face):
    if False:
        for i in range(10):
            print('nop')
    ip_properties = i_face.GetIPProperties()
    int_dict = {}
    if ip_properties.UnicastAddresses.Count > 0:
        names = {af_inet: 'ip_addresses', af_inet6: 'ipv6_addresses'}
        for addrs in ip_properties.UnicastAddresses:
            try:
                if addrs.Address.AddressFamily.ToString() == 'InterNetwork':
                    family = 2
                else:
                    family = 23
            except AttributeError:
                family = addrs.Address.AddressFamily
            if family == af_inet:
                ip = addrs.Address.ToString()
                mask = addrs.IPv4Mask.ToString()
                net = ipaddress.IPv4Network(ip + '/' + mask, False)
                ip_info = {'address': ip, 'netmask': mask, 'broadcast': net.broadcast_address.compressed, 'loopback': addrs.Address.Loopback.ToString()}
            else:
                ip_info = {'address': addrs.Address.ToString().split('%')[0], 'interface_index': int(addrs.Address.ScopeId)}
            ip_info.update({'prefix_length': addrs.PrefixLength})
            try:
                ip_info.update({'prefix_origin': addrs.PrefixOrigin.ToString()})
            except AttributeError:
                ip_info.update({'prefix_origin': enum_prefix_suffix[addrs.PrefixOrigin]})
            try:
                ip_info.update({'suffix_origin': addrs.SuffixOrigin.ToString()})
            except AttributeError:
                ip_info.update({'suffix_origin': enum_prefix_suffix[addrs.SuffixOrigin]})
            int_dict.setdefault(names[family], []).append(ip_info)
    return int_dict

def _get_ip_gateway_info(i_face):
    if False:
        print('Hello World!')
    ip_properties = i_face.GetIPProperties()
    int_dict = {}
    if ip_properties.GatewayAddresses.Count > 0:
        names = {af_inet: 'ip_gateways', af_inet6: 'ipv6_gateways'}
        for addrs in ip_properties.GatewayAddresses:
            try:
                if addrs.Address.AddressFamily.ToString() == 'InterNetwork':
                    family = 2
                else:
                    family = 23
            except AttributeError:
                family = addrs.Address.AddressFamily
            int_dict.setdefault(names[family], []).append(addrs.Address.ToString().split('%')[0])
    return int_dict

def _get_ip_dns_info(i_face):
    if False:
        while True:
            i = 10
    ip_properties = i_face.GetIPProperties()
    int_dict = {}
    if ip_properties.DnsAddresses.Count > 0:
        names = {af_inet: 'ip_dns', af_inet6: 'ipv6_dns'}
        for addrs in ip_properties.DnsAddresses:
            try:
                if addrs.AddressFamily.ToString() == 'InterNetwork':
                    family = 2
                else:
                    family = 23
            except AttributeError:
                family = addrs.AddressFamily
            int_dict.setdefault(names[family], []).append(addrs.ToString().split('%')[0])
    return int_dict

def _get_ip_multicast_info(i_face):
    if False:
        i = 10
        return i + 15
    ip_properties = i_face.GetIPProperties()
    int_dict = {}
    if ip_properties.MulticastAddresses.Count > 0:
        names = {af_inet: 'ip_multicast', af_inet6: 'ipv6_multicast'}
        for addrs in ip_properties.MulticastAddresses:
            try:
                if addrs.Address.AddressFamily.ToString() == 'InterNetwork':
                    family = 2
                else:
                    family = 23
            except AttributeError:
                family = addrs.Address.AddressFamily
            int_dict.setdefault(names[family], []).append(addrs.Address.ToString().split('%')[0])
    return int_dict

def _get_ip_anycast_info(i_face):
    if False:
        print('Hello World!')
    ip_properties = i_face.GetIPProperties()
    int_dict = {}
    if ip_properties.AnycastAddresses.Count > 0:
        names = {af_inet: 'ip_anycast', af_inet6: 'ipv6_anycast'}
        for addrs in ip_properties.AnycastAddresses:
            try:
                if addrs.Address.AddressFamily.ToString() == 'InterNetwork':
                    family = 2
                else:
                    family = 23
            except AttributeError:
                family = addrs.Address.AddressFamily
            int_dict.setdefault(names[family], []).append(addrs.Address.ToString())
    return int_dict

def _get_ip_wins_info(i_face):
    if False:
        i = 10
        return i + 15
    ip_properties = i_face.GetIPProperties()
    int_dict = {}
    if ip_properties.WinsServersAddresses.Count > 0:
        for addrs in ip_properties.WinsServersAddresses:
            int_dict.setdefault('ip_wins', []).append(addrs.ToString())
    return int_dict

def _get_network_interfaces():
    if False:
        for i in range(10):
            print('nop')
    return NetworkInformation.NetworkInterface.GetAllNetworkInterfaces()

def get_interface_info_dot_net_formatted():
    if False:
        i = 10
        return i + 15
    '\n    Returns data gathered via ``get_interface_info_dot_net`` and returns the\n    info in the same manner as ``get_interface_info_wmi``\n\n    Returns:\n        dict: A dictionary of information about all interfaces on the system\n    '
    interfaces = get_interface_info_dot_net()
    i_faces = {}
    for i_face in interfaces:
        if interfaces[i_face]['status'] == 'Up':
            name = interfaces[i_face]['description']
            i_faces.setdefault(name, {}).update({'hwaddr': interfaces[i_face]['physical_address'], 'up': True})
            for ip in interfaces[i_face].get('ip_addresses', []):
                i_faces[name].setdefault('inet', []).append({'address': ip['address'], 'broadcast': ip['broadcast'], 'netmask': ip['netmask'], 'gateway': interfaces[i_face].get('ip_gateways', [''])[0], 'label': name})
            for ip in interfaces[i_face].get('ipv6_addresses', []):
                i_faces[name].setdefault('inet6', []).append({'address': ip['address'], 'gateway': interfaces[i_face].get('ipv6_gateways', [''])[0], 'prefixlen': ip['prefix_length']})
    return i_faces

def get_interface_info_dot_net():
    if False:
        while True:
            i = 10
    '\n    Uses .NET 4.0+ to gather Network Interface information. Should only run on\n    Windows systems newer than Windows 7/Server 2008R2\n\n    Returns:\n        dict: A dictionary of information about all interfaces on the system\n    '
    interfaces = {}
    for i_face in _get_network_interfaces():
        temp_dict = _get_base_properties(i_face)
        temp_dict.update(_get_ip_base_properties(i_face))
        temp_dict.update(_get_ip_unicast_info(i_face))
        temp_dict.update(_get_ip_gateway_info(i_face))
        temp_dict.update(_get_ip_dns_info(i_face))
        temp_dict.update(_get_ip_multicast_info(i_face))
        temp_dict.update(_get_ip_anycast_info(i_face))
        temp_dict.update(_get_ip_wins_info(i_face))
        interfaces[i_face.Name] = temp_dict
    return interfaces

def get_interface_info_wmi():
    if False:
        print('Hello World!')
    '\n    Uses WMI to gather Network Interface information. Should only run on\n    Windows 7/2008 R2 and lower systems. This code was pulled from the\n    ``win_interfaces`` function in ``salt.utils.network`` unchanged.\n\n    Returns:\n        dict: A dictionary of information about all interfaces on the system\n    '
    with salt.utils.winapi.Com():
        c = wmi.WMI()
        i_faces = {}
        for i_face in c.Win32_NetworkAdapterConfiguration(IPEnabled=1):
            i_faces[i_face.Description] = {}
            if i_face.MACAddress:
                i_faces[i_face.Description]['hwaddr'] = i_face.MACAddress
            if i_face.IPEnabled:
                i_faces[i_face.Description]['up'] = True
                for ip in i_face.IPAddress:
                    if '.' in ip:
                        if 'inet' not in i_faces[i_face.Description]:
                            i_faces[i_face.Description]['inet'] = []
                        item = {'address': ip, 'label': i_face.Description}
                        if i_face.DefaultIPGateway:
                            broadcast = next((i for i in i_face.DefaultIPGateway if '.' in i), '')
                            if broadcast:
                                item['broadcast'] = broadcast
                        if i_face.IPSubnet:
                            netmask = next((i for i in i_face.IPSubnet if '.' in i), '')
                            if netmask:
                                item['netmask'] = netmask
                        i_faces[i_face.Description]['inet'].append(item)
                    if ':' in ip:
                        if 'inet6' not in i_faces[i_face.Description]:
                            i_faces[i_face.Description]['inet6'] = []
                        item = {'address': ip}
                        if i_face.DefaultIPGateway:
                            broadcast = next((i for i in i_face.DefaultIPGateway if ':' in i), '')
                            if broadcast:
                                item['broadcast'] = broadcast
                        if i_face.IPSubnet:
                            prefixlen = next((int(i) for i in i_face.IPSubnet if '.' not in i), None)
                            if prefixlen:
                                item['prefixlen'] = prefixlen
                        i_faces[i_face.Description]['inet6'].append(item)
            else:
                i_faces[i_face.Description]['up'] = False
    return i_faces

def get_interface_info():
    if False:
        print('Hello World!')
    '\n    This function will return network interface information for the system and\n    will use the best method to retrieve that information. Windows 7/2008R2 and\n    below will use WMI. Newer systems will use .NET.\n    Returns:\n        dict: A dictionary of information about the Network interfaces\n    '
    if USE_WMI:
        return get_interface_info_wmi()
    else:
        return get_interface_info_dot_net_formatted()