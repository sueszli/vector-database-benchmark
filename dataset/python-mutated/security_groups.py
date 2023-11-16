import ipaddress
from typing import Any

def check_security_group(ingress_rule: Any, protocol: str, ports: list=[], any_address: bool=False) -> bool:
    if False:
        while True:
            i = 10
    '\n    Check if the security group ingress rule has public access to the check_ports using the protocol\n\n    @param ingress_rule: AWS Security Group IpPermissions Ingress Rule\n    {\n        \'FromPort\': 123,\n        \'IpProtocol\': \'string\',\n        \'IpRanges\': [\n            {\n                \'CidrIp\': \'string\',\n                \'Description\': \'string\'\n            },\n        ],\n        \'Ipv6Ranges\': [\n            {\n                \'CidrIpv6\': \'string\',\n                \'Description\': \'string\'\n            },\n        ],\n        \'ToPort\': 123,\n    }\n\n    @param procotol: Protocol to check.\n\n\n    @param ports: List of ports to check. (Default: [])\n\n    @param any_address: If True, only 0.0.0.0/0 or "::/0" will be public and do not search for public addresses. (Default: False)\n    '
    if ingress_rule['IpProtocol'] == '-1':
        for ip_ingress_rule in ingress_rule['IpRanges']:
            if _is_cidr_public(ip_ingress_rule['CidrIp'], any_address):
                return True
        for ip_ingress_rule in ingress_rule['Ipv6Ranges']:
            if _is_cidr_public(ip_ingress_rule['CidrIpv6'], any_address):
                return True
    if 'FromPort' in ingress_rule:
        if ingress_rule['FromPort'] != ingress_rule['ToPort']:
            diff = ingress_rule['ToPort'] - ingress_rule['FromPort'] + 1
            ingress_port_range = []
            for x in range(diff):
                ingress_port_range.append(int(ingress_rule['FromPort']) + x)
        else:
            ingress_port_range = []
            ingress_port_range.append(int(ingress_rule['FromPort']))
        for ip_ingress_rule in ingress_rule['IpRanges']:
            if _is_cidr_public(ip_ingress_rule['CidrIp'], any_address):
                if ports:
                    for port in ports:
                        if port in ingress_port_range and ingress_rule['IpProtocol'] == protocol:
                            return True
                if len(set(ingress_port_range)) == 65536:
                    return True
        for ip_ingress_rule in ingress_rule['Ipv6Ranges']:
            if _is_cidr_public(ip_ingress_rule['CidrIpv6'], any_address):
                if ports:
                    for port in ports:
                        if port in ingress_port_range and ingress_rule['IpProtocol'] == protocol:
                            return True
                if len(set(ingress_port_range)) == 65536:
                    return True
    return False

def _is_cidr_public(cidr: str, any_address: bool=False) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Check if an input CIDR is public\n\n    @param cidr: CIDR 10.22.33.44/8\n\n    @param any_address: If True, only 0.0.0.0/0 or "::/0" will be public and do not search for public addresses. (Default: False)\n    '
    public_IPv4 = '0.0.0.0/0'
    public_IPv6 = '::/0'
    if cidr in (public_IPv4, public_IPv6):
        return True
    if not any_address:
        return ipaddress.ip_network(cidr).is_global