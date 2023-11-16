from typing import Any

def check_network_acl(rules: Any, protocol: str, port: int) -> bool:
    if False:
        i = 10
        return i + 15
    'check_network_acl returns True if the network acls rules has ingress public access to the check_ports using the protocol, otherwise return False\n    - True --> NACL open to the internet\n    - False --> NACL closed to the internet\n    '
    rules_IPv6 = list(filter(lambda rule: rule.get('CidrBlock') is None and (not rule['Egress']), rules))
    for rule in sorted(rules_IPv6, key=lambda rule: rule['RuleNumber']):
        if rule['Ipv6CidrBlock'] == '::/0' and rule['RuleAction'] == 'deny' and (rule['Protocol'] == '-1' or (rule['Protocol'] == protocol and rule['PortRange']['From'] <= port <= rule['PortRange']['To'])):
            break
        if rule['Ipv6CidrBlock'] == '::/0' and rule['RuleAction'] == 'allow' and (rule['Protocol'] == '-1' or (rule['Protocol'] == protocol and rule['PortRange']['From'] <= port <= rule['PortRange']['To'])):
            return True
    rules_IPv4 = list(filter(lambda rule: rule.get('Ipv6CidrBlock') is None and (not rule['Egress']), rules))
    for rule in sorted(rules_IPv4, key=lambda rule: rule['RuleNumber']):
        if rule['CidrBlock'] == '0.0.0.0/0' and rule['RuleAction'] == 'deny' and (rule['Protocol'] == '-1' or (rule['Protocol'] == protocol and rule['PortRange']['From'] <= port <= rule['PortRange']['To'])):
            return False
        if rule['CidrBlock'] == '0.0.0.0/0' and rule['RuleAction'] == 'allow' and (rule['Protocol'] == '-1' or (rule['Protocol'] == protocol and rule['PortRange']['From'] <= port <= rule['PortRange']['To'])):
            return True
    return False