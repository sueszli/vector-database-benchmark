def accelerator_ipaddresstype(type):
    if False:
        print('Hello World!')
    '\n    Property: Accelerator.IpAddressType\n    '
    valid_types = ['IPV4']
    if type not in valid_types:
        raise ValueError('IpAddressType must be one of: "%s"' % ', '.join(valid_types))
    return type

def endpointgroup_healthcheckprotocol(protocol):
    if False:
        i = 10
        return i + 15
    '\n    Property: EndpointGroup.HealthCheckProtocol\n    '
    valid_protocols = ['HTTP', 'HTTPS', 'TCP']
    if protocol not in valid_protocols:
        raise ValueError('HealthCheckProtocol must be one of: "%s"' % ', '.join(valid_protocols))
    return protocol

def listener_clientaffinity(affinity):
    if False:
        print('Hello World!')
    '\n    Property: Listener.ClientAffinity\n    '
    valid_affinities = ['NONE', 'SOURCE_IP']
    if affinity not in valid_affinities:
        raise ValueError('ClientAffinity must be one of: "%s"' % ', '.join(valid_affinities))
    return affinity

def listener_protocol(protocol):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Listener.Protocol\n    '
    valid_protocols = ['TCP', 'UDP']
    if protocol not in valid_protocols:
        raise ValueError('Protocol must be one of: "%s"' % ', '.join(valid_protocols))
    return protocol