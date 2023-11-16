import re
PORT_SPEC = re.compile('^((\\[?(?P<host>[a-fA-F\\d.:]+)\\]?:)?(?P<ext>[\\d]*)(-(?P<ext_end>[\\d]+))?:)?(?P<int>[\\d]+)(-(?P<int_end>[\\d]+))?(?P<proto>/(udp|tcp|sctp))?$')

def add_port_mapping(port_bindings, internal_port, external):
    if False:
        return 10
    if internal_port in port_bindings:
        port_bindings[internal_port].append(external)
    else:
        port_bindings[internal_port] = [external]

def add_port(port_bindings, internal_port_range, external_range):
    if False:
        while True:
            i = 10
    if external_range is None:
        for internal_port in internal_port_range:
            add_port_mapping(port_bindings, internal_port, None)
    else:
        ports = zip(internal_port_range, external_range)
        for (internal_port, external_port) in ports:
            add_port_mapping(port_bindings, internal_port, external_port)

def build_port_bindings(ports):
    if False:
        while True:
            i = 10
    port_bindings = {}
    for port in ports:
        (internal_port_range, external_range) = split_port(port)
        add_port(port_bindings, internal_port_range, external_range)
    return port_bindings

def _raise_invalid_port(port):
    if False:
        while True:
            i = 10
    raise ValueError('Invalid port "%s", should be [[remote_ip:]remote_port[-remote_port]:]port[/protocol]' % port)

def port_range(start, end, proto, randomly_available_port=False):
    if False:
        while True:
            i = 10
    if not start:
        return start
    if not end:
        return [start + proto]
    if randomly_available_port:
        return [f'{start}-{end}{proto}']
    return [str(port) + proto for port in range(int(start), int(end) + 1)]

def split_port(port):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(port, 'legacy_repr'):
        port = port.legacy_repr()
    port = str(port)
    match = PORT_SPEC.match(port)
    if match is None:
        _raise_invalid_port(port)
    parts = match.groupdict()
    host = parts['host']
    proto = parts['proto'] or ''
    internal = port_range(parts['int'], parts['int_end'], proto)
    external = port_range(parts['ext'], parts['ext_end'], '', len(internal) == 1)
    if host is None:
        if external is not None and len(internal) != len(external):
            raise ValueError("Port ranges don't match in length")
        return (internal, external)
    else:
        if not external:
            external = [None] * len(internal)
        elif len(internal) != len(external):
            raise ValueError("Port ranges don't match in length")
        return (internal, [(host, ext_port) for ext_port in external])