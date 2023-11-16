import subprocess, re
from calibre.constants import iswindows, ismacos

def get_address_of_default_gateway(family='AF_INET'):
    if False:
        for i in range(10):
            print('nop')
    import netifaces
    ip = netifaces.gateways()['default'][getattr(netifaces, family)][0]
    if isinstance(ip, bytes):
        ip = ip.decode('ascii')
    return ip

def get_addresses_for_interface(name, family='AF_INET'):
    if False:
        i = 10
        return i + 15
    import netifaces
    for entry in netifaces.ifaddresses(name)[getattr(netifaces, family)]:
        if entry.get('broadcast'):
            addr = entry.get('addr')
            if addr:
                if isinstance(addr, bytes):
                    addr = addr.decode('ascii')
                yield addr
if iswindows:

    def get_default_route_src_address_external():
        if False:
            for i in range(10):
                print('nop')
        raw = subprocess.check_output('route -4 print 0.0.0.0'.split(), creationflags=subprocess.DETACHED_PROCESS).decode('utf-8', 'replace')
        in_table = False
        default_gateway = get_address_of_default_gateway()
        for line in raw.splitlines():
            parts = line.strip().split()
            if in_table:
                if len(parts) == 6:
                    (network, destination, netmask, gateway, interface, metric) = parts
                elif len(parts) == 5:
                    (destination, netmask, gateway, interface, metric) = parts
                if gateway == default_gateway:
                    return interface
            elif parts == 'Network Destination Netmask Gateway Interface Metric'.split():
                in_table = True

    def get_default_route_src_address_api():
        if False:
            print('Hello World!')
        from calibre.utils.iphlpapi import routes
        for route in routes():
            if route.interface and route.destination == '0.0.0.0':
                for addr in get_addresses_for_interface(route.interface):
                    return addr
    get_default_route_src_address = get_default_route_src_address_api
elif ismacos:

    def get_default_route_src_address():
        if False:
            return 10
        raw = subprocess.check_output('route -n get -inet default'.split()).decode('utf-8')
        m = re.search('^\\s*interface:\\s*(\\S+)\\s*$', raw, flags=re.MULTILINE)
        if m is not None:
            interface = m.group(1)
            for addr in get_addresses_for_interface(interface):
                return addr
else:

    def get_default_route_src_address():
        if False:
            i = 10
            return i + 15
        with open('/proc/net/route', 'rb') as f:
            raw = f.read().decode('utf-8')
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) > 1 and parts[1] == '00000000':
                for addr in get_addresses_for_interface(parts[0]):
                    return addr
if __name__ == '__main__':
    print(get_default_route_src_address())