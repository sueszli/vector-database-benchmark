import socket
import itertools
from multiprocessing import Pool

def generate_ip_range(selected_range):
    if False:
        print('Hello World!')
    '\n    generate an IP address range from each provided node.\n    for example `10.0.1-10.1-10` will return a generator\n    object that has IP `10.0.1.1 - 10.0.10.10` in it\n    '
    octets = selected_range.split('.')
    chunks = [map(int, octet.split('-')) for octet in octets]
    ranges = [range(c[0], c[1] + 1) if len(c) == 2 else c for c in chunks]
    for address in itertools.product(*ranges):
        yield '.'.join(map(str, address))

def check_ip_alive(ip):
    if False:
        for i in range(10):
            print('nop')
    '\n    efficiently check if an IP address is alive or not\n    by using the socket.gethostbyaddr function\n    '

    def is_valid_ip(ip):
        if False:
            return 10
        try:
            socket.inet_aton(ip)
            return True
        except:
            return False
    try:
        if not is_valid_ip(ip):
            return False
        else:
            return socket.gethostbyaddr(ip)
    except socket.herror:
        return False

def check_ip_wrapper(generated_ips, limit=250):
    if False:
        while True:
            i = 10
    '\n    multiprocess the check_ip_alive function in order\n    to proces a large amount of IP addresses quickly\n    '
    alive_ips = []
    ips_to_use = []
    i = 0
    proc_pool = Pool(processes=35)
    for ip in generated_ips:
        ips_to_use.append(ip)
        i += 1
        if i == limit:
            break
    for ip in ips_to_use:
        try:
            result = proc_pool.apply_async(check_ip_alive, args=(ip,)).get()
            if not result:
                pass
            else:
                alive_ips.append(ip)
        except Exception:
            pass
    proc_pool.close()
    return alive_ips