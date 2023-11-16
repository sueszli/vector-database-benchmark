import time
from e2b import Sandbox
printed_ports = []

def print_new_port_and_url(open_ports, sandbox):
    if False:
        return 10
    for port in open_ports:
        if port not in printed_ports:
            printed_ports.append(port)
            host = sandbox.get_hostname(port.port)
            port_url = f'https://{host}'
            print(port, port_url)
sandbox = Sandbox(id='base', on_scan_ports=lambda open_ports: print_new_port_and_url(open_ports, sandbox))
proc = sandbox.process.start('python3 -m http.server 8000')
time.sleep(10)
proc.kill()
sandbox.close()