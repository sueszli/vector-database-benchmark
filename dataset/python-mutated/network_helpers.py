import socket
import platform

def get_network_host():
    if False:
        i = 10
        return i + 15
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    if platform.system() == 'Windows':
        return ip_address
    else:
        return '0.0.0.0'