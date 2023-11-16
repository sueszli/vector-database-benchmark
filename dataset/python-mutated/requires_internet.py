"""Common code to check if the internet is available."""
from Bio import MissingExternalDependencyError

def check():
    if False:
        while True:
            i = 10
    try:
        check.available
    except AttributeError:
        RELIABLE_DOMAIN = 'biopython.org'
        import socket
        try:
            socket.getaddrinfo(RELIABLE_DOMAIN, 80, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror as x:
            check.available = False
        else:
            check.available = True
    if not check.available:
        raise MissingExternalDependencyError('internet not available')