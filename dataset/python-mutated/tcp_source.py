import socket
import os
from gnuradio import gr, blocks

def _get_sock_fd(addr, port, server):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the file descriptor for the socket.\n    As a client, block on connect, dup the socket descriptor.\n    As a server, block on accept, dup the client descriptor.\n\n    Args:\n        addr: the ip address string\n        port: the tcp port number\n        server: true for server mode, false for client mode\n\n    Returns:\n        the file descriptor number\n    '
    is_ipv6 = False
    if ':' in addr:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
        is_ipv6 = True
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if server:
        try:
            if is_ipv6:
                bind_addr = addr.replace('::ffff:', '')
                sock.bind((bind_addr, port))
            else:
                sock.bind((addr, port))
            gr.log.info('Waiting for a connection on port ' + str(port))
            sock.listen(1)
            (clientsock, address) = sock.accept()
            return os.dup(clientsock.fileno())
        except OSError as e:
            gr.log.error('Unable to bind to port ' + str(port))
            gr.log.error('Error: ' + e.strerror)
            if is_ipv6:
                gr.log.error('IPv6 HINT: If trying to start a local listener, try "::" for the address.')
            return None
        except:
            gr.log.error('Unable to bind to port ' + str(port))
            return None
    else:
        sock.connect((addr, port))
        return os.dup(sock.fileno())

class tcp_source(gr.hier_block2):

    def __init__(self, itemsize, addr, port, server=True):
        if False:
            while True:
                i = 10
        gr.hier_block2.__init__(self, 'tcp_source', gr.io_signature(0, 0, 0), gr.io_signature(1, 1, itemsize))
        fd = _get_sock_fd(addr, port, server)
        self.connect(blocks.file_descriptor_source(itemsize, fd), self)