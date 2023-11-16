import os
import socket
import sys
from struct import unpack
from typing import Tuple
from twisted.python.sendmsg import recvmsg

def recvfd(socketfd: int) -> Tuple[int, bytes]:
    if False:
        i = 10
        return i + 15
    '\n    Receive a file descriptor from a L{sendmsg} message on the given C{AF_UNIX}\n    socket.\n\n    @param socketfd: An C{AF_UNIX} socket, attached to another process waiting\n        to send sockets via the ancillary data mechanism in L{send1msg}.\n\n    @param fd: C{int}\n\n    @return: a 2-tuple of (new file descriptor, description).\n    @rtype: 2-tuple of (C{int}, C{bytes})\n    '
    ourSocket = socket.fromfd(socketfd, socket.AF_UNIX, socket.SOCK_STREAM)
    (data, ancillary, flags) = recvmsg(ourSocket)
    [(cmsgLevel, cmsgType, packedFD)] = ancillary
    [unpackedFD] = unpack('i', packedFD)
    return (unpackedFD, data)
if __name__ == '__main__':
    (fd, description) = recvfd(int(sys.argv[1]))
    os.write(fd, b'Test fixture data: ' + description + b'.\n')
    os.close(fd)