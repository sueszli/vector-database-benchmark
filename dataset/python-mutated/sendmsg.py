"""
sendmsg(2) and recvmsg(2) support for Python.
"""
from collections import namedtuple
from socket import CMSG_SPACE, SCM_RIGHTS, socket as Socket
from typing import List, Tuple
__all__ = ['sendmsg', 'recvmsg', 'getSocketFamily', 'SCM_RIGHTS']
ReceivedMessage = namedtuple('ReceivedMessage', ['data', 'ancillary', 'flags'])

def sendmsg(socket: Socket, data: bytes, ancillary: List[Tuple[int, int, bytes]]=[], flags: int=0) -> int:
    if False:
        return 10
    '\n    Send a message on a socket.\n\n    @param socket: The socket to send the message on.\n    @param data: Bytes to write to the socket.\n    @param ancillary: Extra data to send over the socket outside of the normal\n        datagram or stream mechanism.  By default no ancillary data is sent.\n    @param flags: Flags to affect how the message is sent.  See the C{MSG_}\n        constants in the sendmsg(2) manual page.  By default no flags are set.\n\n    @return: The return value of the underlying syscall, if it succeeds.\n    '
    return socket.sendmsg([data], ancillary, flags)

def recvmsg(socket: Socket, maxSize: int=8192, cmsgSize: int=4096, flags: int=0) -> ReceivedMessage:
    if False:
        while True:
            i = 10
    '\n    Receive a message on a socket.\n\n    @param socket: The socket to receive the message on.\n    @param maxSize: The maximum number of bytes to receive from the socket using\n        the datagram or stream mechanism. The default maximum is 8192.\n    @param cmsgSize: The maximum number of bytes to receive from the socket\n        outside of the normal datagram or stream mechanism. The default maximum\n        is 4096.\n    @param flags: Flags to affect how the message is sent.  See the C{MSG_}\n        constants in the sendmsg(2) manual page. By default no flags are set.\n\n    @return: A named 3-tuple of the bytes received using the datagram/stream\n        mechanism, a L{list} of L{tuple}s giving ancillary received data, and\n        flags as an L{int} describing the data received.\n    '
    (data, ancillary, flags) = socket.recvmsg(maxSize, CMSG_SPACE(cmsgSize), flags)[0:3]
    return ReceivedMessage(data=data, ancillary=ancillary, flags=flags)

def getSocketFamily(socket: Socket) -> int:
    if False:
        while True:
            i = 10
    '\n    Return the family of the given socket.\n\n    @param socket: The socket to get the family of.\n    '
    return socket.family