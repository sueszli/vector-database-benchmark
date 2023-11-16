"""
Windows implementation of local network interface enumeration.
"""
from ctypes import POINTER, Structure, WinDLL, byref, c_int, c_void_p, cast, create_string_buffer, create_unicode_buffer, wstring_at
from socket import AF_INET6, SOCK_STREAM, socket
WS2_32 = WinDLL('ws2_32')
SOCKET = c_int
DWORD = c_int
LPVOID = c_void_p
LPSOCKADDR = c_void_p
LPWSAPROTOCOL_INFO = c_void_p
LPTSTR = c_void_p
LPDWORD = c_void_p
LPWSAOVERLAPPED = c_void_p
LPWSAOVERLAPPED_COMPLETION_ROUTINE = c_void_p
WSAIoctl = WS2_32.WSAIoctl
WSAIoctl.argtypes = [SOCKET, DWORD, LPVOID, DWORD, LPVOID, DWORD, LPDWORD, LPWSAOVERLAPPED, LPWSAOVERLAPPED_COMPLETION_ROUTINE]
WSAIoctl.restype = c_int
WSAAddressToString = WS2_32.WSAAddressToStringW
WSAAddressToString.argtypes = [LPSOCKADDR, DWORD, LPWSAPROTOCOL_INFO, LPTSTR, LPDWORD]
WSAAddressToString.restype = c_int
SIO_ADDRESS_LIST_QUERY = 1207959574
WSAEFAULT = 10014

class SOCKET_ADDRESS(Structure):
    _fields_ = [('lpSockaddr', c_void_p), ('iSockaddrLength', c_int)]

def make_SAL(ln):
    if False:
        print('Hello World!')

    class SOCKET_ADDRESS_LIST(Structure):
        _fields_ = [('iAddressCount', c_int), ('Address', SOCKET_ADDRESS * ln)]
    return SOCKET_ADDRESS_LIST

def win32GetLinkLocalIPv6Addresses():
    if False:
        print('Hello World!')
    '\n    Return a list of strings in colon-hex format representing all the link local\n    IPv6 addresses available on the system, as reported by\n    I{WSAIoctl}/C{SIO_ADDRESS_LIST_QUERY}.\n    '
    s = socket(AF_INET6, SOCK_STREAM)
    size = 4096
    retBytes = c_int()
    for i in range(2):
        buf = create_string_buffer(size)
        ret = WSAIoctl(s.fileno(), SIO_ADDRESS_LIST_QUERY, 0, 0, buf, size, byref(retBytes), 0, 0)
        if ret and retBytes.value:
            size = retBytes.value
        else:
            break
    if ret:
        raise RuntimeError('WSAIoctl failure')
    addrList = cast(buf, POINTER(make_SAL(0)))
    addrCount = addrList[0].iAddressCount
    addrList = cast(buf, POINTER(make_SAL(addrCount)))
    addressStringBufLength = 1024
    addressStringBuf = create_unicode_buffer(addressStringBufLength)
    retList = []
    for i in range(addrList[0].iAddressCount):
        retBytes.value = addressStringBufLength
        address = addrList[0].Address[i]
        ret = WSAAddressToString(address.lpSockaddr, address.iSockaddrLength, 0, addressStringBuf, byref(retBytes))
        if ret:
            raise RuntimeError('WSAAddressToString failure')
        retList.append(wstring_at(addressStringBuf))
    return [addr for addr in retList if '%' in addr]