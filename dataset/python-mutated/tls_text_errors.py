import socket, ssl, sys

def test(addr):
    if False:
        print('Hello World!')
    s = socket.socket()
    s.connect(addr)
    try:
        s = ssl.wrap_socket(s)
        print('wrap: no exception')
    except OSError as e:
        ok = 'SSL_INVALID_RECORD' in str(e) or 'RECORD_OVERFLOW' in str(e) or 'wrong version' in str(e)
        print('wrap:', ok)
        if not ok:
            print('got exception:', e)
    s.close()
if __name__ == '__main__':
    addr = socket.getaddrinfo('micropython.org', 80)[0][-1]
    test(addr)