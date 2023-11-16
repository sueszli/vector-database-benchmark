import sys, errno, socket, ssl

def test(addr, hostname, block=True):
    if False:
        print('Hello World!')
    print('---', hostname or addr)
    s = socket.socket()
    s.setblocking(block)
    try:
        s.connect(addr)
        print('connected')
    except OSError as e:
        if e.errno != errno.EINPROGRESS:
            raise
        print('EINPROGRESS')
    try:
        if sys.implementation.name == 'micropython':
            s = ssl.wrap_socket(s, do_handshake=block)
        else:
            s = ssl.wrap_socket(s, do_handshake_on_connect=block)
        print('wrap: True')
    except OSError:
        print('wrap: error')
    if not block:
        try:
            while s.write(b'0') is None:
                pass
        except (ValueError, OSError):
            print('write: error')
    s.close()
if __name__ == '__main__':
    addr = socket.getaddrinfo('micropython.org', 80)[0][-1]
    test(addr, None)
    addr = socket.getaddrinfo('micropython.org', 80)[0][-1]
    test(addr, None, False)
    addr = socket.getaddrinfo('test.mosquitto.org', 8883)[0][-1]
    test(addr, 'test.mosquitto.org')