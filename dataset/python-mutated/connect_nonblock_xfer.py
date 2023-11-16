import sys, time, socket, errno, ssl
isMP = sys.implementation.name == 'micropython'

def dp(e):
    if False:
        print('Hello World!')
    pass

def do_connect(peer_addr, tls, handshake):
    if False:
        while True:
            i = 10
    s = socket.socket()
    s.setblocking(False)
    try:
        s.connect(peer_addr)
    except OSError as er:
        print('connect:', er.errno == errno.EINPROGRESS)
        if er.errno != errno.EINPROGRESS:
            print('  got', er.errno)
    if tls:
        try:
            if sys.implementation.name == 'micropython':
                s = ssl.wrap_socket(s, do_handshake=handshake)
            else:
                s = ssl.wrap_socket(s, do_handshake_on_connect=handshake)
            print('wrap: True')
        except Exception as e:
            dp(e)
            print('wrap:', e)
    elif handshake:
        time.sleep(0.2)
    return s

def test(peer_addr, tls=False, handshake=False):
    if False:
        return 10
    hasRW = isMP or tls
    hasSR = not (isMP and tls)
    if hasSR:
        s = do_connect(peer_addr, tls, handshake)
        try:
            ret = s.send(b'1234')
            print('send:', handshake and ret == 4)
        except OSError as er:
            dp(er)
            print('send:', er.errno in (errno.EAGAIN, errno.EINPROGRESS))
        s.close()
    else:
        print('connect:', True)
        if tls:
            print('wrap:', True)
        print('send:', True)
    if hasRW:
        s = do_connect(peer_addr, tls, handshake)
        try:
            ret = s.write(b'1234')
            print('write:', ret in (4, None))
        except OSError as er:
            dp(er)
            print('write:', False)
        except ValueError as er:
            dp(er)
            print('write:', er.args[0] == 'Write on closed or unwrapped SSL socket.')
        s.close()
    else:
        print('connect:', True)
        if tls:
            print('wrap:', True)
        print('write:', True)
    if hasSR:
        s = do_connect(peer_addr, tls, handshake)
        try:
            print('recv:', s.recv(10))
        except OSError as er:
            dp(er)
            print('recv:', er.errno == errno.EAGAIN)
        s.close()
    else:
        print('connect:', True)
        if tls:
            print('wrap:', True)
        print('recv:', True)
    if hasRW:
        s = do_connect(peer_addr, tls, handshake)
        try:
            ret = s.read(10)
            print('read:', ret is None)
        except OSError as er:
            dp(er)
            print('read:', False)
        except ValueError as er:
            dp(er)
            print('read:', er.args[0] == 'Read on closed or unwrapped SSL socket.')
        s.close()
    else:
        print('connect:', True)
        if tls:
            print('wrap:', True)
        print('read:', True)
if __name__ == '__main__':
    print('--- Plain sockets to nowhere ---')
    test(socket.getaddrinfo('192.0.2.1', 80)[0][-1], False, False)
    print('--- SSL sockets to nowhere ---')
    test(socket.getaddrinfo('192.0.2.1', 443)[0][-1], True, False)
    print('--- Plain sockets ---')
    test(socket.getaddrinfo('micropython.org', 80)[0][-1], False, True)
    print('--- SSL sockets ---')
    test(socket.getaddrinfo('micropython.org', 443)[0][-1], True, True)