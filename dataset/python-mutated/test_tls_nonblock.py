import socket, ssl, errno, sys, time, select

def test_one(site, opts):
    if False:
        while True:
            i = 10
    ai = socket.getaddrinfo(site, 443)
    addr = ai[0][-1]
    print(addr)
    s = socket.socket()
    s.setblocking(False)
    try:
        s.connect(addr)
        raise OSError(-1, 'connect blocks')
    except OSError as e:
        if e.errno != errno.EINPROGRESS:
            raise
    if sys.implementation.name != 'micropython':
        select.select([], [s], [])
    try:
        try:
            if sys.implementation.name == 'micropython':
                s = ssl.wrap_socket(s, do_handshake=False)
            else:
                s = ssl.wrap_socket(s, do_handshake_on_connect=False)
        except OSError as e:
            if e.errno != errno.EINPROGRESS:
                raise
        print('wrapped')
        if sys.implementation.name != 'micropython':
            while True:
                try:
                    s.do_handshake()
                    break
                except ssl.SSLError as err:
                    if err.args[0] == ssl.SSL_ERROR_WANT_READ:
                        select.select([s], [], [])
                    elif err.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                        select.select([], [s], [])
                    else:
                        raise
                time.sleep(0.1)
        out = b'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % bytes(site, 'latin')
        while len(out) > 0:
            n = s.write(out)
            if n is None:
                continue
            if n > 0:
                out = out[n:]
            elif n == 0:
                raise OSError(-1, 'unexpected EOF in write')
        print('wrote')
        resp = b''
        while True:
            try:
                b = s.read(128)
            except OSError as err:
                if err.errno == 2:
                    continue
                raise
            if b is None:
                continue
            if len(b) > 0:
                if len(resp) < 1024:
                    resp += b
            elif len(b) == 0:
                break
        print('read')
        if resp[:7] != b'HTTP/1.':
            raise ValueError("response doesn't start with HTTP/1.")
    finally:
        s.close()
SITES = ['google.com', {'host': 'www.google.com'}, 'micropython.org', 'pypi.org', {'host': 'api.pushbullet.com', 'sni': True}]

def main():
    if False:
        i = 10
        return i + 15
    for site in SITES:
        opts = {}
        if isinstance(site, dict):
            opts = site
            site = opts['host']
        try:
            test_one(site, opts)
            print(site, 'ok')
        except Exception as e:
            print(site, 'error')
    print('DONE')
main()