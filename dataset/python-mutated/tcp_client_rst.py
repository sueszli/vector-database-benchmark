import struct, time, socket, select
PORT = 8000

def convert_poll_list(l):
    if False:
        while True:
            i = 10
    return [ev for (_, ev) in l]

def instance0():
    if False:
        while True:
            i = 10
    multitest.globals(IP=multitest.get_network_ip())
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(socket.getaddrinfo('0.0.0.0', PORT)[0][-1])
    s.listen(1)
    multitest.next()
    (s2, _) = s.accept()
    s2.setblocking(False)
    poll = select.poll()
    poll.register(s2, select.POLLIN)
    time.sleep(0.4)
    print(convert_poll_list(poll.poll(1000)))
    try:
        print(s2.recv(10))
        print(convert_poll_list(poll.poll(1000)))
        print(s2.recv(10))
        print(convert_poll_list(poll.poll(1000)))
        print(s2.recv(10))
        print(convert_poll_list(poll.poll(1000)))
    except OSError as er:
        print(er.errno)
    print(convert_poll_list(poll.poll(1000)))
    print(s2.recv(10))
    print(convert_poll_list(poll.poll(1000)))
    s2.close()
    s.close()

def instance1():
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(socket, 'SO_LINGER'):
        multitest.skip()
    multitest.next()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    s.connect(socket.getaddrinfo(IP, PORT)[0][-1])
    lgr_onoff = 1
    lgr_linger = 0
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', lgr_onoff, lgr_linger))
    s.send(b'GET / HTTP/1.0\r\n\r\n')
    time.sleep(0.2)
    s.close()