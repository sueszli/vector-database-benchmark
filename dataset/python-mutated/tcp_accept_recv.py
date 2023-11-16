import socket
PORT = 8000

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
    s.accept()
    try:
        print('recv', s.recv(10))
    except OSError as er:
        print(er.errno in (107, 128))
    s.close()

def instance1():
    if False:
        while True:
            i = 10
    multitest.next()
    s = socket.socket()
    s.connect(socket.getaddrinfo(IP, PORT)[0][-1])
    s.send(b'GET / HTTP/1.0\r\n\r\n')
    s.close()