import socket
CONTENT = b'HTTP/1.0 200 OK\n\nHello #%d from MicroPython!\n'

def main():
    if False:
        return 10
    s = socket.socket()
    ai = socket.getaddrinfo('0.0.0.0', 8080)
    addr = ai[0][-1]
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(addr)
    s.listen(5)
    print('Listening, connect your browser to http://<this_host>:8080/')
    counter = 0
    while True:
        res = s.accept()
        client_s = res[0]
        req = client_s.recv(4096)
        print('Request:')
        print(req)
        client_s.send(CONTENT % counter)
        client_s.close()
        counter += 1
        print()
main()