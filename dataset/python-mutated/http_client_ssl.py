import socket
import ssl

def main(use_stream=True):
    if False:
        print('Hello World!')
    s = socket.socket()
    ai = socket.getaddrinfo('google.com', 443)
    print('Address infos:', ai)
    addr = ai[0][-1]
    print('Connect address:', addr)
    s.connect(addr)
    s = ssl.wrap_socket(s)
    print(s)
    if use_stream:
        s.write(b'GET / HTTP/1.0\r\n\r\n')
        print(s.read(4096))
    else:
        s.send(b'GET / HTTP/1.0\r\n\r\n')
        print(s.recv(4096))
    s.close()
main()