import socket, select

def test(peer_addr):
    if False:
        i = 10
        return i + 15
    s = socket.socket()
    poller = select.poll()
    poller.register(s)
    p = poller.poll(0)
    print(len(p), p[0][-1])
    s.connect(peer_addr)
    print(len(poller.poll(0)))
    p = poller.poll(1000)
    print(len(p), p[0][-1])
    s.close()
if __name__ == '__main__':
    test(socket.getaddrinfo('micropython.org', 80)[0][-1])