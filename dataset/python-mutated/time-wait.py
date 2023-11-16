import errno
import socket
import attr

@attr.s(repr=False)
class Options:
    listen1_early = attr.ib(default=None)
    listen1_middle = attr.ib(default=None)
    listen1_late = attr.ib(default=None)
    server = attr.ib(default=None)
    listen2 = attr.ib(default=None)

    def set(self, which, sock):
        if False:
            print('Hello World!')
        value = getattr(self, which)
        if value is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, value)

    def describe(self):
        if False:
            return 10
        info = []
        for f in attr.fields(self.__class__):
            value = getattr(self, f.name)
            if value is not None:
                info.append(f'{f.name}={value}')
        return 'Set/unset: {}'.format(', '.join(info))

def time_wait(options):
    if False:
        for i in range(10):
            print('nop')
    print(options.describe())
    listen0 = socket.socket()
    listen0.bind(('127.0.0.1', 0))
    sockaddr = listen0.getsockname()
    listen0.close()
    listen1 = socket.socket()
    options.set('listen1_early', listen1)
    listen1.bind(sockaddr)
    listen1.listen(1)
    options.set('listen1_middle', listen1)
    client = socket.socket()
    client.connect(sockaddr)
    options.set('listen1_late', listen1)
    (server, _) = listen1.accept()
    options.set('server', server)
    server.close()
    assert client.recv(10) == b''
    client.close()
    listen1.close()
    listen2 = socket.socket()
    options.set('listen2', listen2)
    try:
        listen2.bind(sockaddr)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print('  -> EADDRINUSE')
        else:
            raise
    else:
        print('  -> ok')
time_wait(Options())
time_wait(Options(listen1_early=True, server=True, listen2=True))
time_wait(Options(listen1_early=True))
time_wait(Options(server=True))
time_wait(Options(listen2=True))
time_wait(Options(listen1_early=True, listen2=True))
time_wait(Options(server=True, listen2=True))
time_wait(Options(listen1_middle=True, listen2=True))
time_wait(Options(listen1_late=True, listen2=True))
time_wait(Options(listen1_middle=True, server=False, listen2=True))