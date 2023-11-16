from micropython import const
try:
    import socket, select, io
except ImportError:
    print('SKIP')
    raise SystemExit
_MP_STREAM_POLL = const(3)
_MP_STREAM_GET_FILENO = const(10)
_MP_STREAM_POLL_RD = const(1)
_MP_STREAM_POLL_WR = const(4)

def print_poll_output(lst):
    if False:
        i = 10
        return i + 15
    print([(type(obj), flags) for (obj, flags) in lst])

class CustomPollable(io.IOBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.poll_state = 0

    def ioctl(self, cmd, arg):
        if False:
            while True:
                i = 10
        if cmd == _MP_STREAM_GET_FILENO:
            return -1
        print('CustomPollable.ioctl', cmd, arg)
        if cmd == _MP_STREAM_POLL:
            if self.poll_state == 'delay_rd':
                self.poll_state = _MP_STREAM_POLL_RD
                return 0
            elif self.poll_state < 0:
                return self.poll_state
            else:
                return self.poll_state & arg
poller = select.poll()
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(socket.getaddrinfo('127.0.0.1', 8000)[0][-1])
except OSError:
    print('SKIP')
    raise SystemExit
x = CustomPollable()
poller.register(s)
poller.register(x)
poller.modify(x, select.POLLIN)
print_poll_output(poller.poll(0))
x.poll_state = _MP_STREAM_POLL_WR
print_poll_output(poller.poll(0))
x.poll_state = _MP_STREAM_POLL_RD
print_poll_output(poller.poll(0))
poller.modify(s, select.POLLIN)
x.poll_state = 'delay_rd'
print_poll_output(poller.poll())
x.poll_state = -1000
try:
    poller.poll(0)
except OSError as er:
    print('OSError', er.errno)
s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
x.poll_state = _MP_STREAM_POLL_RD
poller.register(s2)
poller.unregister(s2)
print_poll_output(poller.poll())
for _ in range(2):
    poller.unregister(s)
    poller.unregister(x)
    poller.register(s2)
    poller.register(s, select.POLLIN)
    poller.register(x, select.POLLIN)
    poller.unregister(s2)
    print_poll_output(poller.poll())
poller.unregister(x)
poller.unregister(s)
s2.close()
s.close()