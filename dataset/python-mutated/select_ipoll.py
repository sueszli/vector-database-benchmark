try:
    import socket, select
except ImportError:
    print('SKIP')
    raise SystemExit

def print_poll_output(lst):
    if False:
        return 10
    print([(type(obj), flags) for (obj, flags) in lst])
poller = select.poll()
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(socket.getaddrinfo('127.0.0.1', 8000)[0][-1])
except OSError:
    print('SKIP')
    raise SystemExit
poller.register(s)
print_poll_output(poller.ipoll(0))
print_poll_output(poller.ipoll(0, 1))
print_poll_output(poller.ipoll(0))
s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s2.bind(socket.getaddrinfo('127.0.0.1', 8001)[0][-1])
poller.register(s)
poller.register(s2)
print_poll_output(poller.ipoll(0))
poller.unregister(s)
print_poll_output(poller.ipoll(0))
poller.unregister(s2)
print_poll_output(poller.ipoll(0))
s2.close()
s.close()