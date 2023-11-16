import errno
import socket
modes = ['default', 'SO_REUSEADDR', 'SO_EXCLUSIVEADDRUSE']
bind_types = ['wildcard', 'specific']

def sock(mode):
    if False:
        for i in range(10):
            print('nop')
    s = socket.socket(family=socket.AF_INET)
    if mode == 'SO_REUSEADDR':
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    elif mode == 'SO_EXCLUSIVEADDRUSE':
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    return s

def bind(sock, bind_type):
    if False:
        i = 10
        return i + 15
    if bind_type == 'wildcard':
        sock.bind(('0.0.0.0', 12345))
    elif bind_type == 'specific':
        sock.bind(('127.0.0.1', 12345))
    else:
        raise AssertionError()

def table_entry(mode1, bind_type1, mode2, bind_type2):
    if False:
        i = 10
        return i + 15
    with sock(mode1) as sock1:
        bind(sock1, bind_type1)
        try:
            with sock(mode2) as sock2:
                bind(sock2, bind_type2)
        except OSError as exc:
            if exc.winerror == errno.WSAEADDRINUSE:
                return 'INUSE'
            elif exc.winerror == errno.WSAEACCES:
                return 'ACCESS'
            raise
        else:
            return 'Success'
print('\n                                                       second bind\n                               | ' + ' | '.join(['%-19s' % mode for mode in modes]))
print('                              ', end='')
for _ in modes:
    print(' | ' + ' | '.join(['%8s' % bind_type for bind_type in bind_types]), end='')
print('\nfirst bind                     -----------------------------------------------------------------')
for mode1 in modes:
    for bind_type1 in bind_types:
        row = []
        for mode2 in modes:
            for bind_type2 in bind_types:
                entry = table_entry(mode1, bind_type1, mode2, bind_type2)
                row.append(entry)
        print(f'{mode1:>19} | {bind_type1:>8} | ' + ' | '.join(['%8s' % entry for entry in row]))