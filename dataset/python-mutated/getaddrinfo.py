import socket, sys

def test_non_existent():
    if False:
        return 10
    try:
        res = socket.getaddrinfo('nonexistent.example.com', 80)
        print('getaddrinfo returned', res)
    except OSError as e:
        print('getaddrinfo raised')

def test_bogus():
    if False:
        for i in range(10):
            print('nop')
    try:
        res = socket.getaddrinfo('hey.!!$$', 80)
        print('getaddrinfo returned', res)
    except OSError as e:
        print('getaddrinfo raised')
    except Exception as e:
        print('getaddrinfo raised')

def test_ip_addr():
    if False:
        while True:
            i = 10
    try:
        res = socket.getaddrinfo('10.10.10.10', 80)
        print('getaddrinfo returned resolutions')
    except Exception as e:
        print('getaddrinfo raised', e)

def test_0_0_0_0():
    if False:
        print('Hello World!')
    try:
        res = socket.getaddrinfo('0.0.0.0', 80)
        print('getaddrinfo returned resolutions')
    except Exception as e:
        print('getaddrinfo raised', e)

def test_valid():
    if False:
        for i in range(10):
            print('nop')
    try:
        res = socket.getaddrinfo('micropython.org', 80)
        print('getaddrinfo returned resolutions')
    except Exception as e:
        print('getaddrinfo raised', e)
test_funs = [n for n in dir() if n.startswith('test_')]
for f in sorted(test_funs):
    print('--', f, end=': ')
    eval(f + '()')