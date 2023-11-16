import sys
if sys.implementation.name == 'micropython' and sys.platform != 'esp32':
    print('SKIP')
    raise SystemExit
import socket, sys

def test_bind_resolves_0_0_0_0():
    if False:
        return 10
    s = socket.socket()
    try:
        s.bind(('0.0.0.0', 31245))
        print('bind actually bound')
        s.close()
    except Exception as e:
        print('bind raised', e)

def test_bind_resolves_localhost():
    if False:
        return 10
    s = socket.socket()
    try:
        s.bind(('localhost', 31245))
        print('bind actually bound')
        s.close()
    except Exception as e:
        print('bind raised', e)

def test_connect_resolves():
    if False:
        for i in range(10):
            print('nop')
    s = socket.socket()
    try:
        s.connect(('micropython.org', 80))
        print('connect actually connected')
        s.close()
    except Exception as e:
        print('connect raised', e)

def test_connect_non_existent():
    if False:
        print('Hello World!')
    s = socket.socket()
    try:
        s.connect(('nonexistent.example.com', 80))
        print('connect actually connected')
        s.close()
    except OSError as e:
        print('connect raised OSError')
    except Exception as e:
        print('connect raised', e)
test_funs = [n for n in dir() if n.startswith('test_')]
for f in sorted(test_funs):
    print('--', f, end=': ')
    eval(f + '()')