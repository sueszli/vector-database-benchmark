"""
Topic: 字节字符串操作
Desc : 
"""
import re

def byte_str():
    if False:
        while True:
            i = 10
    data = b'Hello World'
    print(data[0:5])
    print(data.startswith(b'Hello'))
    print(data.split())
    print(data.replace(b'Hello', b'Hello Cruel'))
    data = bytearray(b'Hello World')
    print(data[0:5])
    print(data.startswith(b'Hello'))
    print(data.split())
    print(data.replace(b'Hello', b'Hello Cruel'))
    data = b'FOO:BAR,SPAM'
    print(re.split(b'[:,]', data))
    s = b'Hello World'
    print(s)
    print(s.decode('utf-8'))
    print('{:10s} {:10d} {:10.2f}'.format('ACME', 100, 490.1).encode('ascii'))
if __name__ == '__main__':
    byte_str()