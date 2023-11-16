"""
Topic: 使用迭代器重写while无限循环
Desc : 
"""
import sys

def process_data():
    if False:
        while True:
            i = 10
    print(data)

def reader(s, size):
    if False:
        return 10
    while True:
        data = s.recv(size)
        if data == b'':
            break

def reader2(s, size):
    if False:
        print('Hello World!')
    for data in iter(lambda : s.recv(size), b''):
        process_data(data)

def iterate_while():
    if False:
        print('Hello World!')
    CHUNKSIZE = 8192
    with open('/etc/passwd') as f:
        for chunk in iter(lambda : f.read(10), ''):
            n = sys.stdout.write(chunk)
if __name__ == '__main__':
    iterate_while()