"""
Topic: 文件描述符到文件的转换
Desc : 
"""
import os
import sys
from socket import socket, AF_INET, SOCK_STREAM

def file_descriptor():
    if False:
        i = 10
        return i + 15
    fd = os.open('somefile.txt', os.O_WRONLY | os.O_CREAT)
    f = open(fd, 'wt')
    f.write('hello world\n')
    f.close()
    bstdout = open(sys.stdout.fileno(), 'wb', closefd=False)
    bstdout.write(b'Hello World\n')
    bstdout.flush()

def echo_client(client_sock, addr):
    if False:
        return 10
    print('Got connection from', addr)
    client_in = open(client_sock.fileno(), 'rt', encoding='latin-1', closefd=False)
    client_out = open(client_sock.fileno(), 'wt', encoding='latin-1', closefd=False)
    for line in client_in:
        client_out.write(line)
        client_out.flush()
    client_sock.close()

def echo_server(address):
    if False:
        return 10
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(address)
    sock.listen(1)
    while True:
        (client, addr) = sock.accept()
        echo_client(client, addr)
if __name__ == '__main__':
    file_descriptor()