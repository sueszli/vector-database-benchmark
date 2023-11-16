"""
Socket编程
"""
import sys
import socket

def server_func(port):
    if False:
        for i in range(10):
            print('nop')
    '\n    服务端\n    '
    server = socket.socket()
    server.bind(('127.0.0.1', port))
    server.listen(10)
    print('服务端已经启动%s端口......' % port)
    (sock_obj, address) = server.accept()
    sock_obj.settimeout(3)
    print('客户端：%s，超时时间：%s' % (address, sock_obj.gettimeout()))
    while True:
        try:
            recv_data = sock_obj.recv(1024).decode('utf-8')
            print('客户端端 -> 服务端: %s' % recv_data)
            if recv_data == 'quit':
                break
            send_data = 'received[%s]' % recv_data
            sock_obj.send(send_data.encode('utf-8'))
            print('服务端 -> 客户端: %s' % send_data)
        except Exception as excep:
            print('error: ', excep)
    sock_obj.close()
    server.close()

def client_func(port):
    if False:
        for i in range(10):
            print('nop')
    '\n    客户端\n    '
    client = socket.socket()
    client.connect(('127.0.0.1', port))
    while True:
        send_data = input('客户端>').strip()
        client.send(send_data.encode('utf-8'))
        if send_data == 'quit':
            break
        recv_data = client.recv(1024).decode('utf-8')
        print('服务端 -> 客户端: %s' % recv_data)
    client.close()
if __name__ == '__main__':
    flag = sys.argv[1]
    if flag == 'server':
        server_func(9901)
    else:
        client_func(9901)