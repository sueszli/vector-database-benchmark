import socket
port = 11000

class TcpShell:

    def __init__(self):
        if False:
            return 10
        global port
        self.port = port
        port += 1

    def listen(self):
        if False:
            while True:
                i = 10
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind(('127.0.0.1', self.port))
        serversocket.listen(0)
        (self.connection, host) = serversocket.accept()
        self.stdin = self.connection.makefile('r')
        self.stdout = self.connection.makefile('w')

    def close(self):
        if False:
            i = 10
            return i + 15
        self.stdout.close()
        self.stdin.close()
        self.connection.close()