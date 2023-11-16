import sys, os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.insert(0, p)
from qqbot.utf8logger import INFO, ERROR
from qqbot.common import STR2BYTES
import socket

class MySocketServer(object):

    def __init__(self, host, port, name='SocketServer', numListen=1):
        if False:
            print('Hello World!')
        self.host = host
        self.port = int(port)
        self.name = name
        self.numListen = numListen

    def Run(self):
        if False:
            while True:
                i = 10
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen(self.numListen)
            self.sock.settimeout(None)
        except socket.error as e:
            ERROR('无法开启 %s ， %s', self.name, e)
            self.onStartFail(e)
        else:
            INFO('已在 %s 的 %s 端口开启 %s', self.host, self.port, self.name)
            self.onStart()
            while True:
                try:
                    (sock, addr) = self.sock.accept()
                except socket.error as e:
                    ERROR('%s 发生 accept 错误，%s', self.name, e)
                    self.onAcceptError(e)
                else:
                    self.onAccept(sock, addr)

    def onAccept(self, sock, addr):
        if False:
            return 10
        sock.settimeout(10.0)
        try:
            data = sock.recv(8192)
        except socket.error as e:
            ERROR('%s 在接收来自 %s:%s 的数据时发送错误，%s', self.name, addr[0], addr[1], e)
            self.onRecvError(sock, addr, e)
            sock.close()
        else:
            if data == b'##STOP':
                INFO('%s 已停止', self.name)
                self.onStop()
                sys.exit(0)
            else:
                self.onData(sock, addr, data)

    def Stop(self):
        if False:
            while True:
                i = 10
        Query(self.host, self.port, b'##STOP')

    def onData(self, sock, addr, data):
        if False:
            while True:
                i = 10
        try:
            resp = self.response(data)
        except Exception as e:
            resp = '%s 在处理 %s:%s 的请求时发生错误，%s' % (self.name, addr[0], addr[1], e)
            ERROR(resp, exc_info=True)
            resp = STR2BYTES(resp)
        try:
            sock.sendall(resp)
        except socket.error as e:
            ERROR('%s 在向 %s:%s 发送数据时发送错误，%s', self.name, addr[0], addr[1], e)
            self.onSendError(sock, addr, data)
        finally:
            sock.close()

    def onStartFail(self, e):
        if False:
            for i in range(10):
                print('nop')
        pass

    def onStart(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def onAcceptError(self, e):
        if False:
            i = 10
            return i + 15
        pass

    def onRecvError(self, sock, addr, e):
        if False:
            for i in range(10):
                print('nop')
        pass

    def onSendError(self, sock, addr, e):
        if False:
            print('Hello World!')
        pass

    def onStop(self):
        if False:
            print('Hello World!')
        pass

    def response(self, data):
        if False:
            return 10
        return b'Hello, ' + data

def Query(host, port, req):
    if False:
        i = 10
        return i + 15
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    resp = b''
    try:
        sock.connect((host, int(port)))
        sock.sendall(req)
        while True:
            data = sock.recv(8096)
            if not data:
                return resp
            else:
                resp += data
    except socket.error:
        return resp
    finally:
        sock.close()
if __name__ == '__main__':
    import sys
    from qqbot.common import SYSTEMSTR2BYTES
    data = ' '.join(sys.argv[1:]).strip()
    if data:
        (host, port) = ('127.0.0.1', 8191)
        if data == '-s':
            MySocketServer(host, port).Run()
        else:
            print(Query(host, port, SYSTEMSTR2BYTES(data)))