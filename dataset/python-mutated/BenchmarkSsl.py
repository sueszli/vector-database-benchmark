from gevent import monkey
monkey.patch_all()
import os
import time
import sys
import socket
import ssl
sys.path.append(os.path.abspath('..'))
import io as StringIO
import gevent
from gevent.server import StreamServer
from gevent.pool import Pool
from Config import config
config.parse()
from util import SslPatch
socks = []
data = os.urandom(1024 * 100)
data += '\n'

def handle(sock_raw, addr):
    if False:
        return 10
    socks.append(sock_raw)
    sock = sock_raw
    try:
        while True:
            line = sock.recv(16 * 1024)
            if not line:
                break
            if line == 'bye\n':
                break
            elif line == 'gotssl\n':
                sock.sendall('yes\n')
                sock = gevent.ssl.wrap_socket(sock_raw, server_side=True, keyfile='../../data/key-rsa.pem', certfile='../../data/cert-rsa.pem', ciphers=ciphers, ssl_version=ssl.PROTOCOL_TLSv1)
            else:
                sock.sendall(data)
    except Exception as err:
        print(err)
    try:
        sock.shutdown(gevent.socket.SHUT_WR)
        sock.close()
    except:
        pass
    socks.remove(sock_raw)
pool = Pool(1000)
server = StreamServer(('127.0.0.1', 1234), handle)
server.start()
total_num = 0
total_bytes = 0
clipher = None
ciphers = 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDH+AES128:ECDHE-RSA-AES128-GCM-SHA256:AES128-GCM-SHA256:AES128-SHA256:AES128-SHA:HIGH:' + '!aNULL:!eNULL:!EXPORT:!DSS:!DES:!RC4:!3DES:!MD5:!PSK'

def getData():
    if False:
        print('Hello World!')
    global total_num, total_bytes, clipher
    data = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 1234))
    sock.send('gotssl\n')
    if sock.recv(128) == 'yes\n':
        sock = ssl.wrap_socket(sock, ciphers=ciphers, ssl_version=ssl.PROTOCOL_TLSv1)
        sock.do_handshake()
        clipher = sock.cipher()
    for req in range(20):
        sock.sendall('req\n')
        buff = StringIO.StringIO()
        data = sock.recv(16 * 1024)
        buff.write(data)
        if not data:
            break
        while not data.endswith('\n'):
            data = sock.recv(16 * 1024)
            if not data:
                break
            buff.write(data)
        total_num += 1
        total_bytes += buff.tell()
        if not data:
            print('No data')
    sock.shutdown(gevent.socket.SHUT_WR)
    sock.close()
s = time.time()

def info():
    if False:
        i = 10
        return i + 15
    import psutil
    import os
    process = psutil.Process(os.getpid())
    if 'memory_info' in dir(process):
        memory_info = process.memory_info
    else:
        memory_info = process.get_memory_info
    while 1:
        print(total_num, 'req', total_bytes / 1024, 'kbytes', 'transfered in', time.time() - s, end=' ')
        print('using', clipher, 'Mem:', memory_info()[0] / float(2 ** 20))
        time.sleep(1)
gevent.spawn(info)
for test in range(1):
    clients = []
    for i in range(500):
        clients.append(gevent.spawn(getData))
    gevent.joinall(clients)
print(total_num, 'req', total_bytes / 1024, 'kbytes', 'transfered in', time.time() - s)