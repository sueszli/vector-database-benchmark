from __future__ import print_function
from __future__ import absolute_import
import socket
import string
import re
import select
import errno
from random import randint
from struct import pack, unpack
import time
from .structure import Structure
CVS_REVISION = '$Revision: 526 $'
INADDR_ANY = '0.0.0.0'
BROADCAST_ADDR = '<broadcast>'
NETBIOS_NS_PORT = 137
NETBIOS_SESSION_PORT = 139
SMB_SESSION_PORT = 445
NODE_B = 0
NODE_P = 8192
NODE_M = 16384
NODE_RESERVED = 24576
NODE_GROUP = 32768
NODE_UNIQUE = 0
TYPE_UNKNOWN = 1
TYPE_WORKSTATION = 0
TYPE_CLIENT = 3
TYPE_SERVER = 32
TYPE_DOMAIN_MASTER = 27
TYPE_DOMAIN_CONTROLLER = 28
TYPE_MASTER_BROWSER = 29
TYPE_BROWSER = 30
TYPE_NETDDE = 31
TYPE_STATUS = 33
OPCODE_QUERY = 0
OPCODE_REGISTRATION = 5
OPCODE_RELEASE = 6
OPCODE_WACK = 7
OPCODE_REFRESH = 8
OPCODE_REQUEST = 0
OPCODE_RESPONSE = 16
NM_FLAGS_BROADCAST = 1
NM_FLAGS_UNICAST = 0
NM_FLAGS_RA = 8
NM_FLAGS_RD = 16
NM_FLAGS_TC = 32
NM_FLAGS_AA = 64
QUESTION_TYPE_NB = 32
QUESTION_TYPE_NBSTAT = 33
QUESTION_CLASS_IN = 1
RR_TYPE_A = 1
RR_TYPE_NS = 2
RR_TYPE_NULL = 10
RR_TYPE_NB = 32
RR_TYPE_NBSTAT = 33
RR_CLASS_IN = 1
RCODE_FMT_ERR = 1
RCODE_SRV_ERR = 2
RCODE_IMP_ERR = 4
RCODE_RFS_ERR = 5
RCODE_ACT_ERR = 6
RCODE_CFT_ERR = 7
NAME_FLAGS_PRM = 512
NAME_FLAGS_ACT = 1024
NAME_FLAG_CNF = 2048
NAME_FLAG_DRG = 4096
NAME_TYPES = {TYPE_UNKNOWN: 'Unknown', TYPE_WORKSTATION: 'Workstation', TYPE_CLIENT: 'Client', TYPE_SERVER: 'Server', TYPE_MASTER_BROWSER: 'Master Browser', TYPE_BROWSER: 'Browser Server', TYPE_DOMAIN_MASTER: 'Domain Master', TYPE_NETDDE: 'NetDDE Server'}
NETBIOS_SESSION_MESSAGE = 0
NETBIOS_SESSION_REQUEST = 129
NETBIOS_SESSION_POSITIVE_RESPONSE = 130
NETBIOS_SESSION_NEGATIVE_RESPONSE = 131
NETBIOS_SESSION_RETARGET_RESPONSE = 132
NETBIOS_SESSION_KEEP_ALIVE = 133

def strerror(errclass, errcode):
    if False:
        return 10
    if errclass == ERRCLASS_OS:
        return ('OS Error', str(errcode))
    elif errclass == ERRCLASS_QUERY:
        return ('Query Error', QUERY_ERRORS.get(errcode, 'Unknown error'))
    elif errclass == ERRCLASS_SESSION:
        return ('Session Error', SESSION_ERRORS.get(errcode, 'Unknown error'))
    else:
        return ('Unknown Error Class', 'Unknown Error')

class NetBIOSError(Exception):
    pass

class NetBIOSTimeout(Exception):

    def __init__(self, message='The NETBIOS connection with the remote host timed out.'):
        if False:
            for i in range(10):
                print('nop')
        Exception.__init__(self, message)

class NBResourceRecord:

    def __init__(self, data=0):
        if False:
            for i in range(10):
                print('nop')
        self._data = data
        try:
            if self._data:
                self.rr_name = re.split('\x00', data)[0]
                offset = len(self.rr_name) + 1
                self.rr_type = unpack('>H', self._data[offset:offset + 2])[0]
                self.rr_class = unpack('>H', self._data[offset + 2:offset + 4])[0]
                self.ttl = unpack('>L', self._data[offset + 4:offset + 8])[0]
                self.rdlength = unpack('>H', self._data[offset + 8:offset + 10])[0]
                self.rdata = self._data[offset + 10:offset + 10 + self.rdlength]
                offset = self.rdlength - 2
                self.unit_id = data[offset:offset + 6]
            else:
                self.rr_name = ''
                self.rr_type = 0
                self.rr_class = 0
                self.ttl = 0
                self.rdlength = 0
                self.rdata = ''
                self.unit_id = ''
        except Exception:
            raise NetBIOSError('Wrong packet format ')

    def set_rr_name(self, name):
        if False:
            print('Hello World!')
        self.rr_name = name

    def set_rr_type(self, name):
        if False:
            i = 10
            return i + 15
        self.rr_type = name

    def set_rr_class(self, cl):
        if False:
            i = 10
            return i + 15
        self.rr_class = cl

    def set_ttl(self, ttl):
        if False:
            for i in range(10):
                print('nop')
        self.ttl = ttl

    def set_rdata(self, rdata):
        if False:
            print('Hello World!')
        self.rdata = rdata
        self.rdlength = len(rdata)

    def get_unit_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self.unit_id

    def get_rr_name(self):
        if False:
            return 10
        return self.rr_name

    def get_rr_class(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rr_class

    def get_ttl(self):
        if False:
            return 10
        return self.ttl

    def get_rdlength(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rdlength

    def get_rdata(self):
        if False:
            print('Hello World!')
        return self.rdata

    def rawData(self):
        if False:
            while True:
                i = 10
        return self.rr_name + pack('!HHLH', self.rr_type, self.rr_class, self.ttl, self.rdlength) + self.rdata

class NBNodeStatusResponse(NBResourceRecord):

    def __init__(self, data=0):
        if False:
            return 10
        NBResourceRecord.__init__(self, data)
        self.num_names = 0
        self.node_names = []
        self.statstics = ''
        self.mac = '00-00-00-00-00-00'
        try:
            if data:
                self._data = self.get_rdata()
                self.num_names = unpack('>B', self._data[:1])[0]
                offset = 1
                for i in range(0, self.num_names):
                    name = self._data[offset:offset + 15]
                    (type, flags) = unpack('>BH', self._data[offset + 15:offset + 18])
                    offset += 18
                    self.node_names.append(NBNodeEntry(name, type, flags))
                self.set_mac_in_hexa(self.get_unit_id())
        except Exception:
            raise NetBIOSError('Wrong packet format ')

    def set_mac_in_hexa(self, data):
        if False:
            print('Hello World!')
        data_aux = ''
        for d in data:
            if data_aux == '':
                data_aux = '%02x' % ord(d)
            else:
                data_aux += '-%02x' % ord(d)
        self.mac = string.upper(data_aux)

    def get_num_names(self):
        if False:
            return 10
        return self.num_names

    def get_mac(self):
        if False:
            print('Hello World!')
        return self.mac

    def set_num_names(self, num):
        if False:
            while True:
                i = 10
        self.num_names = num

    def get_node_names(self):
        if False:
            print('Hello World!')
        return self.node_names

    def add_node_name(self, node_names):
        if False:
            for i in range(10):
                print('nop')
        self.node_names.append(node_names)
        self.num_names += 1

    def rawData(self):
        if False:
            return 10
        res = pack('!B', self.num_names)
        for i in range(0, self.num_names):
            res += self.node_names[i].rawData()

class NBPositiveNameQueryResponse(NBResourceRecord):

    def __init__(self, data=0):
        if False:
            while True:
                i = 10
        NBResourceRecord.__init__(self, data)
        self.addr_entries = []
        if data:
            self._data = self.get_rdata()
            (_qn_length, qn_name, qn_scope) = decode_name(data)
            self._netbios_name = string.rstrip(qn_name[:-1]) + qn_scope
            self._name_type = ord(qn_name[-1])
            self._nb_flags = unpack('!H', self._data[:2])
            offset = 2
            while offset < len(self._data):
                self.addr_entries.append('%d.%d.%d.%d' % unpack('4B', self._data[offset:offset + 4]))
                offset += 4

    def get_netbios_name(self):
        if False:
            print('Hello World!')
        return self._netbios_name

    def get_name_type(self):
        if False:
            while True:
                i = 10
        return self._name_type

    def get_addr_entries(self):
        if False:
            for i in range(10):
                print('nop')
        return self.addr_entries

class NetBIOSPacket:
    """ This is a packet as defined in RFC 1002 """

    def __init__(self, data=0):
        if False:
            i = 10
            return i + 15
        self.name_trn_id = 0
        self.opcode = 0
        self.nm_flags = 0
        self.rcode = 0
        self.qdcount = 0
        self.ancount = 0
        self.nscount = 0
        self.arcount = 0
        self.questions = ''
        self.answers = ''
        if data == 0:
            self._data = ''
        else:
            try:
                self._data = data
                self.opcode = ord(data[2]) >> 3
                self.nm_flags = (ord(data[2]) & 3) << 4 | (ord(data[3]) & 240) >> 4
                self.name_trn_id = unpack('>H', self._data[:2])[0]
                self.rcode = ord(data[3]) & 15
                self.qdcount = unpack('>H', self._data[4:6])[0]
                self.ancount = unpack('>H', self._data[6:8])[0]
                self.nscount = unpack('>H', self._data[8:10])[0]
                self.arcount = unpack('>H', self._data[10:12])[0]
                self.answers = self._data[12:]
            except Exception:
                raise NetBIOSError('Wrong packet format ')

    def set_opcode(self, opcode):
        if False:
            return 10
        self.opcode = opcode

    def set_trn_id(self, trn):
        if False:
            return 10
        self.name_trn_id = trn

    def set_nm_flags(self, nm_flags):
        if False:
            i = 10
            return i + 15
        self.nm_flags = nm_flags

    def set_rcode(self, rcode):
        if False:
            print('Hello World!')
        self.rcode = rcode

    def addQuestion(self, question, qtype, qclass):
        if False:
            while True:
                i = 10
        self.qdcount += 1
        self.questions += question + pack('!HH', qtype, qclass)

    def get_trn_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name_trn_id

    def get_rcode(self):
        if False:
            return 10
        return self.rcode

    def get_nm_flags(self):
        if False:
            return 10
        return self.nm_flags

    def get_opcode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.opcode

    def get_qdcount(self):
        if False:
            print('Hello World!')
        return self.qdcount

    def get_ancount(self):
        if False:
            return 10
        return self.ancount

    def get_nscount(self):
        if False:
            for i in range(10):
                print('nop')
        return self.nscount

    def get_arcount(self):
        if False:
            i = 10
            return i + 15
        return self.arcount

    def rawData(self):
        if False:
            return 10
        secondWord = self.opcode << 11
        secondWord |= self.nm_flags << 4
        secondWord |= self.rcode
        data = pack('!HHHHHH', self.name_trn_id, secondWord, self.qdcount, self.ancount, self.nscount, self.arcount) + self.questions + self.answers
        return data

    def get_answers(self):
        if False:
            return 10
        return self.answers

class NBHostEntry:

    def __init__(self, nbname, nametype, ip):
        if False:
            i = 10
            return i + 15
        self.__nbname = nbname
        self.__nametype = nametype
        self.__ip = ip

    def get_nbname(self):
        if False:
            while True:
                i = 10
        return self.__nbname

    def get_nametype(self):
        if False:
            return 10
        return self.__nametype

    def get_ip(self):
        if False:
            i = 10
            return i + 15
        return self.__ip

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<NBHostEntry instance: NBname="' + self.__nbname + '", IP="' + self.__ip + '">'

class NBNodeEntry:

    def __init__(self, nbname, nametype, flags):
        if False:
            for i in range(10):
                print('nop')
        self.__nbname = string.ljust(nbname, 17)
        self.__nametype = nametype
        self.__flags = flags
        self.__isgroup = flags & 32768
        self.__nodetype = flags & 24576
        self.__deleting = flags & 4096
        self.__isconflict = flags & 2048
        self.__isactive = flags & 1024
        self.__ispermanent = flags & 512

    def get_nbname(self):
        if False:
            i = 10
            return i + 15
        return self.__nbname

    def get_nametype(self):
        if False:
            return 10
        return self.__nametype

    def is_group(self):
        if False:
            print('Hello World!')
        return self.__isgroup

    def get_nodetype(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__nodetype

    def is_deleting(self):
        if False:
            return 10
        return self.__deleting

    def is_conflict(self):
        if False:
            print('Hello World!')
        return self.__isconflict

    def is_active(self):
        if False:
            i = 10
            return i + 15
        return self.__isactive

    def is_permanent(self):
        if False:
            return 10
        return self.__ispermanent

    def set_nbname(self, name):
        if False:
            i = 10
            return i + 15
        self.__nbname = string.ljust(name, 17)

    def set_nametype(self, type):
        if False:
            while True:
                i = 10
        self.__nametype = type

    def set_flags(self, flags):
        if False:
            print('Hello World!')
        self.__flags = flags

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        s = '<NBNodeEntry instance: NBname="' + self.__nbname + '" NameType="' + NAME_TYPES[self.__nametype] + '"'
        if self.__isactive:
            s += ' ACTIVE'
        if self.__isgroup:
            s += ' GROUP'
        if self.__isconflict:
            s += ' CONFLICT'
        if self.__deleting:
            s += ' DELETING'
        return s

    def rawData(self):
        if False:
            print('Hello World!')
        return self.__nbname + pack('!BH', self.__nametype, self.__flags)

class NetBIOS:

    def __init__(self, servport=NETBIOS_NS_PORT):
        if False:
            while True:
                i = 10
        self.__servport = NETBIOS_NS_PORT
        self.__nameserver = None
        self.__broadcastaddr = BROADCAST_ADDR
        self.mac = '00-00-00-00-00-00'

    def _setup_connection(self, dstaddr):
        if False:
            return 10
        port = randint(10000, 60000)
        (af, socktype, proto, _canonname, _sa) = socket.getaddrinfo(dstaddr, port, socket.AF_INET, socket.SOCK_DGRAM)[0]
        s = socket.socket(af, socktype, proto)
        has_bind = 1
        for _i in range(0, 10):
            try:
                s.bind((INADDR_ANY, randint(10000, 60000)))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                has_bind = 1
            except socket.error:
                pass
        if not has_bind:
            raise NetBIOSError('Cannot bind to a good UDP port', ERRCLASS_OS, errno.EAGAIN)
        self.__sock = s

    def set_nameserver(self, nameserver):
        if False:
            return 10
        self.__nameserver = nameserver

    def get_nameserver(self):
        if False:
            while True:
                i = 10
        return self.__nameserver

    def set_broadcastaddr(self, broadcastaddr):
        if False:
            while True:
                i = 10
        self.__broadcastaddr = broadcastaddr

    def get_broadcastaddr(self):
        if False:
            return 10
        return self.__broadcastaddr

    def gethostbyname(self, nbname, qtype=TYPE_WORKSTATION, scope=None, timeout=1):
        if False:
            print('Hello World!')
        return self.__queryname(nbname, self.__nameserver, qtype, scope, timeout)

    def getnodestatus(self, nbname, destaddr=None, type=TYPE_WORKSTATION, scope=None, timeout=1):
        if False:
            while True:
                i = 10
        if destaddr:
            return self.__querynodestatus(nbname, destaddr, type, scope, timeout)
        else:
            return self.__querynodestatus(nbname, self.__nameserver, type, scope, timeout)

    def getnetbiosname(self, ip):
        if False:
            for i in range(10):
                print('nop')
        entries = self.getnodestatus('*', ip)
        entries = filter(lambda x: x.get_nametype() == TYPE_SERVER, entries)
        return entries[0].get_nbname().strip()

    def getmacaddress(self):
        if False:
            print('Hello World!')
        return self.mac

    def __queryname(self, nbname, destaddr, qtype, scope, timeout, retries=0):
        if False:
            return 10
        self._setup_connection(destaddr)
        trn_id = randint(1, 32000)
        p = NetBIOSPacket()
        p.set_trn_id(trn_id)
        netbios_name = nbname.upper()
        qn_label = encode_name(netbios_name, qtype, scope)
        p.addQuestion(qn_label, QUESTION_TYPE_NB, QUESTION_CLASS_IN)
        p.set_nm_flags(NM_FLAGS_RD)
        if not destaddr:
            p.set_nm_flags(p.get_nm_flags() | NM_FLAGS_BROADCAST)
            destaddr = self.__broadcastaddr
        req = p.rawData()
        tries = retries
        while 1:
            self.__sock.sendto(req, (destaddr, self.__servport))
            try:
                (ready, _, _) = select.select([self.__sock.fileno()], [], [], timeout)
                if not ready:
                    if tries:
                        tries -= 1
                    else:
                        raise NetBIOSTimeout
                else:
                    (data, _) = self.__sock.recvfrom(65536, 0)
                    res = NetBIOSPacket(data)
                    if res.get_trn_id() == p.get_trn_id():
                        if res.get_rcode():
                            if res.get_rcode() == 3:
                                return None
                            else:
                                raise NetBIOSError('Negative name query response', ERRCLASS_QUERY, res.get_rcode())
                        if res.get_ancount() != 1:
                            raise NetBIOSError('Malformed response')
                        return NBPositiveNameQueryResponse(res.get_answers())
            except select.error as ex:
                if ex[0] != errno.EINTR and ex[0] != errno.EAGAIN:
                    raise NetBIOSError('Error occurs while waiting for response', ERRCLASS_OS, ex[0])
                raise

    def __querynodestatus(self, nbname, destaddr, type, scope, timeout):
        if False:
            return 10
        self._setup_connection(destaddr)
        trn_id = randint(1, 32000)
        p = NetBIOSPacket()
        p.set_trn_id(trn_id)
        netbios_name = string.upper(nbname)
        qn_label = encode_name(netbios_name, type, scope)
        p.addQuestion(qn_label, QUESTION_TYPE_NBSTAT, QUESTION_CLASS_IN)
        if not destaddr:
            p.set_nm_flags(NM_FLAGS_BROADCAST)
            destaddr = self.__broadcastaddr
        req = p.rawData()
        tries = 3
        while 1:
            try:
                self.__sock.sendto(req, 0, (destaddr, self.__servport))
                (ready, _, _) = select.select([self.__sock.fileno()], [], [], timeout)
                if not ready:
                    if tries:
                        tries -= 1
                    else:
                        raise NetBIOSTimeout
                else:
                    try:
                        (data, _) = self.__sock.recvfrom(65536, 0)
                    except Exception as e:
                        raise NetBIOSError('recvfrom error: %s' % str(e))
                    self.__sock.close()
                    res = NetBIOSPacket(data)
                    if res.get_trn_id() == p.get_trn_id():
                        if res.get_rcode():
                            if res.get_rcode() == 3:
                                raise NetBIOSError('Cannot get data from server')
                            else:
                                raise NetBIOSError('Negative name query response', ERRCLASS_QUERY, res.get_rcode())
                        answ = NBNodeStatusResponse(res.get_answers())
                        self.mac = answ.get_mac()
                        return answ.get_node_names()
            except select.error as ex:
                if ex[0] != errno.EINTR and ex[0] != errno.EAGAIN:
                    raise NetBIOSError('Error occurs while waiting for response', ERRCLASS_OS, ex[0])
            except socket.error as ex:
                raise NetBIOSError('Connection error: %s' % str(ex))

def encode_name(name, type, scope):
    if False:
        return 10
    if name == '*':
        name += '\x00' * 15
    elif len(name) > 15:
        name = name[:15] + chr(type)
    else:
        name = string.ljust(name, 15) + chr(type)
    encoded_name = chr(len(name) * 2) + re.sub('.', _do_first_level_encoding, name)
    if scope:
        encoded_scope = ''
        for s in string.split(scope, '.'):
            encoded_scope = encoded_scope + chr(len(s)) + s
        return encoded_name + encoded_scope + '\x00'
    else:
        return encoded_name + '\x00'

def _do_first_level_encoding(m):
    if False:
        print('Hello World!')
    s = ord(m.group(0))
    return string.uppercase[s >> 4] + string.uppercase[s & 15]

def decode_name(name):
    if False:
        return 10
    name_length = ord(name[0])
    assert name_length == 32
    decoded_name = re.sub('..', _do_first_level_decoding, name[1:33])
    if name[33] == '\x00':
        return (34, decoded_name, '')
    else:
        decoded_domain = ''
        offset = 34
        while 1:
            domain_length = ord(name[offset])
            if domain_length == 0:
                break
            decoded_domain = '.' + name[offset:offset + domain_length]
            offset += domain_length
        return (offset + 1, decoded_name, decoded_domain)

def _do_first_level_decoding(m):
    if False:
        for i in range(10):
            print('nop')
    s = m.group(0)
    return chr(ord(s[0]) - ord('A') << 4 | ord(s[1]) - ord('A'))

class NetBIOSSessionPacket:

    def __init__(self, data=0):
        if False:
            for i in range(10):
                print('nop')
        self.type = 0
        self.flags = 0
        self.length = 0
        if data == 0:
            self._trailer = ''
        else:
            try:
                self.type = ord(data[0])
                if self.type == NETBIOS_SESSION_MESSAGE:
                    self.length = ord(data[1]) << 16 | unpack('!H', data[2:4])[0]
                else:
                    self.flags = ord(data[1])
                    self.length = unpack('!H', data[2:4])[0]
                self._trailer = data[4:]
            except:
                raise NetBIOSError('Wrong packet format ')

    def set_type(self, type):
        if False:
            while True:
                i = 10
        self.type = type

    def get_type(self):
        if False:
            print('Hello World!')
        return self.type

    def rawData(self):
        if False:
            i = 10
            return i + 15
        if self.type == NETBIOS_SESSION_MESSAGE:
            data = pack('!BBH', self.type, self.length >> 16, self.length & 65535) + self._trailer
        else:
            data = pack('!BBH', self.type, self.flags, self.length) + self._trailer
        return data

    def set_trailer(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._trailer = data
        self.length = len(data)

    def get_length(self):
        if False:
            while True:
                i = 10
        return self.length

    def get_trailer(self):
        if False:
            print('Hello World!')
        return self._trailer

class NetBIOSSession:

    def __init__(self, myname, remote_name, remote_host, remote_type=TYPE_SERVER, sess_port=NETBIOS_SESSION_PORT, timeout=None, local_type=TYPE_WORKSTATION, sock=None):
        if False:
            return 10
        if len(myname) > 15:
            self.__myname = string.upper(myname[:15])
        else:
            self.__myname = string.upper(myname)
        self.__local_type = local_type
        assert remote_name
        if remote_name == '*SMBSERVER' and sess_port == SMB_SESSION_PORT:
            remote_name = remote_host
        if remote_name == '*SMBSERVER':
            nb = NetBIOS()
            try:
                res = nb.getnetbiosname(remote_host)
            except:
                res = None
                pass
            if res is not None:
                remote_name = res
        if len(remote_name) > 15:
            self.__remote_name = string.upper(remote_name[:15])
        else:
            self.__remote_name = string.upper(remote_name)
        self.__remote_type = remote_type
        self.__remote_host = remote_host
        if sock is not None:
            self._sock = sock
        else:
            self._sock = self._setup_connection((remote_host, sess_port))
        if sess_port == NETBIOS_SESSION_PORT:
            self._request_session(remote_type, local_type, timeout)

    def get_myname(self):
        if False:
            i = 10
            return i + 15
        return self.__myname

    def get_mytype(self):
        if False:
            print('Hello World!')
        return self.__local_type

    def get_remote_host(self):
        if False:
            i = 10
            return i + 15
        return self.__remote_host

    def get_remote_name(self):
        if False:
            print('Hello World!')
        return self.__remote_name

    def get_remote_type(self):
        if False:
            i = 10
            return i + 15
        return self.__remote_type

    def close(self):
        if False:
            while True:
                i = 10
        self._sock.close()

    def get_socket(self):
        if False:
            print('Hello World!')
        return self._sock

class NetBIOSUDPSessionPacket(Structure):
    TYPE_DIRECT_UNIQUE = 16
    TYPE_DIRECT_GROUP = 17
    FLAGS_MORE_FRAGMENTS = 1
    FLAGS_FIRST_FRAGMENT = 2
    FLAGS_B_NODE = 0
    structure = (('Type', 'B=16'), ('Flags', 'B=2'), ('ID', '<H'), ('_SourceIP', '>L'), ('SourceIP', '"'), ('SourcePort', '>H=138'), ('DataLegth', '>H-Data'), ('Offset', '>H=0'), ('SourceName', 'z'), ('DestinationName', 'z'), ('Data', ':'))

    def getData(self):
        if False:
            while True:
                i = 10
        addr = self['SourceIP'].split('.')
        addr = [int(x) for x in addr]
        addr = (((addr[0] << 8) + addr[1] << 8) + addr[2] << 8) + addr[3]
        self['_SourceIP'] = addr
        return Structure.getData(self)

    def get_trailer(self):
        if False:
            while True:
                i = 10
        return self['Data']

class NetBIOSUDPSession(NetBIOSSession):

    def _setup_connection(self, peer):
        if False:
            print('Hello World!')
        (af, socktype, proto, canonname, sa) = socket.getaddrinfo(peer[0], peer[1], 0, socket.SOCK_DGRAM)[0]
        sock = socket.socket(af, socktype, proto)
        sock.connect(sa)
        sock = socket.socket(af, socktype, proto)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((INADDR_ANY, 138))
        self.peer = peer
        return sock

    def _request_session(self, remote_type, local_type, timeout=None):
        if False:
            return 10
        pass

    def next_id(self):
        if False:
            print('Hello World!')
        if hasattr(self, '__dgram_id'):
            answer = self.__dgram_id
        else:
            self.__dgram_id = randint(1, 65535)
            answer = self.__dgram_id
        self.__dgram_id += 1
        return answer

    def send_packet(self, data):
        if False:
            while True:
                i = 10
        self._sock.connect(self.peer)
        p = NetBIOSUDPSessionPacket()
        p['ID'] = self.next_id()
        p['SourceIP'] = self._sock.getsockname()[0]
        p['SourceName'] = encode_name(self.get_myname(), self.get_mytype(), '')[:-1]
        p['DestinationName'] = encode_name(self.get_remote_name(), self.get_remote_type(), '')[:-1]
        p['Data'] = data
        self._sock.sendto(str(p), self.peer)
        self._sock.close()
        self._sock = self._setup_connection(self.peer)

    def recv_packet(self, timeout=None):
        if False:
            print('Hello World!')
        while 1:
            (data, peer) = self._sock.recvfrom(8192)
            if peer == self.peer:
                break
        return NetBIOSUDPSessionPacket(data)

class NetBIOSTCPSession(NetBIOSSession):

    def __init__(self, myname, remote_name, remote_host, remote_type=TYPE_SERVER, sess_port=NETBIOS_SESSION_PORT, timeout=None, local_type=TYPE_WORKSTATION, sock=None, select_poll=False):
        if False:
            i = 10
            return i + 15
        self.__select_poll = select_poll
        if self.__select_poll:
            self.read_function = self.polling_read
        else:
            self.read_function = self.non_polling_read
        NetBIOSSession.__init__(self, myname, remote_name, remote_host, remote_type=remote_type, sess_port=sess_port, timeout=timeout, local_type=local_type, sock=sock)

    def _setup_connection(self, peer):
        if False:
            for i in range(10):
                print('nop')
        try:
            (af, socktype, proto, canonname, sa) = socket.getaddrinfo(peer[0], peer[1], 0, socket.SOCK_STREAM)[0]
            sock = socket.socket(af, socktype, proto)
            sock.connect(sa)
        except socket.error as e:
            raise socket.error('Connection error (%s:%s)' % (peer[0], peer[1]), e)
        return sock

    def send_packet(self, data):
        if False:
            while True:
                i = 10
        p = NetBIOSSessionPacket()
        p.set_type(NETBIOS_SESSION_MESSAGE)
        p.set_trailer(data)
        self._sock.send(p.rawData())

    def recv_packet(self, timeout=None):
        if False:
            i = 10
            return i + 15
        data = self.__read(timeout)
        return NetBIOSSessionPacket(data)

    def _request_session(self, remote_type, local_type, timeout=None):
        if False:
            while True:
                i = 10
        p = NetBIOSSessionPacket()
        remote_name = encode_name(self.get_remote_name(), remote_type, '')
        myname = encode_name(self.get_myname(), local_type, '')
        p.set_type(NETBIOS_SESSION_REQUEST)
        p.set_trailer(remote_name + myname)
        self._sock.send(p.rawData())
        while 1:
            p = self.recv_packet(timeout)
            if p.get_type() == NETBIOS_SESSION_NEGATIVE_RESPONSE:
                raise NetBIOSError('Cannot request session', ERRCLASS_SESSION, ord(p.get_trailer()[0]))
            elif p.get_type() == NETBIOS_SESSION_POSITIVE_RESPONSE:
                break
            else:
                pass

    def polling_read(self, read_length, timeout):
        if False:
            return 10
        data = ''
        if timeout is None:
            timeout = 3600
        time_left = timeout
        CHUNK_TIME = 0.025
        bytes_left = read_length
        while bytes_left > 0:
            try:
                (ready, _, _) = select.select([self._sock.fileno()], [], [], 0)
                if not ready:
                    if time_left <= 0:
                        raise NetBIOSTimeout
                    else:
                        time.sleep(CHUNK_TIME)
                        time_left -= CHUNK_TIME
                        continue
                received = self._sock.recv(bytes_left)
                if len(received) == 0:
                    raise NetBIOSError('Error while reading from remote', ERRCLASS_OS, None)
                data = data + received
                bytes_left = read_length - len(data)
            except select.error as ex:
                if ex[0] != errno.EINTR and ex[0] != errno.EAGAIN:
                    raise NetBIOSError('Error occurs while reading from remote', ERRCLASS_OS, ex[0])
        return data

    def non_polling_read(self, read_length, timeout):
        if False:
            return 10
        data = ''
        bytes_left = read_length
        while bytes_left > 0:
            try:
                (ready, _, _) = select.select([self._sock.fileno()], [], [], timeout)
                if not ready:
                    raise NetBIOSTimeout
                received = self._sock.recv(bytes_left)
                if len(received) == 0:
                    raise NetBIOSError('Error while reading from remote', ERRCLASS_OS, None)
                data = data + received
                bytes_left = read_length - len(data)
            except select.error as ex:
                if ex[0] != errno.EINTR and ex[0] != errno.EAGAIN:
                    raise NetBIOSError('Error occurs while reading from remote', ERRCLASS_OS, ex[0])
        return data

    def __read(self, timeout=None):
        if False:
            return 10
        data = self.read_function(4, timeout)
        (type, flags, length) = unpack('>ccH', data)
        if ord(type) == NETBIOS_SESSION_MESSAGE:
            length |= ord(flags) << 16
        elif ord(flags) & 1:
            length |= 65536
        data2 = self.read_function(length, timeout)
        return data + data2
ERRCLASS_QUERY = 0
ERRCLASS_SESSION = 240
ERRCLASS_OS = 255
QUERY_ERRORS = {1: 'Request format error. Please file a bug report.', 2: 'Internal server error', 3: 'Name does not exist', 4: 'Unsupported request', 5: 'Request refused'}
SESSION_ERRORS = {128: 'Not listening on called name', 129: 'Not listening for calling name', 130: 'Called name not present', 131: 'Sufficient resources', 143: 'Unspecified error'}

def main():
    if False:
        for i in range(10):
            print('nop')

    def get_netbios_host_by_name(name):
        if False:
            for i in range(10):
                print('nop')
        n = NetBIOS()
        n.set_broadcastaddr('255.255.255.255')
        for qtype in (TYPE_WORKSTATION, TYPE_CLIENT, TYPE_SERVER, TYPE_DOMAIN_MASTER, TYPE_DOMAIN_CONTROLLER):
            try:
                addrs = n.gethostbyname(name, qtype=qtype).get_addr_entries()
            except NetBIOSTimeout:
                continue
            else:
                return addrs
        raise Exception('Host not found')
    n = get_netbios_host_by_name('some-host')
    print(n)
if __name__ == '__main__':
    main()