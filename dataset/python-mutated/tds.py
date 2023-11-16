from __future__ import division
from __future__ import print_function
import struct
import socket
import select
import random
import binascii
import math
import datetime
import string
from impacket import ntlm, uuid, LOG
from impacket.structure import Structure
try:
    from OpenSSL import SSL
except:
    LOG.critical("pyOpenSSL is not installed, can't continue")
    raise

class DummyPrint:

    def logMessage(self, message):
        if False:
            return 10
        if message == '\n':
            print(message)
        elif message == '\r':
            print()
        else:
            print(message, end=' ')
SQLR_PORT = 1434
SQLR_CLNT_BCAST_EX = 2
SQLR_CLNT_UCAST_EX = 3
SQLR_CLNT_UCAST_INST = 4
SQLR_CLNT_UCAST_DAC = 15

class SQLR(Structure):
    commonHdr = (('OpCode', 'B'),)

class SQLR_UCAST_INST(SQLR):
    structure = ('Instance', ':')

    def __init__(self, data=None):
        if False:
            return 10
        SQLR.__init__(self, data)
        if data is not None:
            self['OpCode'] = SQLR_CLNT_UCAST_INST

class SQLR_UCAST_DAC(SQLR):
    structure = (('Protocol', 'B=1'), ('Instance', ':'))

    def __init__(self, data=None):
        if False:
            i = 10
            return i + 15
        SQLR.__init__(self, data)
        if data is not None:
            self['OpCode'] = SQLR_CLNT_UCAST_DAC

class SQLR_Response(SQLR):
    structure = (('Size', '<H'), ('_Data', '_-Data', 'self["Size"]'), ('Data', ':'))

class SQLErrorException(Exception):
    pass
TDS_SQL_BATCH = 1
TDS_PRE_TDS_LOGIN = 2
TDS_RPC = 3
TDS_TABULAR = 4
TDS_ATTENTION = 6
TDS_BULK_LOAD_DATA = 7
TDS_TRANSACTION = 14
TDS_LOGIN7 = 16
TDS_SSPI = 17
TDS_PRE_LOGIN = 18
TDS_STATUS_NORMAL = 0
TDS_STATUS_EOM = 1
TDS_STATUS_RESET_CONNECTION = 8
TDS_STATUS_RESET_SKIPTRANS = 16
TDS_ENCRYPT_OFF = 0
TDS_ENCRYPT_ON = 1
TDS_ENCRYPT_NOT_SUP = 2
TDS_ENCRYPT_REQ = 3
TDS_INTEGRATED_SECURITY_ON = 128
TDS_INIT_LANG_FATAL = 1
TDS_ODBC_ON = 2
TDS_ALTMETADATA_TOKEN = 136
TDS_ALTROW_TOKEN = 211
TDS_COLMETADATA_TOKEN = 129
TDS_COLINFO_TOKEN = 165
TDS_DONE_TOKEN = 253
TDS_DONEPROC_TOKEN = 254
TDS_DONEINPROC_TOKEN = 255
TDS_ENVCHANGE_TOKEN = 227
TDS_ERROR_TOKEN = 170
TDS_INFO_TOKEN = 171
TDS_LOGINACK_TOKEN = 173
TDS_NBCROW_TOKEN = 210
TDS_OFFSET_TOKEN = 120
TDS_ORDER_TOKEN = 169
TDS_RETURNSTATUS_TOKEN = 121
TDS_RETURNVALUE_TOKEN = 172
TDS_ROW_TOKEN = 209
TDS_SSPI_TOKEN = 237
TDS_TABNAME_TOKEN = 164
TDS_ENVCHANGE_DATABASE = 1
TDS_ENVCHANGE_LANGUAGE = 2
TDS_ENVCHANGE_CHARSET = 3
TDS_ENVCHANGE_PACKETSIZE = 4
TDS_ENVCHANGE_UNICODE = 5
TDS_ENVCHANGE_UNICODE_DS = 6
TDS_ENVCHANGE_COLLATION = 7
TDS_ENVCHANGE_TRANS_START = 8
TDS_ENVCHANGE_TRANS_COMMIT = 9
TDS_ENVCHANGE_ROLLBACK = 10
TDS_ENVCHANGE_DTC = 11
TDS_NULL_TYPE = 31
TDS_INT1TYPE = 48
TDS_BITTYPE = 50
TDS_INT2TYPE = 52
TDS_INT4TYPE = 56
TDS_DATETIM4TYPE = 58
TDS_FLT4TYPE = 59
TDS_MONEYTYPE = 60
TDS_DATETIMETYPE = 61
TDS_FLT8TYPE = 62
TDS_MONEY4TYPE = 122
TDS_INT8TYPE = 127
TDS_GUIDTYPE = 36
TDS_INTNTYPE = 38
TDS_DECIMALTYPE = 55
TDS_NUMERICTYPE = 63
TDS_BITNTYPE = 104
TDS_DECIMALNTYPE = 106
TDS_NUMERICNTYPE = 108
TDS_FLTNTYPE = 109
TDS_MONEYNTYPE = 110
TDS_DATETIMNTYPE = 111
TDS_DATENTYPE = 40
TDS_TIMENTYPE = 41
TDS_DATETIME2NTYPE = 42
TDS_DATETIMEOFFSETNTYPE = 43
TDS_CHARTYPE = 47
TDS_VARCHARTYPE = 39
TDS_BINARYTYPE = 45
TDS_VARBINARYTYPE = 37
TDS_BIGVARBINTYPE = 165
TDS_BIGVARCHRTYPE = 167
TDS_BIGBINARYTYPE = 173
TDS_BIGCHARTYPE = 175
TDS_NVARCHARTYPE = 231
TDS_NCHARTYPE = 239
TDS_XMLTYPE = 241
TDS_UDTTYPE = 240
TDS_TEXTTYPE = 35
TDS_IMAGETYPE = 34
TDS_NTEXTTYPE = 99
TDS_SSVARIANTTYPE = 98

class TDSPacket(Structure):
    structure = (('Type', '<B'), ('Status', '<B=1'), ('Length', '>H=8+len(Data)'), ('SPID', '>H=0'), ('PacketID', '<B=0'), ('Window', '<B=0'), ('Data', ':'))

class TDS_PRELOGIN(Structure):
    structure = (('VersionToken', '>B=0'), ('VersionOffset', '>H'), ('VersionLength', '>H=len(self["Version"])'), ('EncryptionToken', '>B=0x1'), ('EncryptionOffset', '>H'), ('EncryptionLength', '>H=1'), ('InstanceToken', '>B=2'), ('InstanceOffset', '>H'), ('InstanceLength', '>H=len(self["Instance"])'), ('ThreadIDToken', '>B=3'), ('ThreadIDOffset', '>H'), ('ThreadIDLength', '>H=4'), ('EndToken', '>B=0xff'), ('_Version', '_-Version', 'self["VersionLength"]'), ('Version', ':'), ('Encryption', 'B'), ('_Instance', '_-Instance', 'self["InstanceLength"]-1'), ('Instance', ':'), ('ThreadID', ':'))

    def getData(self):
        if False:
            return 10
        self['VersionOffset'] = 21
        self['EncryptionOffset'] = self['VersionOffset'] + len(self['Version'])
        self['InstanceOffset'] = self['EncryptionOffset'] + 1
        self['ThreadIDOffset'] = self['InstanceOffset'] + len(self['Instance'])
        return Structure.getData(self)

class TDS_LOGIN(Structure):
    structure = (('Length', '<L=0'), ('TDSVersion', '>L=0x71'), ('PacketSize', '<L=32764'), ('ClientProgVer', '>L=7'), ('ClientPID', '<L=0'), ('ConnectionID', '<L=0'), ('OptionFlags1', '<B=0xe0'), ('OptionFlags2', '<B'), ('TypeFlags', '<B=0'), ('OptionFlags3', '<B=0'), ('ClientTimeZone', '<L=0'), ('ClientLCID', '<L=0'), ('HostNameOffset', '<H'), ('HostNameLength', '<H=len(self["HostName"])//2'), ('UserNameOffset', '<H=0'), ('UserNameLength', '<H=len(self["UserName"])//2'), ('PasswordOffset', '<H=0'), ('PasswordLength', '<H=len(self["Password"])//2'), ('AppNameOffset', '<H'), ('AppNameLength', '<H=len(self["AppName"])//2'), ('ServerNameOffset', '<H'), ('ServerNameLength', '<H=len(self["ServerName"])//2'), ('UnusedOffset', '<H=0'), ('UnusedLength', '<H=0'), ('CltIntNameOffset', '<H'), ('CltIntNameLength', '<H=len(self["CltIntName"])//2'), ('LanguageOffset', '<H=0'), ('LanguageLength', '<H=0'), ('DatabaseOffset', '<H=0'), ('DatabaseLength', '<H=len(self["Database"])//2'), ('ClientID', '6s=b"\x01\x02\x03\x04\x05\x06"'), ('SSPIOffset', '<H'), ('SSPILength', '<H=len(self["SSPI"])'), ('AtchDBFileOffset', '<H'), ('AtchDBFileLength', '<H=len(self["AtchDBFile"])//2'), ('HostName', ':'), ('UserName', ':'), ('Password', ':'), ('AppName', ':'), ('ServerName', ':'), ('CltIntName', ':'), ('Database', ':'), ('SSPI', ':'), ('AtchDBFile', ':'))

    def __init__(self, data=None):
        if False:
            print('Hello World!')
        Structure.__init__(self, data)
        if data is None:
            self['UserName'] = ''
            self['Password'] = ''
            self['Database'] = ''
            self['AtchDBFile'] = ''

    def fromString(self, data):
        if False:
            print('Hello World!')
        Structure.fromString(self, data)
        if self['HostNameLength'] > 0:
            self['HostName'] = data[self['HostNameOffset']:][:self['HostNameLength'] * 2]
        if self['UserNameLength'] > 0:
            self['UserName'] = data[self['UserNameOffset']:][:self['UserNameLength'] * 2]
        if self['PasswordLength'] > 0:
            self['Password'] = data[self['PasswordOffset']:][:self['PasswordLength'] * 2]
        if self['AppNameLength'] > 0:
            self['AppName'] = data[self['AppNameOffset']:][:self['AppNameLength'] * 2]
        if self['ServerNameLength'] > 0:
            self['ServerName'] = data[self['ServerNameOffset']:][:self['ServerNameLength'] * 2]
        if self['CltIntNameLength'] > 0:
            self['CltIntName'] = data[self['CltIntNameOffset']:][:self['CltIntNameLength'] * 2]
        if self['DatabaseLength'] > 0:
            self['Database'] = data[self['DatabaseOffset']:][:self['DatabaseLength'] * 2]
        if self['SSPILength'] > 0:
            self['SSPI'] = data[self['SSPIOffset']:][:self['SSPILength'] * 2]
        if self['AtchDBFileLength'] > 0:
            self['AtchDBFile'] = data[self['AtchDBFileOffset']:][:self['AtchDBFileLength'] * 2]

    def getData(self):
        if False:
            return 10
        index = 36 + 50
        self['HostNameOffset'] = index
        index += len(self['HostName'])
        if self['UserName'] != '':
            self['UserNameOffset'] = index
        else:
            self['UserNameOffset'] = 0
        index += len(self['UserName'])
        if self['Password'] != '':
            self['PasswordOffset'] = index
        else:
            self['PasswordOffset'] = 0
        index += len(self['Password'])
        self['AppNameOffset'] = index
        self['ServerNameOffset'] = self['AppNameOffset'] + len(self['AppName'])
        self['CltIntNameOffset'] = self['ServerNameOffset'] + len(self['ServerName'])
        self['LanguageOffset'] = self['CltIntNameOffset'] + len(self['CltIntName'])
        self['DatabaseOffset'] = self['LanguageOffset']
        self['SSPIOffset'] = self['DatabaseOffset'] + len(self['Database'])
        self['AtchDBFileOffset'] = self['SSPIOffset'] + len(self['SSPI'])
        return Structure.getData(self)

class TDS_LOGIN_ACK(Structure):
    structure = (('TokenType', '<B'), ('Length', '<H'), ('Interface', '<B'), ('TDSVersion', '<L'), ('ProgNameLen', '<B'), ('_ProgNameLen', '_-ProgName', 'self["ProgNameLen"]*2'), ('ProgName', ':'), ('MajorVer', '<B'), ('MinorVer', '<B'), ('BuildNumHi', '<B'), ('BuildNumLow', '<B'))

class TDS_RETURNSTATUS(Structure):
    structure = (('TokenType', '<B'), ('Value', '<L'))

class TDS_INFO_ERROR(Structure):
    structure = (('TokenType', '<B'), ('Length', '<H'), ('Number', '<L'), ('State', '<B'), ('Class', '<B'), ('MsgTextLen', '<H'), ('_MsgTextLen', '_-MsgText', 'self["MsgTextLen"]*2'), ('MsgText', ':'), ('ServerNameLen', '<B'), ('_ServerNameLen', '_-ServerName', 'self["ServerNameLen"]*2'), ('ServerName', ':'), ('ProcNameLen', '<B'), ('_ProcNameLen', '_-ProcName', 'self["ProcNameLen"]*2'), ('ProcName', ':'), ('LineNumber', '<H'))

class TDS_ENVCHANGE(Structure):
    structure = (('TokenType', '<B'), ('Length', '<H=4+len(Data)'), ('Type', '<B'), ('_Data', '_-Data', 'self["Length"]-1'), ('Data', ':'))

class TDS_DONEINPROC(Structure):
    structure = (('TokenType', '<B'), ('Status', '<H'), ('CurCmd', '<H'), ('DoneRowCount', '<L'))

class TDS_ORDER(Structure):
    structure = (('TokenType', '<B'), ('Length', '<H'), ('_Data', '_-Data', 'self["Length"]'), ('Data', ':'))

class TDS_ENVCHANGE_VARCHAR(Structure):
    structure = (('NewValueLen', '<B=len(NewValue)'), ('_NewValue', '_-NewValue', 'self["NewValueLen"]*2'), ('NewValue', ':'), ('OldValueLen', '<B=len(OldValue)'), ('_OldValue', '_-OldValue', 'self["OldValueLen"]*2'), ('OldValue', ':'))

class TDS_ROW(Structure):
    structure = (('TokenType', '<B'), ('Data', ':'))

class TDS_DONE(Structure):
    structure = (('TokenType', '<B'), ('Status', '<H'), ('CurCmd', '<H'), ('DoneRowCount', '<L'))

class TDS_COLMETADATA(Structure):
    structure = (('TokenType', '<B'), ('Count', '<H'), ('Data', ':'))

class MSSQL:

    def __init__(self, address, port=1433, rowsPrinter=DummyPrint()):
        if False:
            i = 10
            return i + 15
        self.packetSize = 32763
        self.server = address
        self.port = port
        self.socket = 0
        self.replies = {}
        self.colMeta = []
        self.rows = []
        self.currentDB = ''
        self.COL_SEPARATOR = '  '
        self.MAX_COL_LEN = 255
        self.lastError = False
        self.tlsSocket = None
        self.__rowsPrinter = rowsPrinter

    def getInstances(self, timeout=5):
        if False:
            print('Hello World!')
        packet = SQLR()
        packet['OpCode'] = SQLR_CLNT_UCAST_EX
        (af, socktype, proto, canonname, sa) = socket.getaddrinfo(self.server, SQLR_PORT, 0, socket.SOCK_DGRAM)[0]
        s = socket.socket(af, socktype, proto)
        s.sendto(packet.getData(), 0, (self.server, SQLR_PORT))
        (ready, _, _) = select.select([s.fileno()], [], [], timeout)
        if not ready:
            return []
        else:
            (data, _) = s.recvfrom(65536, 0)
        s.close()
        resp = SQLR_Response(data)
        entries = resp['Data'].split(b';;')
        entries.pop()
        resp = []
        for (i, entry) in enumerate(entries):
            fields = entry.split(b';')
            ret = {}
            for (j, field) in enumerate(fields):
                if j & 1 == 0:
                    ret[field.decode('utf-8')] = fields[j + 1].decode('utf-8')
            resp.append(ret)
        return resp

    def preLogin(self):
        if False:
            i = 10
            return i + 15
        prelogin = TDS_PRELOGIN()
        prelogin['Version'] = b'\x08\x00\x01U\x00\x00'
        prelogin['Encryption'] = TDS_ENCRYPT_OFF
        prelogin['ThreadID'] = struct.pack('<L', random.randint(0, 65535))
        prelogin['Instance'] = b'MSSQLServer\x00'
        self.sendTDS(TDS_PRE_LOGIN, prelogin.getData(), 0)
        tds = self.recvTDS()
        return TDS_PRELOGIN(tds['Data'])

    def encryptPassword(self, password):
        if False:
            return 10
        return bytes(bytearray([((x & 15) << 4) + ((x & 240) >> 4) ^ 165 for x in bytearray(password)]))

    def connect(self):
        if False:
            i = 10
            return i + 15
        (af, socktype, proto, canonname, sa) = socket.getaddrinfo(self.server, self.port, 0, socket.SOCK_STREAM)[0]
        sock = socket.socket(af, socktype, proto)
        try:
            sock.connect(sa)
        except Exception:
            raise
        self.socket = sock
        return sock

    def disconnect(self):
        if False:
            return 10
        if self.socket:
            return self.socket.close()

    def setPacketSize(self, packetSize):
        if False:
            print('Hello World!')
        self.packetSize = packetSize

    def getPacketSize(self):
        if False:
            return 10
        return self.packetSize

    def socketSendall(self, data):
        if False:
            print('Hello World!')
        if self.tlsSocket is None:
            return self.socket.sendall(data)
        else:
            self.tlsSocket.sendall(data)
            dd = self.tlsSocket.bio_read(self.packetSize)
            return self.socket.sendall(dd)

    def sendTDS(self, packetType, data, packetID=1):
        if False:
            i = 10
            return i + 15
        if len(data) - 8 > self.packetSize:
            remaining = data[self.packetSize - 8:]
            tds = TDSPacket()
            tds['Type'] = packetType
            tds['Status'] = TDS_STATUS_NORMAL
            tds['PacketID'] = packetID
            tds['Data'] = data[:self.packetSize - 8]
            self.socketSendall(tds.getData())
            while len(remaining) > self.packetSize - 8:
                packetID += 1
                tds['PacketID'] = packetID
                tds['Data'] = remaining[:self.packetSize - 8]
                self.socketSendall(tds.getData())
                remaining = remaining[self.packetSize - 8:]
            data = remaining
            packetID += 1
        tds = TDSPacket()
        tds['Type'] = packetType
        tds['Status'] = TDS_STATUS_EOM
        tds['PacketID'] = packetID
        tds['Data'] = data
        self.socketSendall(tds.getData())

    def socketRecv(self, packetSize):
        if False:
            print('Hello World!')
        data = self.socket.recv(packetSize)
        if self.tlsSocket is not None:
            dd = b''
            self.tlsSocket.bio_write(data)
            while True:
                try:
                    dd += self.tlsSocket.read(packetSize)
                except SSL.WantReadError:
                    data2 = self.socket.recv(packetSize - len(data))
                    self.tlsSocket.bio_write(data2)
                    pass
                else:
                    data = dd
                    break
        return data

    def recvTDS(self, packetSize=None):
        if False:
            while True:
                i = 10
        if packetSize is None:
            packetSize = self.packetSize
        packet = TDSPacket(self.socketRecv(packetSize))
        status = packet['Status']
        packetLen = packet['Length'] - 8
        while packetLen > len(packet['Data']):
            data = self.socketRecv(packetSize)
            packet['Data'] += data
        remaining = None
        if packetLen < len(packet['Data']):
            remaining = packet['Data'][packetLen:]
            packet['Data'] = packet['Data'][:packetLen]
        while status != TDS_STATUS_EOM:
            if remaining is not None:
                tmpPacket = TDSPacket(remaining)
            else:
                tmpPacket = TDSPacket(self.socketRecv(packetSize))
            packetLen = tmpPacket['Length'] - 8
            while packetLen > len(tmpPacket['Data']):
                data = self.socketRecv(packetSize)
                tmpPacket['Data'] += data
            remaining = None
            if packetLen < len(tmpPacket['Data']):
                remaining = tmpPacket['Data'][packetLen:]
                tmpPacket['Data'] = tmpPacket['Data'][:packetLen]
            status = tmpPacket['Status']
            packet['Data'] += tmpPacket['Data']
            packet['Length'] += tmpPacket['Length'] - 8
        return packet

    def kerberosLogin(self, database, username, password='', domain='', hashes=None, aesKey='', kdcHost=None, TGT=None, TGS=None, useCache=True):
        if False:
            while True:
                i = 10
        if hashes is not None:
            (lmhash, nthash) = hashes.split(':')
            lmhash = binascii.a2b_hex(lmhash)
            nthash = binascii.a2b_hex(nthash)
        else:
            lmhash = ''
            nthash = ''
        resp = self.preLogin()
        if resp['Encryption'] == TDS_ENCRYPT_REQ or resp['Encryption'] == TDS_ENCRYPT_OFF:
            LOG.info('Encryption required, switching to TLS')
            ctx = SSL.Context(SSL.TLS_METHOD)
            ctx.set_cipher_list('ALL:@SECLEVEL=0'.encode('utf-8'))
            tls = SSL.Connection(ctx, None)
            tls.set_connect_state()
            while True:
                try:
                    tls.do_handshake()
                except SSL.WantReadError:
                    data = tls.bio_read(4096)
                    self.sendTDS(TDS_PRE_LOGIN, data, 0)
                    tds = self.recvTDS()
                    tls.bio_write(tds['Data'])
                else:
                    break
            self.packetSize = 16 * 1024 - 1
            self.tlsSocket = tls
        login = TDS_LOGIN()
        login['HostName'] = ''.join([random.choice(string.ascii_letters) for _ in range(8)]).encode('utf-16le')
        login['AppName'] = ''.join([random.choice(string.ascii_letters) for _ in range(8)]).encode('utf-16le')
        login['ServerName'] = self.server.encode('utf-16le')
        login['CltIntName'] = login['AppName']
        login['ClientPID'] = random.randint(0, 1024)
        login['PacketSize'] = self.packetSize
        if database is not None:
            login['Database'] = database.encode('utf-16le')
        login['OptionFlags2'] = TDS_INIT_LANG_FATAL | TDS_ODBC_ON
        from impacket.spnego import SPNEGO_NegTokenInit, TypesMech
        from impacket.krb5.ccache import CCache
        from impacket.krb5.asn1 import AP_REQ, Authenticator, TGS_REP, seq_set
        from impacket.krb5.kerberosv5 import getKerberosTGT, getKerberosTGS, KerberosError
        from impacket.krb5 import constants
        from impacket.krb5.types import Principal, KerberosTime, Ticket
        from pyasn1.codec.der import decoder, encoder
        from pyasn1.type.univ import noValue
        import datetime
        if useCache:
            (domain, username, TGT, TGS) = CCache.parseFile(domain, username, 'MSSQLSvc/%s:%d' % (self.server, self.port))
            if TGS is None:
                LOG.debug("Searching target's instances to look for port number %s" % self.port)
                instances = self.getInstances()
                instanceName = None
                for i in instances:
                    try:
                        if int(i['tcp']) == self.port:
                            instanceName = i['InstanceName']
                    except Exception as e:
                        pass
                if instanceName:
                    (domain, username, TGT, TGS) = CCache.parseFile(domain, username, 'MSSQLSvc/%s.%s:%s' % (self.server.split('.')[0], domain, instanceName))
        userName = Principal(username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)
        while True:
            if TGT is None:
                if TGS is None:
                    try:
                        (tgt, cipher, oldSessionKey, sessionKey) = getKerberosTGT(userName, password, domain, lmhash, nthash, aesKey, kdcHost)
                    except KerberosError as e:
                        if e.getErrorCode() == constants.ErrorCodes.KDC_ERR_ETYPE_NOSUPP.value:
                            if lmhash == '' and nthash == '' and (aesKey == '' or aesKey is None) and (TGT is None) and (TGS is None):
                                from impacket.ntlm import compute_lmhash, compute_nthash
                                LOG.debug('Got KDC_ERR_ETYPE_NOSUPP, fallback to RC4')
                                lmhash = compute_lmhash(password)
                                nthash = compute_nthash(password)
                                continue
                            else:
                                raise
                        else:
                            raise
            else:
                tgt = TGT['KDC_REP']
                cipher = TGT['cipher']
                sessionKey = TGT['sessionKey']
            if TGS is None:
                serverName = Principal('MSSQLSvc/%s.%s:%d' % (self.server.split('.')[0], domain, self.port), type=constants.PrincipalNameType.NT_SRV_INST.value)
                try:
                    (tgs, cipher, oldSessionKey, sessionKey) = getKerberosTGS(serverName, domain, kdcHost, tgt, cipher, sessionKey)
                except KerberosError as e:
                    if e.getErrorCode() == constants.ErrorCodes.KDC_ERR_ETYPE_NOSUPP.value:
                        if lmhash == '' and nthash == '' and (aesKey == '' or aesKey is None) and (TGT is None) and (TGS is None):
                            from impacket.ntlm import compute_lmhash, compute_nthash
                            LOG.debug('Got KDC_ERR_ETYPE_NOSUPP, fallback to RC4')
                            lmhash = compute_lmhash(password)
                            nthash = compute_nthash(password)
                        else:
                            raise
                    else:
                        raise
                else:
                    break
            else:
                tgs = TGS['KDC_REP']
                cipher = TGS['cipher']
                sessionKey = TGS['sessionKey']
                break
        blob = SPNEGO_NegTokenInit()
        blob['MechTypes'] = [TypesMech['MS KRB5 - Microsoft Kerberos 5']]
        tgs = decoder.decode(tgs, asn1Spec=TGS_REP())[0]
        ticket = Ticket()
        ticket.from_asn1(tgs['ticket'])
        apReq = AP_REQ()
        apReq['pvno'] = 5
        apReq['msg-type'] = int(constants.ApplicationTagNumbers.AP_REQ.value)
        opts = list()
        apReq['ap-options'] = constants.encodeFlags(opts)
        seq_set(apReq, 'ticket', ticket.to_asn1)
        authenticator = Authenticator()
        authenticator['authenticator-vno'] = 5
        authenticator['crealm'] = domain
        seq_set(authenticator, 'cname', userName.components_to_asn1)
        now = datetime.datetime.utcnow()
        authenticator['cusec'] = now.microsecond
        authenticator['ctime'] = KerberosTime.to_asn1(now)
        encodedAuthenticator = encoder.encode(authenticator)
        encryptedEncodedAuthenticator = cipher.encrypt(sessionKey, 11, encodedAuthenticator, None)
        apReq['authenticator'] = noValue
        apReq['authenticator']['etype'] = cipher.enctype
        apReq['authenticator']['cipher'] = encryptedEncodedAuthenticator
        blob['MechToken'] = encoder.encode(apReq)
        login['OptionFlags2'] |= TDS_INTEGRATED_SECURITY_ON
        login['SSPI'] = blob.getData()
        login['Length'] = len(login.getData())
        self.sendTDS(TDS_LOGIN7, login.getData())
        if resp['Encryption'] == TDS_ENCRYPT_OFF:
            self.tlsSocket = None
        tds = self.recvTDS()
        self.replies = self.parseReply(tds['Data'])
        if TDS_LOGINACK_TOKEN in self.replies:
            return True
        else:
            return False

    def login(self, database, username, password='', domain='', hashes=None, useWindowsAuth=False):
        if False:
            return 10
        if hashes is not None:
            (lmhash, nthash) = hashes.split(':')
            lmhash = binascii.a2b_hex(lmhash)
            nthash = binascii.a2b_hex(nthash)
        else:
            lmhash = ''
            nthash = ''
        resp = self.preLogin()
        if resp['Encryption'] == TDS_ENCRYPT_REQ or resp['Encryption'] == TDS_ENCRYPT_OFF:
            LOG.info('Encryption required, switching to TLS')
            ctx = SSL.Context(SSL.TLS_METHOD)
            ctx.set_cipher_list('ALL:@SECLEVEL=0'.encode('utf-8'))
            tls = SSL.Connection(ctx, None)
            tls.set_connect_state()
            while True:
                try:
                    tls.do_handshake()
                except SSL.WantReadError:
                    data = tls.bio_read(4096)
                    self.sendTDS(TDS_PRE_LOGIN, data, 0)
                    tds = self.recvTDS()
                    tls.bio_write(tds['Data'])
                else:
                    break
            self.packetSize = 16 * 1024 - 1
            self.tlsSocket = tls
        login = TDS_LOGIN()
        login['HostName'] = ''.join([random.choice(string.ascii_letters) for i in range(8)]).encode('utf-16le')
        login['AppName'] = ''.join([random.choice(string.ascii_letters) for i in range(8)]).encode('utf-16le')
        login['ServerName'] = self.server.encode('utf-16le')
        login['CltIntName'] = login['AppName']
        login['ClientPID'] = random.randint(0, 1024)
        login['PacketSize'] = self.packetSize
        if database is not None:
            login['Database'] = database.encode('utf-16le')
        login['OptionFlags2'] = TDS_INIT_LANG_FATAL | TDS_ODBC_ON
        if useWindowsAuth is True:
            login['OptionFlags2'] |= TDS_INTEGRATED_SECURITY_ON
            auth = ntlm.getNTLMSSPType1('', '')
            login['SSPI'] = auth.getData()
        else:
            login['UserName'] = username.encode('utf-16le')
            login['Password'] = self.encryptPassword(password.encode('utf-16le'))
            login['SSPI'] = ''
        login['Length'] = len(login.getData())
        self.sendTDS(TDS_LOGIN7, login.getData())
        if resp['Encryption'] == TDS_ENCRYPT_OFF:
            self.tlsSocket = None
        tds = self.recvTDS()
        if useWindowsAuth is True:
            serverChallenge = tds['Data'][3:]
            (type3, exportedSessionKey) = ntlm.getNTLMSSPType3(auth, serverChallenge, username, password, domain, lmhash, nthash)
            self.sendTDS(TDS_SSPI, type3.getData())
            tds = self.recvTDS()
        self.replies = self.parseReply(tds['Data'])
        if TDS_LOGINACK_TOKEN in self.replies:
            return True
        else:
            return False

    def processColMeta(self):
        if False:
            for i in range(10):
                print('nop')
        for col in self.colMeta:
            if col['Type'] in [TDS_NVARCHARTYPE, TDS_NCHARTYPE, TDS_NTEXTTYPE]:
                col['Length'] = col['TypeData'] // 2
                fmt = '%%-%ds'
            elif col['Type'] in [TDS_GUIDTYPE]:
                col['Length'] = 36
                fmt = '%%%ds'
            elif col['Type'] in [TDS_DECIMALNTYPE, TDS_NUMERICNTYPE]:
                col['Length'] = ord(col['TypeData'][0:1])
                fmt = '%%%ds'
            elif col['Type'] in [TDS_DATETIMNTYPE]:
                col['Length'] = 19
                fmt = '%%-%ds'
            elif col['Type'] in [TDS_INT4TYPE, TDS_INTNTYPE]:
                col['Length'] = 11
                fmt = '%%%ds'
            elif col['Type'] in [TDS_FLTNTYPE, TDS_MONEYNTYPE]:
                col['Length'] = 25
                fmt = '%%%ds'
            elif col['Type'] in [TDS_BITNTYPE, TDS_BIGCHARTYPE]:
                col['Length'] = col['TypeData']
                fmt = '%%%ds'
            elif col['Type'] in [TDS_BIGBINARYTYPE, TDS_BIGVARBINTYPE]:
                col['Length'] = col['TypeData'] * 2
                fmt = '%%%ds'
            elif col['Type'] in [TDS_TEXTTYPE, TDS_BIGVARCHRTYPE]:
                col['Length'] = col['TypeData']
                fmt = '%%-%ds'
            else:
                col['Length'] = 10
                fmt = '%%%ds'
            col['minLenght'] = 0
            for row in self.rows:
                if len(str(row[col['Name']])) > col['minLenght']:
                    col['minLenght'] = len(str(row[col['Name']]))
            if col['minLenght'] < col['Length']:
                col['Length'] = col['minLenght']
            if len(col['Name']) > col['Length']:
                col['Length'] = len(col['Name'])
            elif col['Length'] > self.MAX_COL_LEN:
                col['Length'] = self.MAX_COL_LEN
            col['Format'] = fmt % col['Length']

    def printColumnsHeader(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.colMeta) == 0:
            return
        for col in self.colMeta:
            self.__rowsPrinter.logMessage(col['Format'] % col['Name'] + self.COL_SEPARATOR)
        self.__rowsPrinter.logMessage('\r')
        for col in self.colMeta:
            self.__rowsPrinter.logMessage('-' * col['Length'] + self.COL_SEPARATOR)
        self.__rowsPrinter.logMessage('\r')

    def printRows(self):
        if False:
            for i in range(10):
                print('nop')
        if self.lastError is True:
            return
        self.processColMeta()
        self.printColumnsHeader()
        for row in self.rows:
            for col in self.colMeta:
                self.__rowsPrinter.logMessage(col['Format'] % row[col['Name']] + self.COL_SEPARATOR)
            self.__rowsPrinter.logMessage('\n')

    def printReplies(self):
        if False:
            for i in range(10):
                print('nop')
        for keys in list(self.replies.keys()):
            for (i, key) in enumerate(self.replies[keys]):
                if key['TokenType'] == TDS_ERROR_TOKEN:
                    error = 'ERROR(%s): Line %d: %s' % (key['ServerName'].decode('utf-16le'), key['LineNumber'], key['MsgText'].decode('utf-16le'))
                    self.lastError = SQLErrorException('ERROR: Line %d: %s' % (key['LineNumber'], key['MsgText'].decode('utf-16le')))
                elif key['TokenType'] == TDS_INFO_TOKEN:
                    LOG.info('INFO(%s): Line %d: %s' % (key['ServerName'].decode('utf-16le'), key['LineNumber'], key['MsgText'].decode('utf-16le')))
                elif key['TokenType'] == TDS_LOGINACK_TOKEN:
                    LOG.info('ACK: Result: %s - %s (%d%d %d%d) ' % (key['Interface'], key['ProgName'].decode('utf-16le'), key['MajorVer'], key['MinorVer'], key['BuildNumHi'], key['BuildNumLow']))
                elif key['TokenType'] == TDS_ENVCHANGE_TOKEN:
                    if key['Type'] in (TDS_ENVCHANGE_DATABASE, TDS_ENVCHANGE_LANGUAGE, TDS_ENVCHANGE_CHARSET, TDS_ENVCHANGE_PACKETSIZE):
                        record = TDS_ENVCHANGE_VARCHAR(key['Data'])
                        if record['OldValue'] == '':
                            record['OldValue'] = 'None'.encode('utf-16le')
                        elif record['NewValue'] == '':
                            record['NewValue'] = 'None'.encode('utf-16le')
                        if key['Type'] == TDS_ENVCHANGE_DATABASE:
                            _type = 'DATABASE'
                        elif key['Type'] == TDS_ENVCHANGE_LANGUAGE:
                            _type = 'LANGUAGE'
                        elif key['Type'] == TDS_ENVCHANGE_CHARSET:
                            _type = 'CHARSET'
                        elif key['Type'] == TDS_ENVCHANGE_PACKETSIZE:
                            _type = 'PACKETSIZE'
                        else:
                            _type = '%d' % key['Type']
                        LOG.info('ENVCHANGE(%s): Old Value: %s, New Value: %s' % (_type, record['OldValue'].decode('utf-16le'), record['NewValue'].decode('utf-16le')))

    def parseRow(self, token, tuplemode=False):
        if False:
            return 10
        if len(token) == 1:
            return 0
        row = [] if tuplemode else {}
        origDataLen = len(token['Data'])
        data = token['Data']
        for col in self.colMeta:
            _type = col['Type']
            if (_type == TDS_NVARCHARTYPE) | (_type == TDS_NCHARTYPE):
                charLen = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
                data = data[struct.calcsize('<H'):]
                if charLen != 65535:
                    value = data[:charLen].decode('utf-16le')
                    data = data[charLen:]
                else:
                    value = 'NULL'
            elif _type == TDS_BIGVARCHRTYPE:
                charLen = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
                data = data[struct.calcsize('<H'):]
                if charLen != 65535:
                    value = data[:charLen]
                    data = data[charLen:]
                else:
                    value = 'NULL'
            elif _type == TDS_GUIDTYPE:
                uuidLen = ord(data[0:1])
                data = data[1:]
                if uuidLen > 0:
                    uu = data[:uuidLen]
                    value = uuid.bin_to_string(uu)
                    data = data[uuidLen:]
                else:
                    value = 'NULL'
            elif (_type == TDS_NTEXTTYPE) | (_type == TDS_IMAGETYPE):
                charLen = ord(data[0:1])
                if charLen == 0:
                    value = 'NULL'
                    data = data[1:]
                else:
                    data = data[1 + charLen + 8:]
                    charLen = struct.unpack('<L', data[:struct.calcsize('<L')])[0]
                    data = data[struct.calcsize('<L'):]
                    if charLen != 65535:
                        if _type == TDS_NTEXTTYPE:
                            value = data[:charLen].decode('utf-16le')
                        else:
                            value = binascii.b2a_hex(data[:charLen])
                        data = data[charLen:]
                    else:
                        value = 'NULL'
            elif _type == TDS_TEXTTYPE:
                charLen = ord(data[0:1])
                if charLen == 0:
                    value = 'NULL'
                    data = data[1:]
                else:
                    data = data[1 + charLen + 8:]
                    charLen = struct.unpack('<L', data[:struct.calcsize('<L')])[0]
                    data = data[struct.calcsize('<L'):]
                    if charLen != 65535:
                        value = data[:charLen]
                        data = data[charLen:]
                    else:
                        value = 'NULL'
            elif (_type == TDS_BIGVARBINTYPE) | (_type == TDS_BIGBINARYTYPE):
                charLen = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
                data = data[struct.calcsize('<H'):]
                if charLen != 65535:
                    value = binascii.b2a_hex(data[:charLen])
                    data = data[charLen:]
                else:
                    value = 'NULL'
            elif (_type == TDS_DATETIM4TYPE) | (_type == TDS_DATETIMNTYPE) | (_type == TDS_DATETIMETYPE):
                value = ''
                if _type == TDS_DATETIMNTYPE:
                    if ord(data[0:1]) == 4:
                        _type = TDS_DATETIM4TYPE
                    elif ord(data[0:1]) == 8:
                        _type = TDS_DATETIMETYPE
                    else:
                        value = 'NULL'
                    data = data[1:]
                if _type == TDS_DATETIMETYPE:
                    dateValue = struct.unpack('<l', data[:4])[0]
                    data = data[4:]
                    if dateValue < 0:
                        baseDate = datetime.date(1753, 1, 1)
                    else:
                        baseDate = datetime.date(1900, 1, 1)
                    timeValue = struct.unpack('<L', data[:4])[0]
                    data = data[4:]
                elif _type == TDS_DATETIM4TYPE:
                    dateValue = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
                    data = data[struct.calcsize('<H'):]
                    timeValue = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
                    data = data[struct.calcsize('<H'):]
                    baseDate = datetime.date(1900, 1, 1)
                if value != 'NULL':
                    dateValue = datetime.date.fromordinal(baseDate.toordinal() + dateValue)
                    (hours, mod) = divmod(timeValue // 300, 60 * 60)
                    (minutes, second) = divmod(mod, 60)
                    value = datetime.datetime(dateValue.year, dateValue.month, dateValue.day, hours, minutes, second)
            elif (_type == TDS_INT4TYPE) | (_type == TDS_MONEY4TYPE) | (_type == TDS_FLT4TYPE):
                value = struct.unpack('<l', data[:struct.calcsize('<l')])[0]
                data = data[struct.calcsize('<l'):]
            elif _type == TDS_FLTNTYPE:
                valueSize = ord(data[:1])
                if valueSize == 4:
                    fmt = '<f'
                elif valueSize == 8:
                    fmt = '<d'
                data = data[1:]
                if valueSize > 0:
                    value = struct.unpack(fmt, data[:valueSize])[0]
                    data = data[valueSize:]
                else:
                    value = 'NULL'
            elif _type == TDS_MONEYNTYPE:
                valueSize = ord(data[:1])
                if valueSize == 4:
                    fmt = '<l'
                elif valueSize == 8:
                    fmt = '<q'
                data = data[1:]
                if valueSize > 0:
                    value = struct.unpack(fmt, data[:valueSize])[0]
                    if valueSize == 4:
                        value = float(value) // math.pow(10, 4)
                    else:
                        value = float(value >> 32) // math.pow(10, 4)
                    data = data[valueSize:]
                else:
                    value = 'NULL'
            elif _type == TDS_BIGCHARTYPE:
                charLen = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
                data = data[struct.calcsize('<H'):]
                value = data[:charLen]
                data = data[charLen:]
            elif (_type == TDS_INT8TYPE) | (_type == TDS_FLT8TYPE) | (_type == TDS_MONEYTYPE):
                value = struct.unpack('<q', data[:struct.calcsize('<q')])[0]
                data = data[struct.calcsize('<q'):]
            elif _type == TDS_INT2TYPE:
                value = struct.unpack('<H', data[:2])[0]
                data = data[2:]
            elif _type == TDS_DATENTYPE:
                valueSize = ord(data[:1])
                data = data[1:]
                if valueSize > 0:
                    dateBytes = data[:valueSize]
                    dateValue = struct.unpack('<L', '\x00' + dateBytes)[0]
                    value = datetime.date.fromtimestamp(dateValue)
                    data = data[valueSize:]
                else:
                    value = 'NULL'
            elif (_type == TDS_BITTYPE) | (_type == TDS_INT1TYPE):
                value = ord(data[:1])
                data = data[1:]
            elif (_type == TDS_NUMERICNTYPE) | (_type == TDS_DECIMALNTYPE):
                valueLen = ord(data[:1])
                data = data[1:]
                value = data[:valueLen]
                data = data[valueLen:]
                precision = ord(col['TypeData'][1:2])
                scale = ord(col['TypeData'][2:3])
                if valueLen > 0:
                    isPositiveSign = ord(value[0:1])
                    if valueLen - 1 == 2:
                        fmt = '<H'
                    elif valueLen - 1 == 4:
                        fmt = '<L'
                    elif valueLen - 1 == 8:
                        fmt = '<Q'
                    else:
                        value = 'TODO: Interpret TDS_NUMERICNTYPE correctly'
                    number = struct.unpack(fmt, value[1:])[0]
                    number //= math.pow(precision, scale)
                    if isPositiveSign == 0:
                        number *= -1
                    value = number
                else:
                    value = 'NULL'
            elif _type == TDS_BITNTYPE:
                valueSize = ord(data[:1])
                data = data[1:]
                if valueSize > 0:
                    if valueSize == 1:
                        value = ord(data[:valueSize])
                    else:
                        value = data[:valueSize]
                else:
                    value = 'NULL'
                data = data[valueSize:]
            elif _type == TDS_INTNTYPE:
                valueSize = ord(data[:1])
                if valueSize == 1:
                    fmt = '<B'
                elif valueSize == 2:
                    fmt = '<h'
                elif valueSize == 4:
                    fmt = '<l'
                elif valueSize == 8:
                    fmt = '<q'
                else:
                    fmt = ''
                data = data[1:]
                if valueSize > 0:
                    value = struct.unpack(fmt, data[:valueSize])[0]
                    data = data[valueSize:]
                else:
                    value = 'NULL'
            elif _type == TDS_SSVARIANTTYPE:
                raise Exception('ParseRow: SQL Variant type not yet supported :(')
            else:
                raise Exception('ParseROW: Unsupported data type: 0%x' % _type)
            if tuplemode:
                row.append(value)
            else:
                row[col['Name']] = value
        self.rows.append(row)
        return origDataLen - len(data)

    def parseColMetaData(self, token):
        if False:
            while True:
                i = 10
        count = token['Count']
        if count == 65535:
            return 0
        self.colMeta = []
        origDataLen = len(token['Data'])
        data = token['Data']
        for i in range(count):
            column = {}
            userType = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
            data = data[struct.calcsize('<H'):]
            flags = struct.unpack('<H', data[:struct.calcsize('<H')])[0]
            data = data[struct.calcsize('<H'):]
            colType = struct.unpack('<B', data[:struct.calcsize('<B')])[0]
            data = data[struct.calcsize('<B'):]
            if (colType == TDS_BITTYPE) | (colType == TDS_INT1TYPE) | (colType == TDS_INT2TYPE) | (colType == TDS_INT8TYPE) | (colType == TDS_DATETIMETYPE) | (colType == TDS_DATETIM4TYPE) | (colType == TDS_FLT4TYPE) | (colType == TDS_FLT8TYPE) | (colType == TDS_MONEYTYPE) | (colType == TDS_MONEY4TYPE) | (colType == TDS_DATENTYPE) | (colType == TDS_INT4TYPE):
                typeData = ''
            elif (colType == TDS_INTNTYPE) | (colType == TDS_TIMENTYPE) | (colType == TDS_DATETIME2NTYPE) | (colType == TDS_DATETIMEOFFSETNTYPE) | (colType == TDS_FLTNTYPE) | (colType == TDS_MONEYNTYPE) | (colType == TDS_GUIDTYPE) | (colType == TDS_BITNTYPE):
                typeData = ord(data[0:1])
                data = data[1:]
            elif colType == TDS_DATETIMNTYPE:
                typeData = ord(data[0:1])
                data = data[1:]
            elif (colType == TDS_BIGVARBINTYPE) | (colType == TDS_BIGBINARYTYPE) | (colType == TDS_NCHARTYPE) | (colType == TDS_NVARCHARTYPE) | (colType == TDS_BIGVARCHRTYPE) | (colType == TDS_BIGCHARTYPE):
                typeData = struct.unpack('<H', data[:2])[0]
                data = data[2:]
            elif (colType == TDS_DECIMALNTYPE) | (colType == TDS_NUMERICNTYPE) | (colType == TDS_DECIMALTYPE):
                typeData = data[:3]
                data = data[3:]
            elif (colType == TDS_IMAGETYPE) | (colType == TDS_TEXTTYPE) | (colType == TDS_XMLTYPE) | (colType == TDS_SSVARIANTTYPE) | (colType == TDS_NTEXTTYPE):
                typeData = struct.unpack('<L', data[:4])[0]
                data = data[4:]
            else:
                raise Exception('Unsupported data type: 0x%x' % colType)
            if (colType == TDS_NTEXTTYPE) | (colType == TDS_BIGCHARTYPE) | (colType == TDS_BIGVARCHRTYPE) | (colType == TDS_NCHARTYPE) | (colType == TDS_NVARCHARTYPE) | (colType == TDS_TEXTTYPE):
                data = data[5:]
            if (colType == TDS_IMAGETYPE) | (colType == TDS_TEXTTYPE) | (colType == TDS_NTEXTTYPE):
                dataLen = struct.unpack('<H', data[:2])[0]
                data = data[2:]
                data = data[dataLen * 2:]
            colNameLength = struct.unpack('<B', data[:struct.calcsize('<B')])[0]
            data = data[struct.calcsize('<B'):]
            colName = data[:colNameLength * 2].decode('utf-16le')
            data = data[colNameLength * 2:]
            column['Name'] = colName
            column['Type'] = colType
            column['TypeData'] = typeData
            column['Flags'] = flags
            self.colMeta.append(column)
        return origDataLen - len(data)

    def parseReply(self, tokens, tuplemode=False):
        if False:
            for i in range(10):
                print('nop')
        if len(tokens) == 0:
            return False
        replies = {}
        while len(tokens) > 0:
            tokenID = struct.unpack('B', tokens[0:1])[0]
            if tokenID == TDS_ERROR_TOKEN:
                token = TDS_INFO_ERROR(tokens)
            elif tokenID == TDS_RETURNSTATUS_TOKEN:
                token = TDS_RETURNSTATUS(tokens)
            elif tokenID == TDS_INFO_TOKEN:
                token = TDS_INFO_ERROR(tokens)
            elif tokenID == TDS_LOGINACK_TOKEN:
                token = TDS_LOGIN_ACK(tokens)
            elif tokenID == TDS_ENVCHANGE_TOKEN:
                token = TDS_ENVCHANGE(tokens)
                if token['Type'] is TDS_ENVCHANGE_PACKETSIZE:
                    record = TDS_ENVCHANGE_VARCHAR(token['Data'])
                    self.packetSize = int(record['NewValue'].decode('utf-16le'))
                elif token['Type'] is TDS_ENVCHANGE_DATABASE:
                    record = TDS_ENVCHANGE_VARCHAR(token['Data'])
                    self.currentDB = record['NewValue'].decode('utf-16le')
            elif (tokenID == TDS_DONEINPROC_TOKEN) | (tokenID == TDS_DONEPROC_TOKEN):
                token = TDS_DONEINPROC(tokens)
            elif tokenID == TDS_ORDER_TOKEN:
                token = TDS_ORDER(tokens)
            elif tokenID == TDS_ROW_TOKEN:
                token = TDS_ROW(tokens)
                tokenLen = self.parseRow(token, tuplemode)
                token['Data'] = token['Data'][:tokenLen]
            elif tokenID == TDS_COLMETADATA_TOKEN:
                token = TDS_COLMETADATA(tokens)
                tokenLen = self.parseColMetaData(token)
                token['Data'] = token['Data'][:tokenLen]
            elif tokenID == TDS_DONE_TOKEN:
                token = TDS_DONE(tokens)
            else:
                LOG.error('Unknown Token %x' % tokenID)
                return replies
            if (tokenID in replies) is not True:
                replies[tokenID] = list()
            replies[tokenID].append(token)
            tokens = tokens[len(token):]
        return replies

    def batch(self, cmd, tuplemode=False, wait=True):
        if False:
            while True:
                i = 10
        self.rows = []
        self.colMeta = []
        self.lastError = False
        self.sendTDS(TDS_SQL_BATCH, (cmd + '\r\n').encode('utf-16le'))
        if wait:
            tds = self.recvTDS()
            self.replies = self.parseReply(tds['Data'], tuplemode)
            return self.rows
        else:
            return True

    def batchStatement(self, cmd, tuplemode=False):
        if False:
            return 10
        self.rows = []
        self.colMeta = []
        self.lastError = False
        self.sendTDS(TDS_SQL_BATCH, (cmd + '\r\n').encode('utf-16le'))
    sql_query = batch

    def changeDB(self, db):
        if False:
            print('Hello World!')
        if db != self.currentDB:
            chdb = 'use %s' % db
            self.batch(chdb)
            self.printReplies()

    def RunSQLQuery(self, db, sql_query, tuplemode=False, wait=True, **kwArgs):
        if False:
            for i in range(10):
                print('nop')
        db = db or 'master'
        self.changeDB(db)
        self.printReplies()
        ret = self.batch(sql_query, tuplemode, wait)
        if wait:
            self.printReplies()
        if self.lastError:
            raise self.lastError
        if self.lastError:
            raise self.lastError
        return ret

    def RunSQLStatement(self, db, sql_query, wait=True, **kwArgs):
        if False:
            while True:
                i = 10
        self.RunSQLQuery(db, sql_query, wait=wait)
        if self.lastError:
            raise self.lastError
        return True