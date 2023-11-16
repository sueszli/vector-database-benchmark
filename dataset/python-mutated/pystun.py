import binascii
import logging
import random
import socket
__version__ = '0.1.0'
log = logging.getLogger('pystun')
STUN_SERVERS = ('stun.ekiga.net', 'stun.ideasip.com', 'stun.voiparound.com', 'stun.voipbuster.com', 'stun.voipstunt.com', 'stun.voxgratia.org')
stun_servers_list = STUN_SERVERS
DEFAULTS = {'stun_port': 3478, 'source_ip': '0.0.0.0', 'source_port': 54320}
MappedAddress = '0001'
ResponseAddress = '0002'
ChangeRequest = '0003'
SourceAddress = '0004'
ChangedAddress = '0005'
Username = '0006'
Password = '0007'
MessageIntegrity = '0008'
ErrorCode = '0009'
UnknownAttribute = '000A'
ReflectedFrom = '000B'
XorOnly = '0021'
XorMappedAddress = '8020'
ServerName = '8022'
SecondaryAddress = '8050'
BindRequestMsg = '0001'
BindResponseMsg = '0101'
BindErrorResponseMsg = '0111'
SharedSecretRequestMsg = '0002'
SharedSecretResponseMsg = '0102'
SharedSecretErrorResponseMsg = '0112'
dictAttrToVal = {'MappedAddress': MappedAddress, 'ResponseAddress': ResponseAddress, 'ChangeRequest': ChangeRequest, 'SourceAddress': SourceAddress, 'ChangedAddress': ChangedAddress, 'Username': Username, 'Password': Password, 'MessageIntegrity': MessageIntegrity, 'ErrorCode': ErrorCode, 'UnknownAttribute': UnknownAttribute, 'ReflectedFrom': ReflectedFrom, 'XorOnly': XorOnly, 'XorMappedAddress': XorMappedAddress, 'ServerName': ServerName, 'SecondaryAddress': SecondaryAddress}
dictMsgTypeToVal = {'BindRequestMsg': BindRequestMsg, 'BindResponseMsg': BindResponseMsg, 'BindErrorResponseMsg': BindErrorResponseMsg, 'SharedSecretRequestMsg': SharedSecretRequestMsg, 'SharedSecretResponseMsg': SharedSecretResponseMsg, 'SharedSecretErrorResponseMsg': SharedSecretErrorResponseMsg}
dictValToMsgType = {}
dictValToAttr = {}
Blocked = 'Blocked'
OpenInternet = 'Open Internet'
FullCone = 'Full Cone'
SymmetricUDPFirewall = 'Symmetric UDP Firewall'
RestricNAT = 'Restric NAT'
RestricPortNAT = 'Restric Port NAT'
SymmetricNAT = 'Symmetric NAT'
ChangedAddressError = 'Meet an error, when do Test1 on Changed IP and Port'

def _initialize():
    if False:
        print('Hello World!')
    items = list(dictAttrToVal.items())
    for i in range(len(items)):
        dictValToAttr.update({items[i][1]: items[i][0]})
    items = list(dictMsgTypeToVal.items())
    for i in range(len(items)):
        dictValToMsgType.update({items[i][1]: items[i][0]})

def gen_tran_id():
    if False:
        for i in range(10):
            print('nop')
    a = ''.join((random.choice('0123456789ABCDEF') for i in range(32)))
    return a

def stun_test(sock, host, port, source_ip, source_port, send_data=''):
    if False:
        return 10
    retVal = {'Resp': False, 'ExternalIP': None, 'ExternalPort': None, 'SourceIP': None, 'SourcePort': None, 'ChangedIP': None, 'ChangedPort': None}
    str_len = '%#04d' % (len(send_data) / 2)
    tranid = gen_tran_id()
    str_data = ''.join([BindRequestMsg, str_len, tranid, send_data])
    data = binascii.a2b_hex(str_data)
    recvCorr = False
    while not recvCorr:
        recieved = False
        count = 3
        while not recieved:
            log.debug('sendto: %s', (host, port))
            try:
                sock.sendto(data, (host, port))
            except socket.gaierror:
                retVal['Resp'] = False
                return retVal
            try:
                (buf, addr) = sock.recvfrom(2048)
                log.debug('recvfrom: %s', addr)
                recieved = True
            except Exception:
                recieved = False
                if count > 0:
                    count -= 1
                else:
                    retVal['Resp'] = False
                    return retVal
        msgtype = binascii.b2a_hex(buf[0:2]).decode()
        msgtranid = binascii.b2a_hex(buf[4:20]).decode()
        bind_resp_msg = dictValToMsgType[msgtype] == 'BindResponseMsg'
        tranid_match = tranid.upper() == msgtranid.upper()
        if bind_resp_msg and tranid_match:
            recvCorr = True
            retVal['Resp'] = True
            len_message = int(binascii.b2a_hex(buf[2:4]), 16)
            len_remain = len_message
            base = 20
            while len_remain:
                attr_type = binascii.b2a_hex(buf[base:base + 2]).decode()
                attr_len = int(binascii.b2a_hex(buf[base + 2:base + 4]), 16)
                if attr_type == MappedAddress:
                    port = int(binascii.b2a_hex(buf[base + 6:base + 8]), 16)
                    ip = '.'.join([str(int(binascii.b2a_hex(buf[base + 8:base + 9]), 16)), str(int(binascii.b2a_hex(buf[base + 9:base + 10]), 16)), str(int(binascii.b2a_hex(buf[base + 10:base + 11]), 16)), str(int(binascii.b2a_hex(buf[base + 11:base + 12]), 16))])
                    retVal['ExternalIP'] = ip
                    retVal['ExternalPort'] = port
                if attr_type == SourceAddress:
                    port = int(binascii.b2a_hex(buf[base + 6:base + 8]), 16)
                    ip = '.'.join([str(int(binascii.b2a_hex(buf[base + 8:base + 9]), 16)), str(int(binascii.b2a_hex(buf[base + 9:base + 10]), 16)), str(int(binascii.b2a_hex(buf[base + 10:base + 11]), 16)), str(int(binascii.b2a_hex(buf[base + 11:base + 12]), 16))])
                    retVal['SourceIP'] = ip
                    retVal['SourcePort'] = port
                if attr_type == ChangedAddress:
                    port = int(binascii.b2a_hex(buf[base + 6:base + 8]), 16)
                    ip = '.'.join([str(int(binascii.b2a_hex(buf[base + 8:base + 9]), 16)), str(int(binascii.b2a_hex(buf[base + 9:base + 10]), 16)), str(int(binascii.b2a_hex(buf[base + 10:base + 11]), 16)), str(int(binascii.b2a_hex(buf[base + 11:base + 12]), 16))])
                    retVal['ChangedIP'] = ip
                    retVal['ChangedPort'] = port
                base = base + 4 + attr_len
                len_remain = len_remain - (4 + attr_len)
    return retVal

def get_nat_type(s, source_ip, source_port, stun_host=None, stun_port=3478):
    if False:
        i = 10
        return i + 15
    _initialize()
    port = stun_port
    log.debug('Do Test1')
    resp = False
    if stun_host:
        ret = stun_test(s, stun_host, port, source_ip, source_port)
        resp = ret['Resp']
    else:
        for stun_host in stun_servers_list:
            log.debug('Trying STUN host: %s', stun_host)
            ret = stun_test(s, stun_host, port, source_ip, source_port)
            resp = ret['Resp']
            if resp:
                break
    log.debug('stun test result: %s', ret)
    return ret

def get_ip_info(source_ip='0.0.0.0', source_port=54320, stun_host=None, stun_port=3478):
    if False:
        for i in range(10):
            print('nop')
    socket.setdefaulttimeout(2)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((source_ip, source_port))
    nat = get_nat_type(s, source_ip, source_port, stun_host=stun_host, stun_port=stun_port)
    external_ip = nat['ExternalIP']
    external_port = nat['ExternalPort']
    s.close()
    return (external_ip, external_port)