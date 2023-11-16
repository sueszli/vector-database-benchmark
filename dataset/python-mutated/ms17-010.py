from ctypes import *
import socket
import struct

class CMEModule:
    name = 'ms17-010'
    description = 'MS17-010, /!\\ not tested oustide home lab'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def options(self, context, module_options):
        if False:
            return 10
        ' '

    def on_login(self, context, connection):
        if False:
            for i in range(10):
                print('nop')
        if check(connection.host):
            context.log.highlight('VULNERABLE')
            context.log.highlight('Next step: https://www.rapid7.com/db/modules/exploit/windows/smb/ms17_010_eternalblue/')

class SMB_HEADER(Structure):
    """SMB Header decoder."""
    _pack_ = 1
    _fields_ = [('server_component', c_uint32), ('smb_command', c_uint8), ('error_class', c_uint8), ('reserved1', c_uint8), ('error_code', c_uint16), ('flags', c_uint8), ('flags2', c_uint16), ('process_id_high', c_uint16), ('signature', c_uint64), ('reserved2', c_uint16), ('tree_id', c_uint16), ('process_id', c_uint16), ('user_id', c_uint16), ('multiplex_id', c_uint16)]

    def __new__(self, buffer=None):
        if False:
            for i in range(10):
                print('nop')
        return self.from_buffer_copy(buffer)

def generate_smb_proto_payload(*protos):
    if False:
        i = 10
        return i + 15
    'Generate SMB Protocol. Pakcet protos in order.'
    hexdata = []
    for proto in protos:
        hexdata.extend(proto)
    return ''.join(hexdata)

def calculate_doublepulsar_xor_key(s):
    if False:
        for i in range(10):
            print('nop')
    'Calaculate Doublepulsar Xor Key'
    x = 2 * s ^ ((s & 65280 | s << 16) << 8 | (s >> 16 | s & 16711680) >> 8)
    x = x & 4294967295
    return x

def negotiate_proto_request():
    if False:
        while True:
            i = 10
    'Generate a negotiate_proto_request packet.'
    netbios = ['\x00', '\x00\x00T']
    smb_header = ['ÿSMB', 'r', '\x00\x00\x00\x00', '\x18', '\x01(', '\x00\x00', '\x00\x00\x00\x00\x00\x00\x00\x00', '\x00\x00', '\x00\x00', '/K', '\x00\x00', 'Å^']
    negotiate_proto_request = ['\x00', '1\x00', '\x02', 'LANMAN1.0\x00', '\x02', 'LM1.2X002\x00', '\x02', 'NT LANMAN 1.0\x00', '\x02', 'NT LM 0.12\x00']
    return generate_smb_proto_payload(netbios, smb_header, negotiate_proto_request)

def session_setup_andx_request():
    if False:
        for i in range(10):
            print('nop')
    'Generate session setuo andx request.'
    netbios = ['\x00', '\x00\x00c']
    smb_header = ['ÿSMB', 's', '\x00\x00\x00\x00', '\x18', '\x01 ', '\x00\x00', '\x00\x00\x00\x00\x00\x00\x00\x00', '\x00\x00', '\x00\x00', '/K', '\x00\x00', 'Å^']
    session_setup_andx_request = ['\r', 'ÿ', '\x00', '\x00\x00', 'ßÿ', '\x02\x00', '\x01\x00', '\x00\x00\x00\x00', '\x00\x00', '\x00\x00', '\x00\x00\x00\x00', '@\x00\x00\x00', '&\x00', '\x00', '.\x00', 'Windows 2000 2195\x00', 'Windows 2000 5.0\x00']
    return generate_smb_proto_payload(netbios, smb_header, session_setup_andx_request)

def tree_connect_andx_request(ip, userid):
    if False:
        i = 10
        return i + 15
    'Generate tree connect andx request.'
    netbios = ['\x00', '\x00\x00G']
    smb_header = ['ÿSMB', 'u', '\x00\x00\x00\x00', '\x18', '\x01 ', '\x00\x00', '\x00\x00\x00\x00\x00\x00\x00\x00', '\x00\x00', '\x00\x00', '/K', userid, 'Å^']
    ipc = '\\\\{}\\IPC$\x00'.format(ip)
    tree_connect_andx_request = ['\x04', 'ÿ', '\x00', '\x00\x00', '\x00\x00', '\x01\x00', '\x1a\x00', '\x00', ipc.encode(), '?????\x00']
    length = len(''.join(smb_header)) + len(''.join(tree_connect_andx_request))
    netbios[1] = struct.pack('>L', length)[-3:]
    return generate_smb_proto_payload(netbios, smb_header, tree_connect_andx_request)

def peeknamedpipe_request(treeid, processid, userid, multiplex_id):
    if False:
        i = 10
        return i + 15
    'Generate tran2 request'
    netbios = ['\x00', '\x00\x00J']
    smb_header = ['ÿSMB', '%', '\x00\x00\x00\x00', '\x18', '\x01(', '\x00\x00', '\x00\x00\x00\x00\x00\x00\x00\x00', '\x00\x00', treeid, processid, userid, multiplex_id]
    tran_request = ['\x10', '\x00\x00', '\x00\x00', 'ÿÿ', 'ÿÿ', '\x00', '\x00', '\x00\x00', '\x00\x00\x00\x00', '\x00\x00', '\x00\x00', 'J\x00', '\x00\x00', 'J\x00', '\x02', '\x00', '#\x00', '\x00\x00', '\x07\x00', '\\PIPE\\\x00']
    return generate_smb_proto_payload(netbios, smb_header, tran_request)

def trans2_request(treeid, processid, userid, multiplex_id):
    if False:
        print('Hello World!')
    'Generate trans2 request.'
    netbios = ['\x00', '\x00\x00O']
    smb_header = ['ÿSMB', '2', '\x00\x00\x00\x00', '\x18', '\x07À', '\x00\x00', '\x00\x00\x00\x00\x00\x00\x00\x00', '\x00\x00', treeid, processid, userid, multiplex_id]
    trans2_request = ['\x0f', '\x0c\x00', '\x00\x00', '\x01\x00', '\x00\x00', '\x00', '\x00', '\x00\x00', '¦Ù¤\x00', '\x00\x00', '\x0c\x00', 'B\x00', '\x00\x00', 'N\x00', '\x01', '\x00', '\x0e\x00', '\x00\x00', '\x0c\x00' + '\x00' * 12]
    return generate_smb_proto_payload(netbios, smb_header, trans2_request)

def check(ip, port=445):
    if False:
        while True:
            i = 10
    'Check if MS17_010 SMB Vulnerability exists.'
    try:
        buffersize = 1024
        timeout = 5.0
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect((ip, port))
        raw_proto = negotiate_proto_request()
        client.send(raw_proto)
        tcp_response = client.recv(buffersize)
        raw_proto = session_setup_andx_request()
        client.send(raw_proto)
        tcp_response = client.recv(buffersize)
        netbios = tcp_response[:4]
        smb_header = tcp_response[4:36]
        smb = SMB_HEADER(smb_header)
        user_id = struct.pack('<H', smb.user_id)
        session_setup_andx_response = tcp_response[36:]
        native_os = session_setup_andx_response[9:].split('\x00')[0]
        raw_proto = tree_connect_andx_request(ip, user_id)
        client.send(raw_proto)
        tcp_response = client.recv(buffersize)
        netbios = tcp_response[:4]
        smb_header = tcp_response[4:36]
        smb = SMB_HEADER(smb_header)
        tree_id = struct.pack('<H', smb.tree_id)
        process_id = struct.pack('<H', smb.process_id)
        user_id = struct.pack('<H', smb.user_id)
        multiplex_id = struct.pack('<H', smb.multiplex_id)
        raw_proto = peeknamedpipe_request(tree_id, process_id, user_id, multiplex_id)
        client.send(raw_proto)
        tcp_response = client.recv(buffersize)
        netbios = tcp_response[:4]
        smb_header = tcp_response[4:36]
        smb = SMB_HEADER(smb_header)
        nt_status = struct.pack('BBH', smb.error_class, smb.reserved1, smb.error_code)
        if nt_status == '\x05\x02\x00À':
            return True
        elif nt_status in ('\x08\x00\x00À', '"\x00\x00À'):
            return False
        else:
            return False
    except Exception as err:
        return False
    finally:
        client.close()