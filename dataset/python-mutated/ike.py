import struct
from typing import Any, Dict, Tuple, Union, cast
from ivre.types import NmapServiceMatch
from ivre.types.active import NmapPort
from ivre.utils import encode_hex, find_ike_vendor_id

class Values(Dict[int, str]):

    def __getitem__(self, item: int) -> str:
        if False:
            i = 10
            return i + 15
        try:
            return super().__getitem__(item)
        except KeyError:
            return 'UNKNOWN-%d' % item

class NumValues:

    def __getitem__(self, item: int) -> int:
        if False:
            i = 10
            return i + 15
        return item
DOI = Values({0: 'ISAKMP', 1: 'IPSEC', 2: 'GDOI'})
PROTO = Values({1: 'ISAKMP', 2: 'IPSEC_AH', 3: 'IPSEC_ESP', 4: 'IPCOMP'})
NOTIFICATION = Values({1: 'INVALID-PAYLOAD-TYPE', 2: 'DOI-NOT-SUPPORTED', 3: 'SITUATION-NOT-SUPPORTED', 4: 'INVALID-COOKIE', 5: 'INVALID-MAJOR-VERSION', 6: 'INVALID-MINOR-VERSION', 7: 'INVALID-EXCHANGE-TYPE', 8: 'INVALID-FLAGS', 9: 'INVALID-MESSAGE-ID', 10: 'INVALID-PROTOCOL-ID', 11: 'INVALID-SPI', 12: 'INVALID-TRANSFORM-ID', 13: 'ATTRIBUTES-NOT-SUPPORTED', 14: 'NO-PROPOSAL-CHOSEN', 15: 'BAD-PROPOSAL-SYNTAX', 16: 'PAYLOAD-MALFORMED', 17: 'INVALID-KEY-INFORMATION', 18: 'INVALID-ID-INFORMATION', 19: 'INVALID-CERT-ENCODING', 20: 'INVALID-CERTIFICATE', 21: 'CERT-TYPE-UNSUPPORTED', 22: 'INVALID-CERT-AUTHORITY', 23: 'INVALID-HASH-INFORMATION', 24: 'AUTHENTICATION-FAILED', 25: 'INVALID-SIGNATURE', 26: 'ADDRESS-NOTIFICATION', 27: 'NOTIFY-SA-LIFETIME', 28: 'CERTIFICATE-UNAVAILABLE', 29: 'UNSUPPORTED-EXCHANGE-TYPE', 30: 'UNEQUAL-PAYLOAD-LENGTHS'})
TRANSFORM_VALUES: Dict[int, Tuple[str, Union[Values, NumValues]]] = {1: ('Encryption', Values({1: 'DES-CBC', 2: 'IDEA-CBC', 3: 'Blowfish-CBC', 4: 'RC5-R16-B64-CBC', 5: '3DES-CBC', 6: 'CAST-CBC', 7: 'AES-CBC', 8: 'CAMELLIA-CBC'})), 2: ('Hash', Values({1: 'MD5', 2: 'SHA', 3: 'Tiger', 4: 'SHA2-256', 5: 'SHA2-384', 6: 'SHA2-512'})), 3: ('Authentication', Values({1: 'PSK', 2: 'DSS Signature', 3: 'RSA Signature', 4: 'RSA Encryption', 5: 'RSA Revised Encryption', 6: 'ElGamal Encryption', 7: 'ElGamal Revised Encryption', 8: 'ECDSA Signature', 9: 'ECDSA with SHA-256 on the P-256 curve', 10: 'ECDSA with SHA-384 on the P-384 curve', 11: 'ECDSA with SHA-512 on the P-521 curve', 64221: 'HybridInitRSA', 64222: 'HybridRespRSA', 64223: 'HybridInitDSS', 64224: 'HybridRespDSS', 65001: 'XAUTHInitPreShared or GSS-API using Kerberos', 65002: 'XAUTHRespPreShared or Generic GSS-API', 65003: 'XAUTHInitDSS or GSS-API with SPNEGO', 65004: 'XAUTHRespDSS or GSS-API using SPKM', 65005: 'XAUTHInitRSA', 65006: 'XAUTHRespRSA', 65007: 'XAUTHInitRSAEncryption', 65008: 'XAUTHRespRSAEncryption', 65009: 'XAUTHInitRSARevisedEncryption', 65010: 'XAUTHRespRSARevisedEncryptio'})), 4: ('GroupDesc', Values({1: '768MODPgr', 2: '1024MODPgr', 3: 'EC2Ngr155', 4: 'EC2Ngr185', 5: '1536MODPgr', 14: '2048MODPgr', 15: '3072MODPgr', 16: '4096MODPgr', 17: '6144MODPgr', 18: '8192MODPgr'})), 5: ('GroupType', Values({1: 'MODP', 2: 'ECP', 3: 'EC2N'})), 6: ('GroupPrime', NumValues()), 7: ('GroupGenerator1', NumValues()), 8: ('GroupGenerator2', NumValues()), 9: ('GroupCurveA', NumValues()), 10: ('GroupCurveB', NumValues()), 11: ('LifeType', Values({1: 'Seconds', 2: 'Kilobytes'})), 12: ('LifeDuration', NumValues()), 13: ('PRF', NumValues()), 14: ('KeyLength', NumValues()), 15: ('FieldSize', NumValues()), 16: ('GroupOrder', NumValues())}

def info_from_notification(payload: bytes, _: NmapServiceMatch, output: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    payload_len = len(payload)
    if payload_len < 12:
        output.setdefault('protocol', []).append('ISAKMP: Notification payload to short (%d bytes)' % payload_len)
        return
    output.update({'DOI': DOI[struct.unpack('>I', payload[4:8])[0]], 'protocol_id': PROTO[payload[8]], 'notification_type': NOTIFICATION[struct.unpack('>H', payload[10:12])[0]]})

def info_from_vendorid(payload: bytes, service: NmapServiceMatch, output: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    name = find_ike_vendor_id(payload[4:])
    if name is not None:
        if name.startswith(b'Windows-'):
            service['service_product'] = 'Microsoft/Cisco IPsec'
            service['service_version'] = name.decode().replace('-', ' ')
            service['service_ostype'] = 'Windows'
        elif name == b'Windows':
            service['service_product'] = 'Microsoft/Cisco IPsec'
            service['service_ostype'] = 'Windows'
        elif name.startswith(b'Firewall-1 '):
            service['service_product'] = 'Checkpoint VPN-1/Firewall-1'
            service['service_version'] = name.decode().split(None, 1)[1]
            service['service_devicetype'] = 'security-misc'
        elif name.startswith(b'SSH IPSEC Express '):
            service['service_product'] = 'SSH Communications Security IPSec Express'
            service['service_version'] = name.decode().split(None, 3)[3]
        elif name.startswith(b'SSH Sentinel'):
            service['service_product'] = 'SSH Communications Security Sentinel'
            version = name[13:].decode()
            if version:
                service['service_version'] = version
        elif name.startswith(b'SSH QuickSec'):
            service['service_product'] = 'SSH Communications Security QuickSec'
            version = name[13:].decode()
            if version:
                service['service_version'] = version
        elif name.startswith(b'Cisco VPN Concentrator'):
            service['service_product'] = 'Cisco VPN Concentrator'
            version = name[24:-1].decode()
            if version:
                service['service_version'] = version
        elif name.startswith(b'SafeNet SoftRemote'):
            service['service_product'] = 'SafeNet Remote'
            version = name[19:].decode()
            if version:
                service['service_version'] = version
        elif name == b'KAME/racoon':
            service['service_product'] = 'KAME/racoon/IPsec Tools'
        elif name == b'Nortel Contivity':
            service['service_product'] = 'Nortel Contivity'
            service['service_devicetype'] = 'firewall'
        elif name.startswith(b'SonicWall-'):
            service['service_product'] = 'SonicWall'
        elif name.startswith(b'strongSwan'):
            service['service_product'] = 'strongSwan'
            service['service_version'] = name[11:].decode() or '4.3.6'
            service['service_ostype'] = 'Unix'
        elif name == b'ZyXEL ZyWall USG 100':
            service['service_product'] = 'ZyXEL ZyWALL USG 100'
            service['service_devicetype'] = 'firewall'
        elif name.startswith(b'Linux FreeS/WAN '):
            service['service_product'] = 'FreeS/WAN'
            service['service_version'] = name.decode().split(None, 2)[2]
            service['service_ostype'] = 'Unix'
        elif name.startswith(b'Openswan ') or name.startswith(b'Linux Openswan '):
            service['service_product'] = 'Openswan'
            extra_info = name.split(b'Openswan ', 1)[1].decode().split(None, 1)
            service['service_version'] = extra_info[0]
            if len(extra_info) == 2:
                service['service_extrainfo'] = extra_info[1]
            service['service_ostype'] = 'Unix'
        elif name in [b'FreeS/WAN or OpenSWAN', b'FreeS/WAN or OpenSWAN or Libreswan']:
            service['service_product'] = 'FreeS/WAN or Openswan or Libreswan'
            service['service_ostype'] = 'Unix'
        elif name.startswith(b'Libreswan '):
            service['service_product'] = 'Libreswan'
            service['service_version'] = name.decode().split(None, 1)[1]
            service['service_ostype'] = 'Unix'
        elif name == b'OpenPGP':
            service['service_product'] = name.decode()
        elif name in [b'FortiGate', b'ZyXEL ZyWALL Router', b'ZyXEL ZyWALL USG 100']:
            service['service_product'] = name.decode()
            service['service_devicetype'] = 'firewall'
        elif name.startswith(b'Netscreen-'):
            service['service_product'] = 'Juniper'
            service['service_ostype'] = 'NetScreen OS'
            service['service_devicetype'] = 'firewall'
        elif name.startswith(b'StoneGate-'):
            service['service_product'] = 'StoneGate'
            service['service_devicetype'] = 'firewall'
        elif name.startswith(b'Symantec-Raptor'):
            service['service_product'] = 'Symantec-Raptor'
            version = name[16:].decode()
            if version:
                service['service_version'] = version
            service['service_devicetype'] = 'firewall'
        elif name == b'Teldat':
            service['service_product'] = name.decode()
            service['service_devicetype'] = 'broadband router'
    entry = {'value': encode_hex(payload[4:]).decode()}
    if name is not None:
        entry['name'] = name.decode()
    output.setdefault('vendor_ids', []).append(entry)

def info_from_sa(payload: bytes, _: NmapServiceMatch, output: Dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    payload_len = len(payload)
    if payload_len < 20:
        output.setdefault('protocol', []).append('ISAKMP: SA payload to short (%d bytes)' % payload_len)
        return
    output.update({'DOI': DOI[struct.unpack('>I', payload[4:8])[0]]})
    payload = payload[20:]
    payload_type = 3
    while payload_type == 3 and payload:
        transform = {}
        payload_type = payload[0]
        payload_length = struct.unpack('>H', payload[2:4])[0]
        data = payload[8:payload_length]
        payload = payload[payload_length:]
        while data:
            (transf_type, value) = struct.unpack('>HH', data[:4])
            data = data[4:]
            if transf_type & 32768:
                transf_type &= 32767
            else:
                value_length = value
                if value_length > len(data):
                    output.setdefault('protocol', []).append('invalid transform length: %d' % value_length)
                    break
                value = 0
                for val in data[:value_length]:
                    value = value * 256 + val
            try:
                (transf_type, value_decoder) = TRANSFORM_VALUES[transf_type]
            except KeyError:
                transf_type = 'UNKNOWN-%d' % transf_type
            else:
                value = value_decoder[value]
            transform[transf_type] = value
        if transform:
            output.setdefault('transforms', []).append(transform)
    if payload:
        output.setdefault('protocol', []).append('unexpected payload in transforms: %r' % payload)
PAYLOADS = {1: (info_from_sa, 'SA'), 11: (info_from_notification, 'Notification'), 13: (info_from_vendorid, 'Vendor ID')}

def analyze_ike_payload(payload: bytes, probe: str='ike') -> NmapPort:
    if False:
        for i in range(10):
            print('nop')
    service: NmapServiceMatch = {}
    output: Dict[str, Any] = {}
    if probe == 'ike-ipsec-nat-t':
        if payload.startswith(b'\x00\x00\x00\x00'):
            payload = payload[4:]
        else:
            output.setdefault('protocol', []).append('ike-ipsec-nat-t: missing non-ESP marker')
    payload_len = len(payload)
    if payload_len < 28:
        return {}
    if not payload.startswith(b'\x00\x11"3'):
        return {}
    payload_len_proto = struct.unpack('>I', payload[24:28])[0]
    if payload_len < payload_len_proto:
        return {}
    payload_type = payload[16]
    payload = payload[28:]
    while payload_type and len(payload) >= 4:
        payload_length = struct.unpack('>H', payload[2:4])[0]
        if payload_type in PAYLOADS:
            (specific_parser, type_name) = PAYLOADS[payload_type]
            output.setdefault('type', []).append(type_name)
            specific_parser(payload[:payload_length], service, output)
        (payload_type, payload) = (payload[0], payload[payload_length:])
    if service.get('service_version') == 'Unknown Vsn':
        del service['service_version']
    if not output:
        return {}
    txtoutput = []
    if 'transforms' in output:
        txtoutput.append('Transforms:')
        for tr in output['transforms']:
            txtoutput.append('  - %s' % ', '.join(('%s: %s' % (key, value) for (key, value) in sorted(tr.items()))))
    if 'vendor_ids' in output:
        txtoutput.append('Vendor IDs:')
        for vid in output['vendor_ids']:
            txtoutput.append('  - %s' % vid.get('name', vid['value']))
    if 'notification_type' in output:
        txtoutput.append('Notification: %s' % output['notification_type'])
    result: NmapPort = {'service_name': 'isakmp', 'scripts': [{'id': 'ike-info', 'output': '\n'.join(txtoutput), 'ike-info': output}]}
    result.update(cast(NmapPort, service))
    return result