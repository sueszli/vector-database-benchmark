from __future__ import print_function
from struct import pack, unpack, calcsize
GSS_API_SPNEGO_UUID = '+\x06\x01\x05\x05\x02'
ASN1_SEQUENCE = 48
ASN1_AID = 96
ASN1_OID = 6
ASN1_OCTET_STRING = 4
ASN1_MECH_TYPE = 160
ASN1_MECH_TOKEN = 162
ASN1_SUPPORTED_MECH = 161
ASN1_RESPONSE_TOKEN = 162
ASN1_ENUMERATED = 10
MechTypes = {'+\x06\x01\x04\x01\x827\x02\x02\x1e': 'SNMPv2-SMI::enterprises.311.2.2.30', '+\x06\x01\x04\x01\x827\x02\x02\n': 'NTLMSSP - Microsoft NTLM Security Support Provider', '*\x86H\x82÷\x12\x01\x02\x02': 'MS KRB5 - Microsoft Kerberos 5', '*\x86H\x86÷\x12\x01\x02\x02': 'KRB5 - Kerberos 5', '*\x86H\x86÷\x12\x01\x02\x02\x03': 'KRB5 - Kerberos 5 - User to User'}
TypesMech = dict(((v, k) for (k, v) in MechTypes.iteritems()))

def asn1encode(data=''):
    if False:
        while True:
            i = 10
    if 0 <= len(data) <= 127:
        res = pack('B', len(data)) + data
    elif 128 <= len(data) <= 255:
        res = pack('BB', 129, len(data)) + data
    elif 256 <= len(data) <= 65535:
        res = pack('!BH', 130, len(data)) + data
    elif 65536 <= len(data) <= 16777215:
        res = pack('!BBH', 131, len(data) >> 16, len(data) & 65535) + data
    elif 16777216 <= len(data) <= 4294967295:
        res = pack('!BL', 132, len(data)) + data
    else:
        raise Exception('Error in asn1encode')
    return str(res)

def asn1decode(data=''):
    if False:
        i = 10
        return i + 15
    len1 = unpack('B', data[:1])[0]
    data = data[1:]
    if len1 == 129:
        pad = calcsize('B')
        len2 = unpack('B', data[:pad])[0]
        data = data[pad:]
        ans = data[:len2]
    elif len1 == 130:
        pad = calcsize('H')
        len2 = unpack('!H', data[:pad])[0]
        data = data[pad:]
        ans = data[:len2]
    elif len1 == 131:
        pad = calcsize('B') + calcsize('!H')
        (len2, len3) = unpack('!BH', data[:pad])
        data = data[pad:]
        ans = data[:len2 << 16 + len3]
    elif len1 == 132:
        pad = calcsize('!L')
        len2 = unpack('!L', data[:pad])[0]
        data = data[pad:]
        ans = data[:len2]
    else:
        pad = 0
        ans = data[:len1]
    return (ans, len(ans) + pad + 1)

class GSSAPI:

    def __init__(self, data=None):
        if False:
            print('Hello World!')
        self.fields = {}
        self['UUID'] = GSS_API_SPNEGO_UUID
        if data:
            self.fromString(data)
        pass

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        self.fields[key] = value

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self.fields[key]

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        del self.fields[key]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.getData())

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return len(self.getData())

    def fromString(self, data=None):
        if False:
            i = 10
            return i + 15
        next_byte = unpack('B', data[:1])[0]
        if next_byte != ASN1_AID:
            raise Exception('Unknown AID=%x' % next_byte)
        data = data[1:]
        (decode_data, total_bytes) = asn1decode(data)
        next_byte = unpack('B', decode_data[:1])[0]
        if next_byte != ASN1_OID:
            raise Exception('OID tag not found %x' % next_byte)
        decode_data = decode_data[1:]
        (uuid, total_bytes) = asn1decode(decode_data)
        self['OID'] = uuid
        self['Payload'] = decode_data[total_bytes:]

    def dump(self):
        if False:
            i = 10
            return i + 15
        for i in self.fields.keys():
            print('%s: {%r}' % (i, self[i]))

    def getData(self):
        if False:
            i = 10
            return i + 15
        ans = pack('B', ASN1_AID)
        ans += asn1encode(pack('B', ASN1_OID) + asn1encode(self['UUID']) + self['Payload'])
        return ans

class SPNEGO_NegTokenResp:
    SPNEGO_NEG_TOKEN_RESP = 161
    SPNEGO_NEG_TOKEN_TARG = 160

    def __init__(self, data=None):
        if False:
            return 10
        self.fields = {}
        if data:
            self.fromString(data)
        pass

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.fields[key] = value

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self.fields[key]

    def __delitem__(self, key):
        if False:
            return 10
        del self.fields[key]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.getData())

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return len(self.getData())

    def fromString(self, data=0):
        if False:
            i = 10
            return i + 15
        payload = data
        next_byte = unpack('B', payload[:1])[0]
        if next_byte != SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP:
            raise Exception('NegTokenResp not found %x' % next_byte)
        payload = payload[1:]
        (decode_data, total_bytes) = asn1decode(payload)
        next_byte = unpack('B', decode_data[:1])[0]
        if next_byte != ASN1_SEQUENCE:
            raise Exception('SEQUENCE tag not found %x' % next_byte)
        decode_data = decode_data[1:]
        (decode_data, total_bytes) = asn1decode(decode_data)
        next_byte = unpack('B', decode_data[:1])[0]
        if next_byte != ASN1_MECH_TYPE:
            if next_byte != ASN1_RESPONSE_TOKEN:
                raise Exception('MechType/ResponseToken tag not found %x' % next_byte)
        else:
            decode_data2 = decode_data[1:]
            (decode_data2, total_bytes) = asn1decode(decode_data2)
            next_byte = unpack('B', decode_data2[:1])[0]
            if next_byte != ASN1_ENUMERATED:
                raise Exception('Enumerated tag not found %x' % next_byte)
            (item, total_bytes2) = asn1decode(decode_data)
            self['NegResult'] = item
            decode_data = decode_data[1:]
            decode_data = decode_data[total_bytes:]
            if len(decode_data) == 0:
                return
            next_byte = unpack('B', decode_data[:1])[0]
            if next_byte != ASN1_SUPPORTED_MECH:
                if next_byte != ASN1_RESPONSE_TOKEN:
                    raise Exception('Supported Mech/ResponseToken tag not found %x' % next_byte)
            else:
                decode_data2 = decode_data[1:]
                (decode_data2, total_bytes) = asn1decode(decode_data2)
                next_byte = unpack('B', decode_data2[:1])[0]
                if next_byte != ASN1_OID:
                    raise Exception('OID tag not found %x' % next_byte)
                decode_data2 = decode_data2[1:]
                (item, total_bytes2) = asn1decode(decode_data2)
                self['SupportedMech'] = item
                decode_data = decode_data[1:]
                decode_data = decode_data[total_bytes:]
                next_byte = unpack('B', decode_data[:1])[0]
                if next_byte != ASN1_RESPONSE_TOKEN:
                    raise Exception('Response token tag not found %x' % next_byte)
        decode_data = decode_data[1:]
        (decode_data, total_bytes) = asn1decode(decode_data)
        next_byte = unpack('B', decode_data[:1])[0]
        if next_byte != ASN1_OCTET_STRING:
            raise Exception('Octet string token tag not found %x' % next_byte)
        decode_data = decode_data[1:]
        (decode_data, total_bytes) = asn1decode(decode_data)
        self['ResponseToken'] = decode_data

    def dump(self):
        if False:
            return 10
        for i in self.fields.keys():
            print('%s: {%r}' % (i, self[i]))

    def getData(self):
        if False:
            return 10
        ans = pack('B', SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP)
        if 'NegResult' in self.fields and 'SupportedMech' in self.fields:
            ans += asn1encode(pack('B', ASN1_SEQUENCE) + asn1encode(pack('B', SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_TARG) + asn1encode(pack('B', ASN1_ENUMERATED) + asn1encode(self['NegResult'])) + pack('B', ASN1_SUPPORTED_MECH) + asn1encode(pack('B', ASN1_OID) + asn1encode(self['SupportedMech'])) + pack('B', ASN1_RESPONSE_TOKEN) + asn1encode(pack('B', ASN1_OCTET_STRING) + asn1encode(self['ResponseToken']))))
        elif 'NegResult' in self.fields:
            ans += asn1encode(pack('B', ASN1_SEQUENCE) + asn1encode(pack('B', SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_TARG) + asn1encode(pack('B', ASN1_ENUMERATED) + asn1encode(self['NegResult']))))
        else:
            ans += asn1encode(pack('B', ASN1_SEQUENCE) + asn1encode(pack('B', ASN1_RESPONSE_TOKEN) + asn1encode(pack('B', ASN1_OCTET_STRING) + asn1encode(self['ResponseToken']))))
        return ans

class SPNEGO_NegTokenInit(GSSAPI):
    SPNEGO_NEG_TOKEN_INIT = 160

    def fromString(self, data=0):
        if False:
            while True:
                i = 10
        GSSAPI.fromString(self, data)
        payload = self['Payload']
        next_byte = unpack('B', payload[:1])[0]
        if next_byte != SPNEGO_NegTokenInit.SPNEGO_NEG_TOKEN_INIT:
            raise Exception('NegTokenInit not found %x' % next_byte)
        payload = payload[1:]
        (decode_data, total_bytes) = asn1decode(payload)
        next_byte = unpack('B', decode_data[:1])[0]
        if next_byte != ASN1_SEQUENCE:
            raise Exception('SEQUENCE tag not found %x' % next_byte)
        decode_data = decode_data[1:]
        (decode_data, total_bytes2) = asn1decode(decode_data)
        next_byte = unpack('B', decode_data[:1])[0]
        if next_byte != ASN1_MECH_TYPE:
            raise Exception('MechType tag not found %x' % next_byte)
        decode_data = decode_data[1:]
        remaining_data = decode_data
        (decode_data, total_bytes3) = asn1decode(decode_data)
        next_byte = unpack('B', decode_data[:1])[0]
        if next_byte != ASN1_SEQUENCE:
            raise Exception('SEQUENCE tag not found %x' % next_byte)
        decode_data = decode_data[1:]
        (decode_data, total_bytes4) = asn1decode(decode_data)
        self['MechTypes'] = []
        while decode_data:
            next_byte = unpack('B', decode_data[:1])[0]
            if next_byte != ASN1_OID:
                break
            decode_data = decode_data[1:]
            (item, total_bytes) = asn1decode(decode_data)
            self['MechTypes'].append(item)
            decode_data = decode_data[total_bytes:]
        decode_data = remaining_data[total_bytes3:]
        if len(decode_data) > 0:
            next_byte = unpack('B', decode_data[:1])[0]
            if next_byte == ASN1_MECH_TOKEN:
                decode_data = decode_data[1:]
                (decode_data, total_bytes) = asn1decode(decode_data)
                next_byte = unpack('B', decode_data[:1])[0]
                if next_byte == ASN1_OCTET_STRING:
                    decode_data = decode_data[1:]
                    (decode_data, total_bytes) = asn1decode(decode_data)
                    self['MechToken'] = decode_data

    def getData(self):
        if False:
            print('Hello World!')
        mechTypes = ''
        for i in self['MechTypes']:
            mechTypes += pack('B', ASN1_OID)
            mechTypes += asn1encode(i)
        mechToken = ''
        if 'MechToken' in self.fields:
            mechToken = pack('B', ASN1_MECH_TOKEN) + asn1encode(pack('B', ASN1_OCTET_STRING) + asn1encode(self['MechToken']))
        ans = pack('B', SPNEGO_NegTokenInit.SPNEGO_NEG_TOKEN_INIT)
        ans += asn1encode(pack('B', ASN1_SEQUENCE) + asn1encode(pack('B', ASN1_MECH_TYPE) + asn1encode(pack('B', ASN1_SEQUENCE) + asn1encode(mechTypes)) + mechToken))
        self['Payload'] = ans
        return GSSAPI.getData(self)