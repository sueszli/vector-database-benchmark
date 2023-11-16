import binascii
from .x509 import ASN1_Node, bytestr_to_int, decode_OID

def a2b_base64(s):
    if False:
        return 10
    try:
        b = bytearray(binascii.a2b_base64(s))
    except Exception as e:
        raise SyntaxError('base64 error: %s' % e)
    return b

def b2a_base64(b):
    if False:
        return 10
    return binascii.b2a_base64(b)

def dePem(s, name):
    if False:
        while True:
            i = 10
    'Decode a PEM string into a bytearray of its payload.\n\n    The input must contain an appropriate PEM prefix and postfix\n    based on the input name string, e.g. for name="CERTIFICATE":\n\n    -----BEGIN CERTIFICATE-----\n    MIIBXDCCAUSgAwIBAgIBADANBgkqhkiG9w0BAQUFADAPMQ0wCwYDVQQDEwRUQUNL\n    ...\n    KoZIhvcNAQEFBQADAwA5kw==\n    -----END CERTIFICATE-----\n\n    The first such PEM block in the input will be found, and its\n    payload will be base64 decoded and returned.\n    '
    prefix = '-----BEGIN %s-----' % name
    postfix = '-----END %s-----' % name
    start = s.find(prefix)
    if start == -1:
        raise SyntaxError('Missing PEM prefix')
    end = s.find(postfix, start + len(prefix))
    if end == -1:
        raise SyntaxError('Missing PEM postfix')
    s = s[start + len('-----BEGIN %s-----' % name):end]
    retBytes = a2b_base64(s)
    return retBytes

def dePemList(s, name):
    if False:
        for i in range(10):
            print('nop')
    'Decode a sequence of PEM blocks into a list of bytearrays.\n\n    The input must contain any number of PEM blocks, each with the appropriate\n    PEM prefix and postfix based on the input name string, e.g. for\n    name="TACK BREAK SIG".  Arbitrary text can appear between and before and\n    after the PEM blocks.  For example:\n\n    " Created by TACK.py 0.9.3 Created at 2012-02-01T00:30:10Z -----BEGIN TACK\n    BREAK SIG-----\n    ATKhrz5C6JHJW8BF5fLVrnQss6JnWVyEaC0p89LNhKPswvcC9/s6+vWLd9snYTUv\n    YMEBdw69PUP8JB4AdqA3K6Ap0Fgd9SSTOECeAKOUAym8zcYaXUwpk0+WuPYa7Zmm\n    SkbOlK4ywqt+amhWbg9txSGUwFO5tWUHT3QrnRlE/e3PeNFXLx5Bckg= -----END TACK\n    BREAK SIG----- Created by TACK.py 0.9.3 Created at 2012-02-01T00:30:11Z\n    -----BEGIN TACK BREAK SIG-----\n    ATKhrz5C6JHJW8BF5fLVrnQss6JnWVyEaC0p89LNhKPswvcC9/s6+vWLd9snYTUv\n    YMEBdw69PUP8JB4AdqA3K6BVCWfcjN36lx6JwxmZQncS6sww7DecFO/qjSePCxwM\n    +kdDqX/9/183nmjx6bf0ewhPXkA0nVXsDYZaydN8rJU1GaMlnjcIYxY= -----END TACK\n    BREAK SIG----- "\n\n    All such PEM blocks will be found, decoded, and return in an ordered list\n    of bytearrays, which may have zero elements if not PEM blocks are found.\n     '
    bList = []
    prefix = '-----BEGIN %s-----' % name
    postfix = '-----END %s-----' % name
    while 1:
        start = s.find(prefix)
        if start == -1:
            return bList
        end = s.find(postfix, start + len(prefix))
        if end == -1:
            raise SyntaxError('Missing PEM postfix')
        s2 = s[start + len(prefix):end]
        retBytes = a2b_base64(s2)
        bList.append(retBytes)
        s = s[end + len(postfix):]

def pem(b, name):
    if False:
        for i in range(10):
            print('nop')
    'Encode a payload bytearray into a PEM string.\n\n    The input will be base64 encoded, then wrapped in a PEM prefix/postfix\n    based on the name string, e.g. for name="CERTIFICATE":\n\n    -----BEGIN CERTIFICATE-----\n    MIIBXDCCAUSgAwIBAgIBADANBgkqhkiG9w0BAQUFADAPMQ0wCwYDVQQDEwRUQUNL\n    ...\n    KoZIhvcNAQEFBQADAwA5kw==\n    -----END CERTIFICATE-----\n    '
    s1 = b2a_base64(b)[:-1]
    s2 = b''
    while s1:
        s2 += s1[:64] + b'\n'
        s1 = s1[64:]
    s = ('-----BEGIN %s-----\n' % name).encode('ascii') + s2 + ('-----END %s-----\n' % name).encode('ascii')
    return s

def pemSniff(inStr, name):
    if False:
        return 10
    searchStr = '-----BEGIN %s-----' % name
    return searchStr in inStr

def parse_private_key(s):
    if False:
        i = 10
        return i + 15
    'Parse a string containing a PEM-encoded <privateKey>.'
    if pemSniff(s, 'PRIVATE KEY'):
        bytes = dePem(s, 'PRIVATE KEY')
        return _parsePKCS8(bytes)
    elif pemSniff(s, 'RSA PRIVATE KEY'):
        bytes = dePem(s, 'RSA PRIVATE KEY')
        return _parseSSLeay(bytes)
    else:
        raise SyntaxError('Not a PEM private key file')

def _parsePKCS8(_bytes):
    if False:
        print('Hello World!')
    s = ASN1_Node(_bytes)
    root = s.root()
    version_node = s.first_child(root)
    version = bytestr_to_int(s.get_value_of_type(version_node, 'INTEGER'))
    if version != 0:
        raise SyntaxError('Unrecognized PKCS8 version')
    rsaOID_node = s.next_node(version_node)
    ii = s.first_child(rsaOID_node)
    rsaOID = decode_OID(s.get_value_of_type(ii, 'OBJECT IDENTIFIER'))
    if rsaOID != '1.2.840.113549.1.1.1':
        raise SyntaxError('Unrecognized AlgorithmIdentifier')
    privkey_node = s.next_node(rsaOID_node)
    value = s.get_value_of_type(privkey_node, 'OCTET STRING')
    return _parseASN1PrivateKey(value)

def _parseSSLeay(bytes):
    if False:
        return 10
    return _parseASN1PrivateKey(ASN1_Node(bytes))

def bytesToNumber(s):
    if False:
        while True:
            i = 10
    return int(binascii.hexlify(s), 16)

def _parseASN1PrivateKey(s):
    if False:
        i = 10
        return i + 15
    s = ASN1_Node(s)
    root = s.root()
    version_node = s.first_child(root)
    version = bytestr_to_int(s.get_value_of_type(version_node, 'INTEGER'))
    if version != 0:
        raise SyntaxError('Unrecognized RSAPrivateKey version')
    n = s.next_node(version_node)
    e = s.next_node(n)
    d = s.next_node(e)
    p = s.next_node(d)
    q = s.next_node(p)
    dP = s.next_node(q)
    dQ = s.next_node(dP)
    qInv = s.next_node(dQ)
    return list(map(lambda x: bytesToNumber(s.get_value_of_type(x, 'INTEGER')), [n, e, d, p, q, dP, dQ, qInv]))