import base64
import hashlib

def sign(data, privatekey):
    if False:
        return 10
    import rsa
    from rsa import pkcs1
    if 'BEGIN RSA PRIVATE KEY' not in privatekey:
        privatekey = '-----BEGIN RSA PRIVATE KEY-----\n%s\n-----END RSA PRIVATE KEY-----' % privatekey
    priv = rsa.PrivateKey.load_pkcs1(privatekey)
    sign = rsa.pkcs1.sign(data, priv, 'SHA-256')
    return sign

def verify(data, publickey, sign):
    if False:
        while True:
            i = 10
    import rsa
    from rsa import pkcs1
    pub = rsa.PublicKey.load_pkcs1(publickey, format='DER')
    try:
        valid = rsa.pkcs1.verify(data, sign, pub)
    except pkcs1.VerificationError:
        valid = False
    return valid

def privatekeyToPublickey(privatekey):
    if False:
        i = 10
        return i + 15
    import rsa
    from rsa import pkcs1
    if 'BEGIN RSA PRIVATE KEY' not in privatekey:
        privatekey = '-----BEGIN RSA PRIVATE KEY-----\n%s\n-----END RSA PRIVATE KEY-----' % privatekey
    priv = rsa.PrivateKey.load_pkcs1(privatekey)
    pub = rsa.PublicKey(priv.n, priv.e)
    return pub.save_pkcs1('DER')

def publickeyToOnion(publickey):
    if False:
        i = 10
        return i + 15
    return base64.b32encode(hashlib.sha1(publickey).digest()[:10]).lower().decode('ascii')