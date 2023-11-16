"""
TLS cipher suites.

A comprehensive list of specified cipher suites can be consulted at:
https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml
"""
from scapy.layers.tls.crypto.kx_algs import _tls_kx_algs
from scapy.layers.tls.crypto.hash import _tls_hash_algs
from scapy.layers.tls.crypto.h_mac import _tls_hmac_algs
from scapy.layers.tls.crypto.ciphers import _tls_cipher_algs

def get_algs_from_ciphersuite_name(ciphersuite_name):
    if False:
        return 10
    '\n    Return the 3-tuple made of the Key Exchange Algorithm class, the Cipher\n    class and the HMAC class, through the parsing of the ciphersuite name.\n    '
    tls1_3 = False
    if ciphersuite_name.startswith('TLS'):
        s = ciphersuite_name[4:]
        if s.endswith('CCM') or s.endswith('CCM_8'):
            (kx_name, s) = s.split('_WITH_')
            kx_alg = _tls_kx_algs.get(kx_name)
            hash_alg = _tls_hash_algs.get('SHA256')
            cipher_alg = _tls_cipher_algs.get(s)
            hmac_alg = None
        else:
            if 'WITH' in s:
                (kx_name, s) = s.split('_WITH_')
                kx_alg = _tls_kx_algs.get(kx_name)
            else:
                tls1_3 = True
                kx_alg = _tls_kx_algs.get('TLS13')
            hash_name = s.split('_')[-1]
            hash_alg = _tls_hash_algs.get(hash_name)
            cipher_name = s[:-(len(hash_name) + 1)]
            if tls1_3:
                cipher_name += '_TLS13'
            cipher_alg = _tls_cipher_algs.get(cipher_name)
            hmac_alg = None
            if cipher_alg is not None and cipher_alg.type != 'aead':
                hmac_name = 'HMAC-%s' % hash_name
                hmac_alg = _tls_hmac_algs.get(hmac_name)
    elif ciphersuite_name.startswith('SSL'):
        s = ciphersuite_name[7:]
        kx_alg = _tls_kx_algs.get('SSLv2')
        (cipher_name, hash_name) = s.split('_WITH_')
        cipher_alg = _tls_cipher_algs.get(cipher_name.rstrip('_EXPORT40'))
        kx_alg.export = cipher_name.endswith('_EXPORT40')
        hmac_alg = _tls_hmac_algs.get('HMAC-NULL')
        hash_alg = _tls_hash_algs.get(hash_name)
    return (kx_alg, cipher_alg, hmac_alg, hash_alg, tls1_3)
_tls_cipher_suites = {}
_tls_cipher_suites_cls = {}

class _GenericCipherSuiteMetaclass(type):
    """
    Cipher suite classes are automatically registered through this metaclass.
    Their name attribute equates their respective class name.

    We also pre-compute every expected length of the key block to be generated,
    which may vary according to the current tls_version. The default is set to
    the TLS 1.2 length, and the value should be set at class instantiation.

    Regarding the AEAD cipher suites, note that the 'hmac_alg' attribute will
    be set to None. Yet, we always need a 'hash_alg' for the PRF.
    """

    def __new__(cls, cs_name, bases, dct):
        if False:
            print('Hello World!')
        cs_val = dct.get('val')
        if cs_name != '_GenericCipherSuite':
            (kx, c, hm, h, tls1_3) = get_algs_from_ciphersuite_name(cs_name)
            if c is None or h is None or (kx is None and (not tls1_3)):
                dct['usable'] = False
            else:
                dct['usable'] = True
                dct['name'] = cs_name
                dct['kx_alg'] = kx
                dct['cipher_alg'] = c
                dct['hmac_alg'] = hm
                dct['hash_alg'] = h
                if not tls1_3:
                    kb_len = 2 * c.key_len
                    if c.type == 'stream' or c.type == 'block':
                        kb_len += 2 * hm.key_len
                    kb_len_v1_0 = kb_len
                    if c.type == 'block':
                        kb_len_v1_0 += 2 * c.block_size
                    elif c.type == 'aead':
                        kb_len_v1_0 += 2 * c.fixed_iv_len
                        kb_len += 2 * c.fixed_iv_len
                    dct['_key_block_len_v1_0'] = kb_len_v1_0
                    dct['key_block_len'] = kb_len
            _tls_cipher_suites[cs_val] = cs_name
        the_class = super(_GenericCipherSuiteMetaclass, cls).__new__(cls, cs_name, bases, dct)
        if cs_name != '_GenericCipherSuite':
            _tls_cipher_suites_cls[cs_val] = the_class
        return the_class

class _GenericCipherSuite(metaclass=_GenericCipherSuiteMetaclass):

    def __init__(self, tls_version=771):
        if False:
            for i in range(10):
                print('nop')
        '\n        Most of the attributes are fixed and have already been set by the\n        metaclass, but we still have to provide tls_version differentiation.\n\n        For now, the key_block_len remains the only application if this.\n        Indeed for TLS 1.1+, when using a block cipher, there are no implicit\n        IVs derived from the master secret. Note that an overlong key_block_len\n        would not affect the secret generation (the trailing bytes would\n        simply be discarded), but we still provide this for completeness.\n        '
        super(_GenericCipherSuite, self).__init__()
        if tls_version <= 769:
            self.key_block_len = self._key_block_len_v1_0

class TLS_NULL_WITH_NULL_NULL(_GenericCipherSuite):
    val = 0

class TLS_RSA_WITH_NULL_MD5(_GenericCipherSuite):
    val = 1

class TLS_RSA_WITH_NULL_SHA(_GenericCipherSuite):
    val = 2

class TLS_RSA_EXPORT_WITH_RC4_40_MD5(_GenericCipherSuite):
    val = 3

class TLS_RSA_WITH_RC4_128_MD5(_GenericCipherSuite):
    val = 4

class TLS_RSA_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 5

class TLS_RSA_EXPORT_WITH_RC2_CBC_40_MD5(_GenericCipherSuite):
    val = 6

class TLS_RSA_WITH_IDEA_CBC_SHA(_GenericCipherSuite):
    val = 7

class TLS_RSA_EXPORT_WITH_DES40_CBC_SHA(_GenericCipherSuite):
    val = 8

class TLS_RSA_WITH_DES_CBC_SHA(_GenericCipherSuite):
    val = 9

class TLS_RSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 10

class TLS_DH_DSS_EXPORT_WITH_DES40_CBC_SHA(_GenericCipherSuite):
    val = 11

class TLS_DH_DSS_WITH_DES_CBC_SHA(_GenericCipherSuite):
    val = 12

class TLS_DH_DSS_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 13

class TLS_DH_RSA_EXPORT_WITH_DES40_CBC_SHA(_GenericCipherSuite):
    val = 14

class TLS_DH_RSA_WITH_DES_CBC_SHA(_GenericCipherSuite):
    val = 15

class TLS_DH_RSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 16

class TLS_DHE_DSS_EXPORT_WITH_DES40_CBC_SHA(_GenericCipherSuite):
    val = 17

class TLS_DHE_DSS_WITH_DES_CBC_SHA(_GenericCipherSuite):
    val = 18

class TLS_DHE_DSS_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 19

class TLS_DHE_RSA_EXPORT_WITH_DES40_CBC_SHA(_GenericCipherSuite):
    val = 20

class TLS_DHE_RSA_WITH_DES_CBC_SHA(_GenericCipherSuite):
    val = 21

class TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 22

class TLS_DH_anon_EXPORT_WITH_RC4_40_MD5(_GenericCipherSuite):
    val = 23

class TLS_DH_anon_WITH_RC4_128_MD5(_GenericCipherSuite):
    val = 24

class TLS_DH_anon_EXPORT_WITH_DES40_CBC_SHA(_GenericCipherSuite):
    val = 25

class TLS_DH_anon_WITH_DES_CBC_SHA(_GenericCipherSuite):
    val = 26

class TLS_DH_anon_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 27

class TLS_KRB5_WITH_DES_CBC_SHA(_GenericCipherSuite):
    val = 30

class TLS_KRB5_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 31

class TLS_KRB5_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 32

class TLS_KRB5_WITH_IDEA_CBC_SHA(_GenericCipherSuite):
    val = 33

class TLS_KRB5_WITH_DES_CBC_MD5(_GenericCipherSuite):
    val = 34

class TLS_KRB5_WITH_3DES_EDE_CBC_MD5(_GenericCipherSuite):
    val = 35

class TLS_KRB5_WITH_RC4_128_MD5(_GenericCipherSuite):
    val = 36

class TLS_KRB5_WITH_IDEA_CBC_MD5(_GenericCipherSuite):
    val = 37

class TLS_KRB5_EXPORT_WITH_DES40_CBC_SHA(_GenericCipherSuite):
    val = 38

class TLS_KRB5_EXPORT_WITH_RC2_CBC_40_SHA(_GenericCipherSuite):
    val = 39

class TLS_KRB5_EXPORT_WITH_RC4_40_SHA(_GenericCipherSuite):
    val = 40

class TLS_KRB5_EXPORT_WITH_DES40_CBC_MD5(_GenericCipherSuite):
    val = 41

class TLS_KRB5_EXPORT_WITH_RC2_CBC_40_MD5(_GenericCipherSuite):
    val = 42

class TLS_KRB5_EXPORT_WITH_RC4_40_MD5(_GenericCipherSuite):
    val = 43

class TLS_PSK_WITH_NULL_SHA(_GenericCipherSuite):
    val = 44

class TLS_DHE_PSK_WITH_NULL_SHA(_GenericCipherSuite):
    val = 45

class TLS_RSA_PSK_WITH_NULL_SHA(_GenericCipherSuite):
    val = 46

class TLS_RSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 47

class TLS_DH_DSS_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 48

class TLS_DH_RSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49

class TLS_DHE_DSS_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 50

class TLS_DHE_RSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 51

class TLS_DH_anon_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 52

class TLS_RSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 53

class TLS_DH_DSS_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 54

class TLS_DH_RSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 55

class TLS_DHE_DSS_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 56

class TLS_DHE_RSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 57

class TLS_DH_anon_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 58

class TLS_RSA_WITH_NULL_SHA256(_GenericCipherSuite):
    val = 59

class TLS_RSA_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 60

class TLS_RSA_WITH_AES_256_CBC_SHA256(_GenericCipherSuite):
    val = 61

class TLS_DH_DSS_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 62

class TLS_DH_RSA_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 63

class TLS_DHE_DSS_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 64

class TLS_RSA_WITH_CAMELLIA_128_CBC_SHA(_GenericCipherSuite):
    val = 65

class TLS_DH_DSS_WITH_CAMELLIA_128_CBC_SHA(_GenericCipherSuite):
    val = 66

class TLS_DH_RSA_WITH_CAMELLIA_128_CBC_SHA(_GenericCipherSuite):
    val = 67

class TLS_DHE_DSS_WITH_CAMELLIA_128_CBC_SHA(_GenericCipherSuite):
    val = 68

class TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA(_GenericCipherSuite):
    val = 69

class TLS_DH_anon_WITH_CAMELLIA_128_CBC_SHA(_GenericCipherSuite):
    val = 70

class TLS_DHE_RSA_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 103

class TLS_DH_DSS_WITH_AES_256_CBC_SHA256(_GenericCipherSuite):
    val = 104

class TLS_DH_RSA_WITH_AES_256_CBC_SHA256(_GenericCipherSuite):
    val = 105

class TLS_DHE_DSS_WITH_AES_256_CBC_SHA256(_GenericCipherSuite):
    val = 106

class TLS_DHE_RSA_WITH_AES_256_CBC_SHA256(_GenericCipherSuite):
    val = 107

class TLS_DH_anon_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 108

class TLS_DH_anon_WITH_AES_256_CBC_SHA256(_GenericCipherSuite):
    val = 109

class TLS_RSA_WITH_CAMELLIA_256_CBC_SHA(_GenericCipherSuite):
    val = 132

class TLS_DH_DSS_WITH_CAMELLIA_256_CBC_SHA(_GenericCipherSuite):
    val = 133

class TLS_DH_RSA_WITH_CAMELLIA_256_CBC_SHA(_GenericCipherSuite):
    val = 134

class TLS_DHE_DSS_WITH_CAMELLIA_256_CBC_SHA(_GenericCipherSuite):
    val = 135

class TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA(_GenericCipherSuite):
    val = 136

class TLS_DH_anon_WITH_CAMELLIA_256_CBC_SHA(_GenericCipherSuite):
    val = 137

class TLS_PSK_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 138

class TLS_PSK_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 139

class TLS_PSK_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 140

class TLS_PSK_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 141

class TLS_DHE_PSK_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 142

class TLS_DHE_PSK_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 143

class TLS_DHE_PSK_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 144

class TLS_DHE_PSK_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 145

class TLS_RSA_PSK_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 146

class TLS_RSA_PSK_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 147

class TLS_RSA_PSK_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 148

class TLS_RSA_PSK_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 149

class TLS_RSA_WITH_SEED_CBC_SHA(_GenericCipherSuite):
    val = 150

class TLS_DH_DSS_WITH_SEED_CBC_SHA(_GenericCipherSuite):
    val = 151

class TLS_DH_RSA_WITH_SEED_CBC_SHA(_GenericCipherSuite):
    val = 152

class TLS_DHE_DSS_WITH_SEED_CBC_SHA(_GenericCipherSuite):
    val = 153

class TLS_DHE_RSA_WITH_SEED_CBC_SHA(_GenericCipherSuite):
    val = 154

class TLS_DH_anon_WITH_SEED_CBC_SHA(_GenericCipherSuite):
    val = 155

class TLS_RSA_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 156

class TLS_RSA_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 157

class TLS_DHE_RSA_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 158

class TLS_DHE_RSA_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 159

class TLS_DH_RSA_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 160

class TLS_DH_RSA_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 161

class TLS_DHE_DSS_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 162

class TLS_DHE_DSS_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 163

class TLS_DH_DSS_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 164

class TLS_DH_DSS_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 165

class TLS_DH_anon_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 166

class TLS_DH_anon_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 167

class TLS_PSK_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 168

class TLS_PSK_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 169

class TLS_DHE_PSK_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 170

class TLS_DHE_PSK_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 171

class TLS_RSA_PSK_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 172

class TLS_RSA_PSK_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 173

class TLS_PSK_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 174

class TLS_PSK_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 175

class TLS_PSK_WITH_NULL_SHA256(_GenericCipherSuite):
    val = 176

class TLS_PSK_WITH_NULL_SHA384(_GenericCipherSuite):
    val = 177

class TLS_DHE_PSK_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 178

class TLS_DHE_PSK_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 179

class TLS_DHE_PSK_WITH_NULL_SHA256(_GenericCipherSuite):
    val = 180

class TLS_DHE_PSK_WITH_NULL_SHA384(_GenericCipherSuite):
    val = 181

class TLS_RSA_PSK_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 182

class TLS_RSA_PSK_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 183

class TLS_RSA_PSK_WITH_NULL_SHA256(_GenericCipherSuite):
    val = 184

class TLS_RSA_PSK_WITH_NULL_SHA384(_GenericCipherSuite):
    val = 185

class TLS_RSA_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 186

class TLS_DH_DSS_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 187

class TLS_DH_RSA_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 188

class TLS_DHE_DSS_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 189

class TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 190

class TLS_DH_anon_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 191

class TLS_RSA_WITH_CAMELLIA_256_CBC_SHA256(_GenericCipherSuite):
    val = 192

class TLS_DH_DSS_WITH_CAMELLIA_256_CBC_SHA256(_GenericCipherSuite):
    val = 193

class TLS_DH_RSA_WITH_CAMELLIA_256_CBC_SHA256(_GenericCipherSuite):
    val = 194

class TLS_DHE_DSS_WITH_CAMELLIA_256_CBC_SHA256(_GenericCipherSuite):
    val = 195

class TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA256(_GenericCipherSuite):
    val = 196

class TLS_DH_anon_WITH_CAMELLIA_256_CBC_SHA256(_GenericCipherSuite):
    val = 197

class TLS_ECDH_ECDSA_WITH_NULL_SHA(_GenericCipherSuite):
    val = 49153

class TLS_ECDH_ECDSA_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 49154

class TLS_ECDH_ECDSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49155

class TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49156

class TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49157

class TLS_ECDHE_ECDSA_WITH_NULL_SHA(_GenericCipherSuite):
    val = 49158

class TLS_ECDHE_ECDSA_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 49159

class TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49160

class TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49161

class TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49162

class TLS_ECDH_RSA_WITH_NULL_SHA(_GenericCipherSuite):
    val = 49163

class TLS_ECDH_RSA_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 49164

class TLS_ECDH_RSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49165

class TLS_ECDH_RSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49166

class TLS_ECDH_RSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49167

class TLS_ECDHE_RSA_WITH_NULL_SHA(_GenericCipherSuite):
    val = 49168

class TLS_ECDHE_RSA_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 49169

class TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49170

class TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49171

class TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49172

class TLS_ECDH_anon_WITH_NULL_SHA(_GenericCipherSuite):
    val = 49173

class TLS_ECDH_anon_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 49174

class TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49175

class TLS_ECDH_anon_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49176

class TLS_ECDH_anon_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49177

class TLS_SRP_SHA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49178

class TLS_SRP_SHA_RSA_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49179

class TLS_SRP_SHA_DSS_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49180

class TLS_SRP_SHA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49181

class TLS_SRP_SHA_RSA_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49182

class TLS_SRP_SHA_DSS_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49183

class TLS_SRP_SHA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49184

class TLS_SRP_SHA_RSA_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49185

class TLS_SRP_SHA_DSS_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49186

class TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 49187

class TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 49188

class TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 49189

class TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 49190

class TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 49191

class TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 49192

class TLS_ECDH_RSA_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 49193

class TLS_ECDH_RSA_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 49194

class TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 49195

class TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 49196

class TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 49197

class TLS_ECDH_ECDSA_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 49198

class TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 49199

class TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 49200

class TLS_ECDH_RSA_WITH_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 49201

class TLS_ECDH_RSA_WITH_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 49202

class TLS_ECDHE_PSK_WITH_RC4_128_SHA(_GenericCipherSuite):
    val = 49203

class TLS_ECDHE_PSK_WITH_3DES_EDE_CBC_SHA(_GenericCipherSuite):
    val = 49204

class TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA(_GenericCipherSuite):
    val = 49205

class TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA(_GenericCipherSuite):
    val = 49206

class TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA256(_GenericCipherSuite):
    val = 49207

class TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA384(_GenericCipherSuite):
    val = 49208

class TLS_ECDHE_PSK_WITH_NULL_SHA(_GenericCipherSuite):
    val = 49209

class TLS_ECDHE_PSK_WITH_NULL_SHA256(_GenericCipherSuite):
    val = 49210

class TLS_ECDHE_PSK_WITH_NULL_SHA384(_GenericCipherSuite):
    val = 49211

class TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49266

class TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49267

class TLS_ECDH_ECDSA_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49268

class TLS_ECDH_ECDSA_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49269

class TLS_ECDHE_RSA_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49270

class TLS_ECDHE_RSA_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49271

class TLS_ECDH_RSA_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49272

class TLS_ECDH_RSA_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49273

class TLS_RSA_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49274

class TLS_RSA_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49275

class TLS_DHE_RSA_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49276

class TLS_DHE_RSA_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49277

class TLS_DH_RSA_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49278

class TLS_DH_RSA_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49279

class TLS_DHE_DSS_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49280

class TLS_DHE_DSS_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49281

class TLS_DH_DSS_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49282

class TLS_DH_DSS_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49283

class TLS_DH_anon_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49284

class TLS_DH_anon_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49285

class TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49286

class TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49287

class TLS_ECDH_ECDSA_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49288

class TLS_ECDH_ECDSA_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49289

class TLS_ECDHE_RSA_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49290

class TLS_ECDHE_RSA_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49291

class TLS_ECDH_RSA_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49292

class TLS_ECDH_RSA_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49293

class TLS_PSK_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49294

class TLS_PSK_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49295

class TLS_DHE_PSK_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49296

class TLS_DHE_PSK_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49297

class TLS_RSA_PSK_WITH_CAMELLIA_128_GCM_SHA256(_GenericCipherSuite):
    val = 49298

class TLS_RSA_PSK_WITH_CAMELLIA_256_GCM_SHA384(_GenericCipherSuite):
    val = 49299

class TLS_PSK_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49300

class TLS_PSK_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49301

class TLS_DHE_PSK_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49302

class TLS_DHE_PSK_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49303

class TLS_RSA_PSK_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49304

class TLS_RSA_PSK_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49305

class TLS_ECDHE_PSK_WITH_CAMELLIA_128_CBC_SHA256(_GenericCipherSuite):
    val = 49306

class TLS_ECDHE_PSK_WITH_CAMELLIA_256_CBC_SHA384(_GenericCipherSuite):
    val = 49307

class TLS_RSA_WITH_AES_128_CCM(_GenericCipherSuite):
    val = 49308

class TLS_RSA_WITH_AES_256_CCM(_GenericCipherSuite):
    val = 49309

class TLS_DHE_RSA_WITH_AES_128_CCM(_GenericCipherSuite):
    val = 49310

class TLS_DHE_RSA_WITH_AES_256_CCM(_GenericCipherSuite):
    val = 49311

class TLS_RSA_WITH_AES_128_CCM_8(_GenericCipherSuite):
    val = 49312

class TLS_RSA_WITH_AES_256_CCM_8(_GenericCipherSuite):
    val = 49313

class TLS_DHE_RSA_WITH_AES_128_CCM_8(_GenericCipherSuite):
    val = 49314

class TLS_DHE_RSA_WITH_AES_256_CCM_8(_GenericCipherSuite):
    val = 49315

class TLS_PSK_WITH_AES_128_CCM(_GenericCipherSuite):
    val = 49316

class TLS_PSK_WITH_AES_256_CCM(_GenericCipherSuite):
    val = 49317

class TLS_DHE_PSK_WITH_AES_128_CCM(_GenericCipherSuite):
    val = 49318

class TLS_DHE_PSK_WITH_AES_256_CCM(_GenericCipherSuite):
    val = 49319

class TLS_PSK_WITH_AES_128_CCM_8(_GenericCipherSuite):
    val = 49320

class TLS_PSK_WITH_AES_256_CCM_8(_GenericCipherSuite):
    val = 49321

class TLS_DHE_PSK_WITH_AES_128_CCM_8(_GenericCipherSuite):
    val = 49322

class TLS_DHE_PSK_WITH_AES_256_CCM_8(_GenericCipherSuite):
    val = 49323

class TLS_ECDHE_ECDSA_WITH_AES_128_CCM(_GenericCipherSuite):
    val = 49324

class TLS_ECDHE_ECDSA_WITH_AES_256_CCM(_GenericCipherSuite):
    val = 49325

class TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8(_GenericCipherSuite):
    val = 49326

class TLS_ECDHE_ECDSA_WITH_AES_256_CCM_8(_GenericCipherSuite):
    val = 49327

class TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256_OLD(_GenericCipherSuite):
    val = 52243

class TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256_OLD(_GenericCipherSuite):
    val = 52244

class TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256_OLD(_GenericCipherSuite):
    val = 52245

class TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 52392

class TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 52393

class TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 52394

class TLS_PSK_WITH_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 52395

class TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 52396

class TLS_DHE_PSK_WITH_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 52397

class TLS_RSA_PSK_WITH_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 52398

class TLS_AES_128_GCM_SHA256(_GenericCipherSuite):
    val = 4865

class TLS_AES_256_GCM_SHA384(_GenericCipherSuite):
    val = 4866

class TLS_CHACHA20_POLY1305_SHA256(_GenericCipherSuite):
    val = 4867

class TLS_AES_128_CCM_SHA256(_GenericCipherSuite):
    val = 4868

class TLS_AES_128_CCM_8_SHA256(_GenericCipherSuite):
    val = 4869

class SSL_CK_RC4_128_WITH_MD5(_GenericCipherSuite):
    val = 65664

class SSL_CK_RC4_128_EXPORT40_WITH_MD5(_GenericCipherSuite):
    val = 131200

class SSL_CK_RC2_128_CBC_WITH_MD5(_GenericCipherSuite):
    val = 196736

class SSL_CK_RC2_128_CBC_EXPORT40_WITH_MD5(_GenericCipherSuite):
    val = 262272

class SSL_CK_IDEA_128_CBC_WITH_MD5(_GenericCipherSuite):
    val = 327808

class SSL_CK_DES_64_CBC_WITH_MD5(_GenericCipherSuite):
    val = 393280

class SSL_CK_DES_192_EDE3_CBC_WITH_MD5(_GenericCipherSuite):
    val = 458944
_tls_cipher_suites[255] = 'TLS_EMPTY_RENEGOTIATION_INFO_SCSV'
_tls_cipher_suites[22016] = 'TLS_FALLBACK_SCSV'

def get_usable_ciphersuites(li, kx):
    if False:
        while True:
            i = 10
    '\n    From a list of proposed ciphersuites, this function returns a list of\n    usable cipher suites, i.e. for which key exchange, cipher and hash\n    algorithms are known to be implemented and usable in current version of the\n    TLS extension. The order of the cipher suites in the list returned by the\n    function matches the one of the proposal.\n    '
    res = []
    for c in li:
        if c in _tls_cipher_suites_cls:
            cipher = _tls_cipher_suites_cls[c]
            if cipher.usable:
                if cipher.kx_alg.anonymous or kx in cipher.kx_alg.name or cipher.kx_alg.name == 'TLS13':
                    res.append(c)
    return res