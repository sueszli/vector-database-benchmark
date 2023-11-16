import hashlib
import struct
import hmac
import base58
try:
    hashlib.new('ripemd160')
except ValueError:
    from . import _ripemd

    def ripemd160(*args):
        if False:
            i = 10
            return i + 15
        return _ripemd.new(*args)
else:

    def ripemd160(*args):
        if False:
            print('Hello World!')
        return hashlib.new('ripemd160', *args)

class ECC:
    CURVES = {'secp112r1': (704, 4451685225093714772084598273548427, 4451685225093714776491891542548933, 4451685225093714772084598273548424, 2061118396808653202902996166388514, (188281465057972534892223778713752, 3419875491033170827167861896082688)), 'secp112r2': (705, 4451685225093714772084598273548427, 1112921306273428674967732714786891, 1970543761890640310119143205433388, 1660538572255285715897238774208265, (1534098225527667214992304222930499, 3525120595527770847583704454622871)), 'secp128r1': (706, 340282366762482138434845932244680310783, 340282366762482138443322565580356624661, 340282366762482138434845932244680310780, 308990863222245658030922601041482374867, (29408993404948928992877151431649155974, 275621562871047521857442314737465260675)), 'secp128r2': (707, 340282366762482138434845932244680310783, 85070591690620534603955721926813660579, 284470887156368047300405921324061011681, 126188322377389722996253562430093625949, (164048790688614013222215505581242564928, 52787839253935625605232456597451787076)), 'secp160k1': (708, 1461501637330902918203684832716283019651637554291, 1461501637330902918203686915170869725397159163571, 0, 7, (338530205676502674729549372677647997389429898939, 842365456698940303598009444920994870805149798382)), 'secp160r1': (709, 1461501637330902918203684832716283019653785059327, 1461501637330902918203687197606826779884643492439, 1461501637330902918203684832716283019653785059324, 163235791306168110546604919403271579530548345413, (425826231723888350446541592701409065913635568770, 203520114162904107873991457957346892027982641970)), 'secp160r2': (710, 1461501637330902918203684832716283019651637554291, 1461501637330902918203685083571792140653176136043, 1461501637330902918203684832716283019651637554288, 1032640608390511495214075079957864673410201913530, (473058756663038503608844550604547710019657059949, 1454008495369951658060798698479395908327453245230)), 'secp192k1': (711, 6277101735386680763835789423207666416102355444459739541047, 6277101735386680763835789423061264271957123915200845512077, 0, 3, (5377521262291226325198505011805525673063229037935769709693, 3805108391982600717572440947423858335415441070543209377693)), 'prime192v1': (409, 6277101735386680763835789423207666416083908700390324961279, 6277101735386680763835789423176059013767194773182842284081, 6277101735386680763835789423207666416083908700390324961276, 2455155546008943817740293915197451784769108058161191238065, (602046282375688656758213480587526111916698976636884684818, 174050332293622031404857552280219410364023488927386650641)), 'secp224k1': (712, 26959946667150639794667015087019630673637144422540572481099315275117, 26959946667150639794667015087019640346510327083120074548994958668279, 0, 5, (16983810465656793445178183341822322175883642221536626637512293983324, 13272896753306862154536785447615077600479862871316829862783613755813)), 'secp224r1': (713, 26959946667150639794667015087019630673557916260026308143510066298881, 26959946667150639794667015087019625940457807714424391721682722368061, 26959946667150639794667015087019630673557916260026308143510066298878, 18958286285566608000408668544493926415504680968679321075787234672564, (19277929113566293071110308034699488026831934219452440156649784352033, 19926808758034470970197974370888749184205991990603949537637343198772)), 'secp256k1': (714, 115792089237316195423570985008687907853269984665640564039457584007908834671663, 115792089237316195423570985008687907852837564279074904382605163141518161494337, 0, 7, (55066263022277343669578718895168534326250603453777594175500187360389116729240, 32670510020758816978083085130507043184471273380659243275938904335757337482424)), 'prime256v1': (715, 115792089210356248762697446949407573530086143415290314195533631308867097853951, 115792089210356248762697446949407573529996955224135760342422259061068512044369, 115792089210356248762697446949407573530086143415290314195533631308867097853948, 41058363725152142129326129780047268409114441015993725554835256314039467401291, (48439561293906451759052585252797914202762949526041747995844080717082404635286, 36134250956749795798585127919587881956611106672985015071877198253568414405109)), 'secp384r1': (716, 39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319, 39402006196394479212279040100143613805079739270465446667946905279627659399113263569398956308152294913554433653942643, 39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112316, 27580193559959705877849011840389048093056905856361568521428707301988689241309860865136260764883745107765439761230575, (26247035095799689268623156744566981891852923491109213387815615900925518854738050089022388053975719786650872476732087, 8325710961489029985546751289520108179287853048861315594709205902480503199884419224438643760392947333078086511627871)), 'secp521r1': (717, 6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151, 6864797660130609714981900799081393217269435300143305409394463459185543183397655394245057746333217197532963996371363321113864768612440380340372808892707005449, 6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057148, 1093849038073734274511112390766805569936207598951683748994586394495953116150735016013708737573759623248592132296706313309438452531591012912142327488478985984, (2661740802050217063228768716723360960729859168756973147706671368418802944996427808491545080627771902352094241225065558662157113545570916814161637315895999846, 3757180025770020463545507224491183603594455134769762486694567779615544477440556316691234405012945539562144444537289428522585666729196580810124344277578376784))}

    def __init__(self, backend, aes):
        if False:
            for i in range(10):
                print('nop')
        self._backend = backend
        self._aes = aes

    def get_curve(self, name):
        if False:
            print('Hello World!')
        if name not in self.CURVES:
            raise ValueError('Unknown curve {}'.format(name))
        (nid, p, n, a, b, g) = self.CURVES[name]
        return EllipticCurve(self._backend(p, n, a, b, g), self._aes, nid)

    def get_backend(self):
        if False:
            while True:
                i = 10
        return self._backend.get_backend()

class EllipticCurve:

    def __init__(self, backend, aes, nid):
        if False:
            return 10
        self._backend = backend
        self._aes = aes
        self.nid = nid

    def _encode_public_key(self, x, y, is_compressed=True, raw=True):
        if False:
            return 10
        if raw:
            if is_compressed:
                return bytes([2 + y[-1] % 2]) + x
            else:
                return bytes([4]) + x + y
        else:
            return struct.pack('!HH', self.nid, len(x)) + x + struct.pack('!H', len(y)) + y

    def _decode_public_key(self, public_key, partial=False):
        if False:
            return 10
        if not public_key:
            raise ValueError('No public key')
        if public_key[0] == 4:
            expected_length = 1 + 2 * self._backend.public_key_length
            if partial:
                if len(public_key) < expected_length:
                    raise ValueError('Invalid uncompressed public key length')
            elif len(public_key) != expected_length:
                raise ValueError('Invalid uncompressed public key length')
            x = public_key[1:1 + self._backend.public_key_length]
            y = public_key[1 + self._backend.public_key_length:expected_length]
            if partial:
                return ((x, y), expected_length)
            else:
                return (x, y)
        elif public_key[0] in (2, 3):
            expected_length = 1 + self._backend.public_key_length
            if partial:
                if len(public_key) < expected_length:
                    raise ValueError('Invalid compressed public key length')
            elif len(public_key) != expected_length:
                raise ValueError('Invalid compressed public key length')
            (x, y) = self._backend.decompress_point(public_key[:expected_length])
            if x != public_key[1:expected_length]:
                raise ValueError('Incorrect compressed public key')
            if partial:
                return ((x, y), expected_length)
            else:
                return (x, y)
        else:
            raise ValueError('Invalid public key prefix')

    def _decode_public_key_openssl(self, public_key, partial=False):
        if False:
            i = 10
            return i + 15
        if not public_key:
            raise ValueError('No public key')
        i = 0
        (nid,) = struct.unpack('!H', public_key[i:i + 2])
        i += 2
        if nid != self.nid:
            raise ValueError('Wrong curve')
        (xlen,) = struct.unpack('!H', public_key[i:i + 2])
        i += 2
        if len(public_key) - i < xlen:
            raise ValueError('Too short public key')
        x = public_key[i:i + xlen]
        i += xlen
        (ylen,) = struct.unpack('!H', public_key[i:i + 2])
        i += 2
        if len(public_key) - i < ylen:
            raise ValueError('Too short public key')
        y = public_key[i:i + ylen]
        i += ylen
        if partial:
            return ((x, y), i)
        else:
            if i < len(public_key):
                raise ValueError('Too long public key')
            return (x, y)

    def new_private_key(self, is_compressed=False):
        if False:
            print('Hello World!')
        return self._backend.new_private_key() + (b'\x01' if is_compressed else b'')

    def private_to_public(self, private_key):
        if False:
            for i in range(10):
                print('nop')
        if len(private_key) == self._backend.public_key_length:
            is_compressed = False
        elif len(private_key) == self._backend.public_key_length + 1 and private_key[-1] == 1:
            is_compressed = True
            private_key = private_key[:-1]
        else:
            raise ValueError('Private key has invalid length')
        (x, y) = self._backend.private_to_public(private_key)
        return self._encode_public_key(x, y, is_compressed=is_compressed)

    def private_to_wif(self, private_key):
        if False:
            while True:
                i = 10
        return base58.b58encode_check(b'\x80' + private_key)

    def wif_to_private(self, wif):
        if False:
            return 10
        dec = base58.b58decode_check(wif)
        if dec[0] != 128:
            raise ValueError('Invalid network (expected mainnet)')
        return dec[1:]

    def public_to_address(self, public_key):
        if False:
            for i in range(10):
                print('nop')
        h = hashlib.sha256(public_key).digest()
        hash160 = ripemd160(h).digest()
        return base58.b58encode_check(b'\x00' + hash160)

    def private_to_address(self, private_key):
        if False:
            return 10
        return self.public_to_address(self.private_to_public(private_key))

    def derive(self, private_key, public_key):
        if False:
            for i in range(10):
                print('nop')
        if len(private_key) == self._backend.public_key_length + 1 and private_key[-1] == 1:
            private_key = private_key[:-1]
        if len(private_key) != self._backend.public_key_length:
            raise ValueError('Private key has invalid length')
        if not isinstance(public_key, tuple):
            public_key = self._decode_public_key(public_key)
        return self._backend.ecdh(private_key, public_key)

    def _digest(self, data, hash):
        if False:
            print('Hello World!')
        if hash is None:
            return data
        elif callable(hash):
            return hash(data)
        elif hash == 'sha1':
            return hashlib.sha1(data).digest()
        elif hash == 'sha256':
            return hashlib.sha256(data).digest()
        elif hash == 'sha512':
            return hashlib.sha512(data).digest()
        else:
            raise ValueError('Unknown hash/derivation method')

    def encrypt(self, data, public_key, algo='aes-256-cbc', derivation='sha256', mac='hmac-sha256', return_aes_key=False):
        if False:
            print('Hello World!')
        private_key = self.new_private_key()
        ecdh = self.derive(private_key, public_key)
        key = self._digest(ecdh, derivation)
        k_enc_len = self._aes.get_algo_key_length(algo)
        if len(key) < k_enc_len:
            raise ValueError('Too short digest')
        (k_enc, k_mac) = (key[:k_enc_len], key[k_enc_len:])
        (ciphertext, iv) = self._aes.encrypt(data, k_enc, algo=algo)
        ephem_public_key = self.private_to_public(private_key)
        ephem_public_key = self._decode_public_key(ephem_public_key)
        ephem_public_key = self._encode_public_key(*ephem_public_key, raw=False)
        ciphertext = iv + ephem_public_key + ciphertext
        if callable(mac):
            tag = mac(k_mac, ciphertext)
        elif mac == 'hmac-sha256':
            h = hmac.new(k_mac, digestmod='sha256')
            h.update(ciphertext)
            tag = h.digest()
        elif mac == 'hmac-sha512':
            h = hmac.new(k_mac, digestmod='sha512')
            h.update(ciphertext)
            tag = h.digest()
        elif mac is None:
            tag = b''
        else:
            raise ValueError('Unsupported MAC')
        if return_aes_key:
            return (ciphertext + tag, k_enc)
        else:
            return ciphertext + tag

    def decrypt(self, ciphertext, private_key, algo='aes-256-cbc', derivation='sha256', mac='hmac-sha256'):
        if False:
            i = 10
            return i + 15
        if callable(mac):
            tag_length = mac.digest_size
        elif mac == 'hmac-sha256':
            tag_length = hmac.new(b'', digestmod='sha256').digest_size
        elif mac == 'hmac-sha512':
            tag_length = hmac.new(b'', digestmod='sha512').digest_size
        elif mac is None:
            tag_length = 0
        else:
            raise ValueError('Unsupported MAC')
        if len(ciphertext) < tag_length:
            raise ValueError('Ciphertext is too small to contain MAC tag')
        if tag_length == 0:
            tag = b''
        else:
            (ciphertext, tag) = (ciphertext[:-tag_length], ciphertext[-tag_length:])
        orig_ciphertext = ciphertext
        if len(ciphertext) < 16:
            raise ValueError('Ciphertext is too small to contain IV')
        (iv, ciphertext) = (ciphertext[:16], ciphertext[16:])
        (public_key, pos) = self._decode_public_key_openssl(ciphertext, partial=True)
        ciphertext = ciphertext[pos:]
        ecdh = self.derive(private_key, public_key)
        key = self._digest(ecdh, derivation)
        k_enc_len = self._aes.get_algo_key_length(algo)
        if len(key) < k_enc_len:
            raise ValueError('Too short digest')
        (k_enc, k_mac) = (key[:k_enc_len], key[k_enc_len:])
        if callable(mac):
            expected_tag = mac(k_mac, orig_ciphertext)
        elif mac == 'hmac-sha256':
            h = hmac.new(k_mac, digestmod='sha256')
            h.update(orig_ciphertext)
            expected_tag = h.digest()
        elif mac == 'hmac-sha512':
            h = hmac.new(k_mac, digestmod='sha512')
            h.update(orig_ciphertext)
            expected_tag = h.digest()
        elif mac is None:
            expected_tag = b''
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError('Invalid MAC tag')
        return self._aes.decrypt(ciphertext, iv, k_enc, algo=algo)

    def sign(self, data, private_key, hash='sha256', recoverable=False, entropy=None):
        if False:
            while True:
                i = 10
        if len(private_key) == self._backend.public_key_length:
            is_compressed = False
        elif len(private_key) == self._backend.public_key_length + 1 and private_key[-1] == 1:
            is_compressed = True
            private_key = private_key[:-1]
        else:
            raise ValueError('Private key has invalid length')
        data = self._digest(data, hash)
        if not entropy:
            v = b'\x01' * len(data)
            k = b'\x00' * len(data)
            k = hmac.new(k, v + b'\x00' + private_key + data, 'sha256').digest()
            v = hmac.new(k, v, 'sha256').digest()
            k = hmac.new(k, v + b'\x01' + private_key + data, 'sha256').digest()
            v = hmac.new(k, v, 'sha256').digest()
            entropy = hmac.new(k, v, 'sha256').digest()
        return self._backend.sign(data, private_key, recoverable, is_compressed, entropy=entropy)

    def recover(self, signature, data, hash='sha256'):
        if False:
            return 10
        if len(signature) != 1 + 2 * self._backend.public_key_length:
            raise ValueError('Cannot recover an unrecoverable signature')
        (x, y) = self._backend.recover(signature, self._digest(data, hash))
        is_compressed = signature[0] >= 31
        return self._encode_public_key(x, y, is_compressed=is_compressed)

    def verify(self, signature, data, public_key, hash='sha256'):
        if False:
            print('Hello World!')
        if len(signature) == 1 + 2 * self._backend.public_key_length:
            signature = signature[1:]
        if len(signature) != 2 * self._backend.public_key_length:
            raise ValueError('Invalid signature format')
        if not isinstance(public_key, tuple):
            public_key = self._decode_public_key(public_key)
        return self._backend.verify(signature, self._digest(data, hash), public_key)

    def derive_child(self, seed, child):
        if False:
            print('Hello World!')
        if not 0 <= child < 2 ** 31:
            raise ValueError('Invalid child index')
        return self._backend.derive_child(seed, child)