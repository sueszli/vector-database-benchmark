"""
@author:       Brendan Dolan-Gavitt
@license:      GNU General Public License 2.0
@contact:      bdolangavitt@wesleyan.edu
"""
import volatility.obj as obj
import volatility.win32.rawreg as rawreg
import volatility.win32.hive as hive
from Crypto.Hash import MD5, MD4
from Crypto.Cipher import ARC4, DES
from struct import unpack, pack
odd_parity = [1, 1, 2, 2, 4, 4, 7, 7, 8, 8, 11, 11, 13, 13, 14, 14, 16, 16, 19, 19, 21, 21, 22, 22, 25, 25, 26, 26, 28, 28, 31, 31, 32, 32, 35, 35, 37, 37, 38, 38, 41, 41, 42, 42, 44, 44, 47, 47, 49, 49, 50, 50, 52, 52, 55, 55, 56, 56, 59, 59, 61, 61, 62, 62, 64, 64, 67, 67, 69, 69, 70, 70, 73, 73, 74, 74, 76, 76, 79, 79, 81, 81, 82, 82, 84, 84, 87, 87, 88, 88, 91, 91, 93, 93, 94, 94, 97, 97, 98, 98, 100, 100, 103, 103, 104, 104, 107, 107, 109, 109, 110, 110, 112, 112, 115, 115, 117, 117, 118, 118, 121, 121, 122, 122, 124, 124, 127, 127, 128, 128, 131, 131, 133, 133, 134, 134, 137, 137, 138, 138, 140, 140, 143, 143, 145, 145, 146, 146, 148, 148, 151, 151, 152, 152, 155, 155, 157, 157, 158, 158, 161, 161, 162, 162, 164, 164, 167, 167, 168, 168, 171, 171, 173, 173, 174, 174, 176, 176, 179, 179, 181, 181, 182, 182, 185, 185, 186, 186, 188, 188, 191, 191, 193, 193, 194, 194, 196, 196, 199, 199, 200, 200, 203, 203, 205, 205, 206, 206, 208, 208, 211, 211, 213, 213, 214, 214, 217, 217, 218, 218, 220, 220, 223, 223, 224, 224, 227, 227, 229, 229, 230, 230, 233, 233, 234, 234, 236, 236, 239, 239, 241, 241, 242, 242, 244, 244, 247, 247, 248, 248, 251, 251, 253, 253, 254, 254]
p = [8, 5, 4, 2, 11, 9, 13, 3, 0, 6, 1, 12, 14, 10, 15, 7]
aqwerty = '!@#$%^&*()qwertyUIOPAzxcvbnmQQQQQQQQQQQQ)(*@&%\x00'
anum = '0123456789012345678901234567890123456789\x00'
antpassword = 'NTPASSWORD\x00'
almpassword = 'LMPASSWORD\x00'
lmkey = 'KGS!@#$%'
empty_lm = 'aad3b435b51404eeaad3b435b51404ee'.decode('hex')
empty_nt = '31d6cfe0d16ae931b73c59d7e0c089c0'.decode('hex')

def str_to_key(s):
    if False:
        for i in range(10):
            print('nop')
    key = []
    key.append(ord(s[0]) >> 1)
    key.append((ord(s[0]) & 1) << 6 | ord(s[1]) >> 2)
    key.append((ord(s[1]) & 3) << 5 | ord(s[2]) >> 3)
    key.append((ord(s[2]) & 7) << 4 | ord(s[3]) >> 4)
    key.append((ord(s[3]) & 15) << 3 | ord(s[4]) >> 5)
    key.append((ord(s[4]) & 31) << 2 | ord(s[5]) >> 6)
    key.append((ord(s[5]) & 63) << 1 | ord(s[6]) >> 7)
    key.append(ord(s[6]) & 127)
    for i in range(8):
        key[i] = key[i] << 1
        key[i] = odd_parity[key[i]]
    return ''.join((chr(k) for k in key))

def sid_to_key(sid):
    if False:
        for i in range(10):
            print('nop')
    s1 = ''
    s1 += chr(sid & 255)
    s1 += chr(sid >> 8 & 255)
    s1 += chr(sid >> 16 & 255)
    s1 += chr(sid >> 24 & 255)
    s1 += s1[0]
    s1 += s1[1]
    s1 += s1[2]
    s2 = s1[3] + s1[0] + s1[1] + s1[2]
    s2 += s2[0] + s2[1] + s2[2]
    return (str_to_key(s1), str_to_key(s2))

def hash_lm(pw):
    if False:
        i = 10
        return i + 15
    pw = pw[:14].upper()
    pw = pw + '\x00' * (14 - len(pw))
    d1 = DES.new(str_to_key(pw[:7]), DES.MODE_ECB)
    d2 = DES.new(str_to_key(pw[7:]), DES.MODE_ECB)
    return d1.encrypt(lmkey) + d2.encrypt(lmkey)

def hash_nt(pw):
    if False:
        for i in range(10):
            print('nop')
    return MD4.new(pw.encode('utf-16-le')).digest()

def find_control_set(sysaddr):
    if False:
        while True:
            i = 10
    root = rawreg.get_root(sysaddr)
    if not root:
        return 1
    csselect = rawreg.open_key(root, ['Select'])
    if not csselect:
        return 1
    for v in rawreg.values(csselect):
        if v.Name == 'Current':
            return v.Data
    return 1

def get_bootkey(sysaddr):
    if False:
        for i in range(10):
            print('nop')
    cs = find_control_set(sysaddr)
    lsa_base = ['ControlSet{0:03}'.format(cs), 'Control', 'Lsa']
    lsa_keys = ['JD', 'Skew1', 'GBG', 'Data']
    root = rawreg.get_root(sysaddr)
    if not root:
        return None
    lsa = rawreg.open_key(root, lsa_base)
    if not lsa:
        return None
    bootkey = ''
    for lk in lsa_keys:
        key = rawreg.open_key(lsa, [lk])
        class_data = sysaddr.read(key.Class, key.ClassLength)
        if class_data == None:
            return ''
        bootkey += class_data.decode('utf-16-le').decode('hex')
    bootkey_scrambled = ''
    for i in range(len(bootkey)):
        bootkey_scrambled += bootkey[p[i]]
    return bootkey_scrambled

def get_hbootkey(samaddr, bootkey):
    if False:
        return 10
    sam_account_path = ['SAM', 'Domains', 'Account']
    if not bootkey:
        return None
    root = rawreg.get_root(samaddr)
    if not root:
        return None
    sam_account_key = rawreg.open_key(root, sam_account_path)
    if not sam_account_key:
        return None
    F = None
    for v in rawreg.values(sam_account_key):
        if v.Name == 'F':
            F = samaddr.read(v.Data, v.DataLength)
    if not F:
        return None
    md5 = MD5.new()
    md5.update(F[112:128] + aqwerty + bootkey + anum)
    rc4_key = md5.digest()
    rc4 = ARC4.new(rc4_key)
    hbootkey = rc4.encrypt(F[128:160])
    return hbootkey

def get_user_keys(samaddr):
    if False:
        print('Hello World!')
    user_key_path = ['SAM', 'Domains', 'Account', 'Users']
    root = rawreg.get_root(samaddr)
    if not root:
        return []
    user_key = rawreg.open_key(root, user_key_path)
    if not user_key:
        return []
    return [k for k in rawreg.subkeys(user_key) if k.Name != 'Names']

def decrypt_single_hash(rid, hbootkey, enc_hash, lmntstr):
    if False:
        while True:
            i = 10
    (des_k1, des_k2) = sid_to_key(rid)
    d1 = DES.new(des_k1, DES.MODE_ECB)
    d2 = DES.new(des_k2, DES.MODE_ECB)
    md5 = MD5.new()
    md5.update(hbootkey[:16] + pack('<L', rid) + lmntstr)
    rc4_key = md5.digest()
    rc4 = ARC4.new(rc4_key)
    obfkey = rc4.encrypt(enc_hash)
    hash = d1.decrypt(obfkey[:8]) + d2.decrypt(obfkey[8:])
    return hash

def decrypt_hashes(rid, enc_lm_hash, enc_nt_hash, hbootkey):
    if False:
        i = 10
        return i + 15
    if enc_lm_hash:
        lmhash = decrypt_single_hash(rid, hbootkey, enc_lm_hash, almpassword)
    else:
        lmhash = ''
    if enc_nt_hash:
        nthash = decrypt_single_hash(rid, hbootkey, enc_nt_hash, antpassword)
    else:
        nthash = ''
    return (lmhash, nthash)

def encrypt_single_hash(rid, hbootkey, hash, lmntstr):
    if False:
        print('Hello World!')
    (des_k1, des_k2) = sid_to_key(rid)
    d1 = DES.new(des_k1, DES.MODE_ECB)
    d2 = DES.new(des_k2, DES.MODE_ECB)
    enc_hash = d1.encrypt(hash[:8]) + d2.encrypt(hash[8:])
    md5 = MD5.new()
    md5.update(hbootkey[:16] + pack('<L', rid) + lmntstr)
    rc4_key = md5.digest()
    rc4 = ARC4.new(rc4_key)
    obfkey = rc4.encrypt(enc_hash)
    return obfkey

def encrypt_hashes(rid, lm_hash, nt_hash, hbootkey):
    if False:
        while True:
            i = 10
    if lm_hash:
        enc_lmhash = encrypt_single_hash(rid, hbootkey, lm_hash, almpassword)
    else:
        enc_lmhash = ''
    if nt_hash:
        enc_nthash = encrypt_single_hash(rid, hbootkey, nt_hash, antpassword)
    else:
        enc_nthash = ''
    return (enc_lmhash, enc_nthash)

def get_user_hashes(user_key, hbootkey):
    if False:
        print('Hello World!')
    samaddr = user_key.obj_vm
    rid = int(str(user_key.Name), 16)
    V = None
    for v in rawreg.values(user_key):
        if v.Name == 'V':
            V = samaddr.read(v.Data, v.DataLength)
    if not V:
        return None
    lm_offset = unpack('<L', V[156:160])[0] + 204 + 4
    lm_len = unpack('<L', V[160:164])[0] - 4
    nt_offset = unpack('<L', V[168:172])[0] + 204 + 4
    nt_len = unpack('<L', V[172:176])[0] - 4
    if lm_len:
        enc_lm_hash = V[lm_offset:lm_offset + 16]
    else:
        enc_lm_hash = ''
    if nt_len:
        enc_nt_hash = V[nt_offset:nt_offset + 16]
    else:
        enc_nt_hash = ''
    return decrypt_hashes(rid, enc_lm_hash, enc_nt_hash, hbootkey)

def get_user_name(user_key):
    if False:
        while True:
            i = 10
    samaddr = user_key.obj_vm
    V = None
    for v in rawreg.values(user_key):
        if v.Name == 'V':
            V = samaddr.read(v.Data, v.DataLength)
    if not V:
        return None
    name_offset = unpack('<L', V[12:16])[0] + 204
    name_length = unpack('<L', V[16:20])[0]
    if name_length > len(V):
        return None
    username = V[name_offset:name_offset + name_length].decode('utf-16-le')
    return username

def get_user_desc(user_key):
    if False:
        while True:
            i = 10
    samaddr = user_key.obj_vm
    V = None
    for v in rawreg.values(user_key):
        if v.Name == 'V':
            V = samaddr.read(v.Data, v.DataLength)
    if not V:
        return None
    desc_offset = unpack('<L', V[36:40])[0] + 204
    desc_length = unpack('<L', V[40:44])[0]
    desc = V[desc_offset:desc_offset + desc_length].decode('utf-16-le')
    return desc

def dump_hashes(sysaddr, samaddr):
    if False:
        for i in range(10):
            print('nop')
    if sysaddr == None:
        yield obj.NoneObject('SYSTEM address is None: Did you use the correct profile?')
    if samaddr == None:
        yield obj.NoneObject('SAM address is None: Did you use the correct profile?')
    bootkey = get_bootkey(sysaddr)
    hbootkey = get_hbootkey(samaddr, bootkey)
    if hbootkey:
        for user in get_user_keys(samaddr):
            ret = get_user_hashes(user, hbootkey)
            if not ret:
                yield obj.NoneObject('Cannot get user hashes for {0}'.format(user))
            else:
                (lmhash, nthash) = ret
                if not lmhash:
                    lmhash = empty_lm
                if not nthash:
                    nthash = empty_nt
                name = get_user_name(user)
                if name is not None:
                    name = name.encode('ascii', 'ignore')
                else:
                    name = '(unavailable)'
                yield '{0}:{1}:{2}:{3}:::'.format(name, int(str(user.Name), 16), lmhash.encode('hex'), nthash.encode('hex'))
    else:
        yield obj.NoneObject('Hbootkey is not valid')

def dump_memory_hashes(addr_space, config, syshive, samhive):
    if False:
        i = 10
        return i + 15
    if syshive != None and samhive != None:
        sysaddr = hive.HiveAddressSpace(addr_space, config, syshive)
        samaddr = hive.HiveAddressSpace(addr_space, config, samhive)
        return dump_hashes(sysaddr, samaddr)
    return obj.NoneObject('SYSTEM or SAM address is None: Did you use the correct profile?')

def dump_file_hashes(syshive_fname, samhive_fname):
    if False:
        i = 10
        return i + 15
    sysaddr = hive.HiveFileAddressSpace(syshive_fname)
    samaddr = hive.HiveFileAddressSpace(samhive_fname)
    return dump_hashes(sysaddr, samaddr)