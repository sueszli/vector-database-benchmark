"""
@author:       Brendan Dolan-Gavitt
@license:      GNU General Public License 2.0
@contact:      bdolangavitt@wesleyan.edu
"""
import volatility.debug as debug
import volatility.obj as obj
import struct
ROOT_INDEX = 32
LH_SIG = 'lh'
LF_SIG = 'lf'
RI_SIG = 'ri'
NK_SIG = 'nk'
VK_SIG = 'vk'
BIG_DATA_MAGIC = 16344
KEY_FLAGS = {'KEY_IS_VOLATILE': 1, 'KEY_HIVE_EXIT': 2, 'KEY_HIVE_ENTRY': 4, 'KEY_NO_DELETE': 8, 'KEY_SYM_LINK': 16, 'KEY_COMP_NAME': 32, 'KEY_PREFEF_HANDLE': 64, 'KEY_VIRT_MIRRORED': 128, 'KEY_VIRT_TARGET': 256, 'KEY_VIRTUAL_STORE': 512}
VALUE_TYPES = dict(enumerate(['REG_NONE', 'REG_SZ', 'REG_EXPAND_SZ', 'REG_BINARY', 'REG_DWORD', 'REG_DWORD_BIG_ENDIAN', 'REG_LINK', 'REG_MULTI_SZ', 'REG_RESOURCE_LIST', 'REG_FULL_RESOURCE_DESCRIPTOR', 'REG_RESOURCE_REQUIREMENTS_LIST', 'REG_QWORD']))

def get_root(address_space, stable=True):
    if False:
        while True:
            i = 10
    if stable:
        return obj.Object('_CM_KEY_NODE', ROOT_INDEX, address_space)
    else:
        return obj.Object('_CM_KEY_NODE', ROOT_INDEX | 2147483648, address_space)

def open_key(root, key):
    if False:
        print('Hello World!')
    if key == []:
        return root
    if not root.is_valid():
        return None
    keyname = key.pop(0)
    for s in subkeys(root):
        if s.Name.upper() == keyname.upper():
            return open_key(s, key)
    debug.debug("Couldn't find subkey {0} of {1}".format(keyname, root.Name), 1)
    return obj.NoneObject("Couldn't find subkey {0} of {1}".format(keyname, root.Name))

def read_sklist(sk):
    if False:
        return 10
    if sk.Signature.v() == LH_SIG or sk.Signature.v() == LF_SIG:
        for i in sk.List:
            yield i
    elif sk.Signature.v() == RI_SIG:
        for i in range(sk.Count):
            ptr_off = sk.List.obj_offset + i * 4
            if not sk.obj_vm.is_valid_address(ptr_off):
                continue
            ssk_off = obj.Object('unsigned int', ptr_off, sk.obj_vm)
            if not sk.obj_vm.is_valid_address(ssk_off):
                continue
            ssk = obj.Object('_CM_KEY_INDEX', ssk_off, sk.obj_vm)
            if ssk == sk:
                break
            for i in read_sklist(ssk):
                yield i

def subkeys(key):
    if False:
        print('Hello World!')
    if not key.is_valid():
        return
    for index in range(2):
        if int(key.SubKeyCounts[index]) > 0:
            sk_off = key.SubKeyLists[index]
            sk = obj.Object('_CM_KEY_INDEX', sk_off, key.obj_vm)
            if not sk or not sk.is_valid():
                pass
            else:
                for i in read_sklist(sk):
                    if i.Signature.v() == NK_SIG and i.Parent.dereference().Name == key.Name:
                        yield i

def values(key):
    if False:
        for i in range(10):
            print('nop')
    return [v for v in key.ValueList.List.dereference() if v.Signature.v() == VK_SIG]

def key_flags(key):
    if False:
        i = 10
        return i + 15
    return [k for k in KEY_FLAGS if key.Flags & KEY_FLAGS[k]]
value_formats = {'REG_DWORD': '<L', 'REG_DWORD_BIG_ENDIAN': '>L', 'REG_QWORD': '<Q'}

def value_data(val):
    if False:
        while True:
            i = 10
    inline = val.DataLength & 2147483648
    if inline:
        inline_len = val.DataLength & 2147483647
        if inline_len == 0 or inline_len > 4:
            valdata = None
        else:
            valdata = val.obj_vm.read(val.Data.obj_offset, inline_len)
    elif val.obj_vm.hive.Version == 5 and val.DataLength > 16384:
        datalen = val.DataLength
        big_data = obj.Object('_CM_BIG_DATA', val.Data, val.obj_vm)
        valdata = ''
        thelist = []
        if not big_data.Count or big_data.Count > 2147483648:
            thelist = []
        else:
            for i in range(big_data.Count):
                ptr_off = big_data.List + i * 4
                chunk_addr = obj.Object('unsigned int', ptr_off, val.obj_vm)
                if not val.obj_vm.is_valid_address(chunk_addr):
                    continue
                thelist.append(chunk_addr)
        for chunk in thelist:
            amount_to_read = min(BIG_DATA_MAGIC, datalen)
            chunk_data = val.obj_vm.read(chunk, amount_to_read)
            if not chunk_data:
                valdata = None
                break
            valdata += chunk_data
            datalen -= amount_to_read
    else:
        valdata = val.obj_vm.read(val.Data, val.DataLength)
    valtype = VALUE_TYPES.get(val.Type.v(), 'REG_UNKNOWN')
    if valdata == None:
        return (valtype, obj.NoneObject('Value data is unreadable'))
    if valtype in ['REG_DWORD', 'REG_DWORD_BIG_ENDIAN', 'REG_QWORD']:
        if len(valdata) != struct.calcsize(value_formats[valtype]):
            return (valtype, obj.NoneObject('Value data did not match the expected data size for a {0}'.format(valtype)))
    if valtype in ['REG_SZ', 'REG_EXPAND_SZ', 'REG_LINK']:
        valdata = valdata.decode('utf-16-le', 'ignore')
    elif valtype == 'REG_MULTI_SZ':
        valdata = valdata.decode('utf-16-le', 'ignore').split('\x00')
    elif valtype in ['REG_DWORD', 'REG_DWORD_BIG_ENDIAN', 'REG_QWORD']:
        valdata = struct.unpack(value_formats[valtype], valdata)[0]
    return (valtype, valdata)

def walk(root):
    if False:
        while True:
            i = 10
    yield root
    for k in subkeys(root):
        for j in walk(k):
            yield j