import struct
import pefile
from PyInstaller.compat import win32api

def pefile_check_control_flow_guard(filename):
    if False:
        print('Hello World!')
    '\n    Checks if the specified PE file has CFG (Control Flow Guard) enabled.\n\n    Parameters\n    ----------\n    filename : str\n        Path to the PE file to inspect.\n\n    Returns\n    ----------\n    bool\n        True if file is a PE file with CFG enabled. False if CFG is not enabled or if file could not be processed using\n        the pefile library.\n    '
    try:
        pe = pefile.PE(filename, fast_load=True)
        return bool(pe.OPTIONAL_HEADER.DllCharacteristics & 16384)
    except Exception:
        return False
LOAD_LIBRARY_AS_DATAFILE = 2

def getRaw(text):
    if False:
        i = 10
        return i + 15
    "\n    Encodes text as UTF-16LE (Microsoft 'Unicode') for use in structs.\n    "
    return text.encode('UTF-16LE')

def read_version_info_from_executable(exe_filename):
    if False:
        return 10
    "\n    Read the version information structure from the given executable's resources, and return it as an instance of\n    `VSVersionInfo` structure.\n    "
    h = win32api.LoadLibraryEx(exe_filename, 0, LOAD_LIBRARY_AS_DATAFILE)
    res = win32api.EnumResourceNames(h, pefile.RESOURCE_TYPE['RT_VERSION'])
    if not len(res):
        return None
    data = win32api.LoadResource(h, pefile.RESOURCE_TYPE['RT_VERSION'], res[0])
    info = VSVersionInfo()
    info.fromRaw(data)
    win32api.FreeLibrary(h)
    return info

def nextDWord(offset):
    if False:
        print('Hello World!')
    '\n    Align `offset` to the next 4-byte boundary.\n    '
    return offset + 3 >> 2 << 2

class VSVersionInfo:
    """
    WORD  wLength;        // length of the VS_VERSION_INFO structure
    WORD  wValueLength;   // length of the Value member
    WORD  wType;          // 1 means text, 0 means binary
    WCHAR szKey[];        // Contains the Unicode string "VS_VERSION_INFO".
    WORD  Padding1[];
    VS_FIXEDFILEINFO Value;
    WORD  Padding2[];
    WORD  Children[];     // zero or more StringFileInfo or VarFileInfo
                          // structures (or both) that are children of the
                          // current version structure.
    """

    def __init__(self, ffi=None, kids=None):
        if False:
            i = 10
            return i + 15
        self.ffi = ffi
        self.kids = kids or []

    def fromRaw(self, data):
        if False:
            return 10
        (i, (sublen, vallen, wType, nm)) = parseCommon(data)
        i = nextDWord(i)
        self.ffi = FixedFileInfo()
        j = self.ffi.fromRaw(data, i)
        i = j
        while i < sublen:
            j = i
            (i, (csublen, cvallen, ctyp, nm)) = parseCommon(data, i)
            if nm.strip() == 'StringFileInfo':
                sfi = StringFileInfo()
                k = sfi.fromRaw(csublen, cvallen, nm, data, i, j + csublen)
                self.kids.append(sfi)
                i = k
            else:
                vfi = VarFileInfo()
                k = vfi.fromRaw(csublen, cvallen, nm, data, i, j + csublen)
                self.kids.append(vfi)
                i = k
            i = j + csublen
            i = nextDWord(i)
        return i

    def toRaw(self):
        if False:
            for i in range(10):
                print('nop')
        raw_name = getRaw('VS_VERSION_INFO')
        rawffi = self.ffi.toRaw()
        vallen = len(rawffi)
        typ = 0
        sublen = 6 + len(raw_name) + 2
        pad = b''
        if sublen % 4:
            pad = b'\x00\x00'
        sublen = sublen + len(pad) + vallen
        pad2 = b''
        if sublen % 4:
            pad2 = b'\x00\x00'
        tmp = b''.join([kid.toRaw() for kid in self.kids])
        sublen = sublen + len(pad2) + len(tmp)
        return struct.pack('hhh', sublen, vallen, typ) + raw_name + b'\x00\x00' + pad + rawffi + pad2 + tmp

    def __eq__(self, other):
        if False:
            return 10
        return self.toRaw() == other

    def __str__(self, indent=''):
        if False:
            print('Hello World!')
        indent = indent + '  '
        tmp = [kid.__str__(indent + '  ') for kid in self.kids]
        tmp = ', \n'.join(tmp)
        return '\n'.join(['# UTF-8', '#', "# For more details about fixed file info 'ffi' see:", '# http://msdn.microsoft.com/en-us/library/ms646997.aspx', 'VSVersionInfo(', indent + f'ffi={self.ffi.__str__(indent)},', indent + 'kids=[', tmp, indent + ']', ')'])

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'versioninfo.VSVersionInfo(ffi=%r, kids=%r)' % (self.ffi, self.kids)

def parseCommon(data, start=0):
    if False:
        for i in range(10):
            print('nop')
    i = start + 6
    (wLength, wValueLength, wType) = struct.unpack('3h', data[start:i])
    (i, text) = parseUString(data, i, i + wLength)
    return (i, (wLength, wValueLength, wType, text))

def parseUString(data, start, limit):
    if False:
        while True:
            i = 10
    i = start
    while i < limit:
        if data[i:i + 2] == b'\x00\x00':
            break
        i += 2
    text = data[start:i].decode('UTF-16LE')
    i += 2
    return (i, text)

class FixedFileInfo:
    """
    DWORD dwSignature;        //Contains the value 0xFEEFO4BD
    DWORD dwStrucVersion;     // binary version number of this structure.
                              // The high-order word of this member contains
                              // the major version number, and the low-order
                              // word contains the minor version number.
    DWORD dwFileVersionMS;    // most significant 32 bits of the file's binary
                              // version number
    DWORD dwFileVersionLS;    //
    DWORD dwProductVersionMS; // most significant 32 bits of the binary version
                              // number of the product with which this file was
                              // distributed
    DWORD dwProductVersionLS; //
    DWORD dwFileFlagsMask;    // bitmask that specifies the valid bits in
                              // dwFileFlags. A bit is valid only if it was
                              // defined when the file was created.
    DWORD dwFileFlags;        // VS_FF_DEBUG, VS_FF_PATCHED etc.
    DWORD dwFileOS;           // VOS_NT, VOS_WINDOWS32 etc.
    DWORD dwFileType;         // VFT_APP etc.
    DWORD dwFileSubtype;      // 0 unless VFT_DRV or VFT_FONT or VFT_VXD
    DWORD dwFileDateMS;
    DWORD dwFileDateLS;
    """

    def __init__(self, filevers=(0, 0, 0, 0), prodvers=(0, 0, 0, 0), mask=63, flags=0, OS=262148, fileType=1, subtype=0, date=(0, 0)):
        if False:
            while True:
                i = 10
        self.sig = 4277077181
        self.strucVersion = 65536
        self.fileVersionMS = filevers[0] << 16 | filevers[1] & 65535
        self.fileVersionLS = filevers[2] << 16 | filevers[3] & 65535
        self.productVersionMS = prodvers[0] << 16 | prodvers[1] & 65535
        self.productVersionLS = prodvers[2] << 16 | prodvers[3] & 65535
        self.fileFlagsMask = mask
        self.fileFlags = flags
        self.fileOS = OS
        self.fileType = fileType
        self.fileSubtype = subtype
        self.fileDateMS = date[0]
        self.fileDateLS = date[1]

    def fromRaw(self, data, i):
        if False:
            return 10
        (self.sig, self.strucVersion, self.fileVersionMS, self.fileVersionLS, self.productVersionMS, self.productVersionLS, self.fileFlagsMask, self.fileFlags, self.fileOS, self.fileType, self.fileSubtype, self.fileDateMS, self.fileDateLS) = struct.unpack('13L', data[i:i + 52])
        return i + 52

    def toRaw(self):
        if False:
            for i in range(10):
                print('nop')
        return struct.pack('13L', self.sig, self.strucVersion, self.fileVersionMS, self.fileVersionLS, self.productVersionMS, self.productVersionLS, self.fileFlagsMask, self.fileFlags, self.fileOS, self.fileType, self.fileSubtype, self.fileDateMS, self.fileDateLS)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.toRaw() == other

    def __str__(self, indent=''):
        if False:
            print('Hello World!')
        fv = (self.fileVersionMS >> 16, self.fileVersionMS & 65535, self.fileVersionLS >> 16, self.fileVersionLS & 65535)
        pv = (self.productVersionMS >> 16, self.productVersionMS & 65535, self.productVersionLS >> 16, self.productVersionLS & 65535)
        fd = (self.fileDateMS, self.fileDateLS)
        tmp = ['FixedFileInfo(', '# filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)', '# Set not needed items to zero 0.', 'filevers=%s,' % (fv,), 'prodvers=%s,' % (pv,), "# Contains a bitmask that specifies the valid bits 'flags'r", 'mask=%s,' % hex(self.fileFlagsMask), '# Contains a bitmask that specifies the Boolean attributes of the file.', 'flags=%s,' % hex(self.fileFlags), '# The operating system for which this file was designed.', '# 0x4 - NT and there is no need to change it.', 'OS=%s,' % hex(self.fileOS), '# The general type of file.', '# 0x1 - the file is an application.', 'fileType=%s,' % hex(self.fileType), '# The function of the file.', '# 0x0 - the function is not defined for this fileType', 'subtype=%s,' % hex(self.fileSubtype), '# Creation date and time stamp.', 'date=%s' % (fd,), ')']
        return f'\n{indent}  '.join(tmp)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        fv = (self.fileVersionMS >> 16, self.fileVersionMS & 65535, self.fileVersionLS >> 16, self.fileVersionLS & 65535)
        pv = (self.productVersionMS >> 16, self.productVersionMS & 65535, self.productVersionLS >> 16, self.productVersionLS & 65535)
        fd = (self.fileDateMS, self.fileDateLS)
        return 'versioninfo.FixedFileInfo(filevers=%r, prodvers=%r, mask=0x%x, flags=0x%x, OS=0x%x, fileType=%r, subtype=0x%x, date=%r)' % (fv, pv, self.fileFlagsMask, self.fileFlags, self.fileOS, self.fileType, self.fileSubtype, fd)

class StringFileInfo:
    """
    WORD        wLength;      // length of the version resource
    WORD        wValueLength; // length of the Value member in the current
                              // VS_VERSION_INFO structure
    WORD        wType;        // 1 means text, 0 means binary
    WCHAR       szKey[];      // Contains the Unicode string 'StringFileInfo'.
    WORD        Padding[];
    StringTable Children[];   // list of zero or more String structures
    """

    def __init__(self, kids=None):
        if False:
            while True:
                i = 10
        self.name = 'StringFileInfo'
        self.kids = kids or []

    def fromRaw(self, sublen, vallen, name, data, i, limit):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        while i < limit:
            st = StringTable()
            j = st.fromRaw(data, i, limit)
            self.kids.append(st)
            i = j
        return i

    def toRaw(self):
        if False:
            while True:
                i = 10
        raw_name = getRaw(self.name)
        vallen = 0
        typ = 1
        sublen = 6 + len(raw_name) + 2
        pad = b''
        if sublen % 4:
            pad = b'\x00\x00'
        tmp = b''.join([kid.toRaw() for kid in self.kids])
        sublen = sublen + len(pad) + len(tmp)
        return struct.pack('hhh', sublen, vallen, typ) + raw_name + b'\x00\x00' + pad + tmp

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.toRaw() == other

    def __str__(self, indent=''):
        if False:
            print('Hello World!')
        new_indent = indent + '  '
        tmp = ', \n'.join((kid.__str__(new_indent) for kid in self.kids))
        return f'{indent}StringFileInfo(\n{new_indent}[\n{tmp}\n{new_indent}])'

    def __repr__(self):
        if False:
            return 10
        return 'versioninfo.StringFileInfo(%r)' % self.kids

class StringTable:
    """
    WORD   wLength;
    WORD   wValueLength;
    WORD   wType;
    WCHAR  szKey[];
    String Children[];    // list of zero or more String structures.
    """

    def __init__(self, name=None, kids=None):
        if False:
            i = 10
            return i + 15
        self.name = name or ''
        self.kids = kids or []

    def fromRaw(self, data, i, limit):
        if False:
            while True:
                i = 10
        (i, (cpsublen, cpwValueLength, cpwType, self.name)) = parseCodePage(data, i, limit)
        i = nextDWord(i)
        while i < limit:
            ss = StringStruct()
            j = ss.fromRaw(data, i, limit)
            i = j
            self.kids.append(ss)
            i = nextDWord(i)
        return i

    def toRaw(self):
        if False:
            print('Hello World!')
        raw_name = getRaw(self.name)
        vallen = 0
        typ = 1
        sublen = 6 + len(raw_name) + 2
        tmp = []
        for kid in self.kids:
            raw = kid.toRaw()
            if len(raw) % 4:
                raw = raw + b'\x00\x00'
            tmp.append(raw)
        tmp = b''.join(tmp)
        sublen += len(tmp)
        return struct.pack('hhh', sublen, vallen, typ) + raw_name + b'\x00\x00' + tmp

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.toRaw() == other

    def __str__(self, indent=''):
        if False:
            return 10
        new_indent = indent + '  '
        tmp = (',\n' + new_indent).join((str(kid) for kid in self.kids))
        return f"{indent}StringTable(\n{new_indent}'{self.name}',\n{new_indent}[{tmp}])"

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'versioninfo.StringTable(%r, %r)' % (self.name, self.kids)

class StringStruct:
    """
    WORD   wLength;
    WORD   wValueLength;
    WORD   wType;
    WCHAR  szKey[];
    WORD   Padding[];
    String Value[];
    """

    def __init__(self, name=None, val=None):
        if False:
            while True:
                i = 10
        self.name = name or ''
        self.val = val or ''

    def fromRaw(self, data, i, limit):
        if False:
            return 10
        (i, (sublen, vallen, typ, self.name)) = parseCommon(data, i)
        limit = i + sublen
        i = nextDWord(i)
        (i, self.val) = parseUString(data, i, limit)
        return i

    def toRaw(self):
        if False:
            print('Hello World!')
        raw_name = getRaw(self.name)
        raw_val = getRaw(self.val)
        vallen = len(self.val) + 1
        typ = 1
        sublen = 6 + len(raw_name) + 2
        pad = b''
        if sublen % 4:
            pad = b'\x00\x00'
        sublen = sublen + len(pad) + vallen * 2
        return struct.pack('hhh', sublen, vallen, typ) + raw_name + b'\x00\x00' + pad + raw_val + b'\x00\x00'

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.toRaw() == other

    def __str__(self, indent=''):
        if False:
            while True:
                i = 10
        return 'StringStruct(%r, %r)' % (self.name, self.val)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'versioninfo.StringStruct(%r, %r)' % (self.name, self.val)

def parseCodePage(data, i, limit):
    if False:
        for i in range(10):
            print('nop')
    (i, (sublen, wValueLength, wType, nm)) = parseCommon(data, i)
    return (i, (sublen, wValueLength, wType, nm))

class VarFileInfo:
    """
    WORD  wLength;        // length of the version resource
    WORD  wValueLength;   // length of the Value member in the current
                          // VS_VERSION_INFO structure
    WORD  wType;          // 1 means text, 0 means binary
    WCHAR szKey[];        // Contains the Unicode string 'VarFileInfo'.
    WORD  Padding[];
    Var   Children[];     // list of zero or more Var structures
    """

    def __init__(self, kids=None):
        if False:
            for i in range(10):
                print('nop')
        self.kids = kids or []

    def fromRaw(self, sublen, vallen, name, data, i, limit):
        if False:
            print('Hello World!')
        self.sublen = sublen
        self.vallen = vallen
        self.name = name
        i = nextDWord(i)
        while i < limit:
            vs = VarStruct()
            j = vs.fromRaw(data, i, limit)
            self.kids.append(vs)
            i = j
        return i

    def toRaw(self):
        if False:
            print('Hello World!')
        self.vallen = 0
        self.wType = 1
        self.name = 'VarFileInfo'
        raw_name = getRaw(self.name)
        sublen = 6 + len(raw_name) + 2
        pad = b''
        if sublen % 4:
            pad = b'\x00\x00'
        tmp = b''.join([kid.toRaw() for kid in self.kids])
        self.sublen = sublen + len(pad) + len(tmp)
        return struct.pack('hhh', self.sublen, self.vallen, self.wType) + raw_name + b'\x00\x00' + pad + tmp

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.toRaw() == other

    def __str__(self, indent=''):
        if False:
            return 10
        return indent + 'VarFileInfo([%s])' % ', '.join((str(kid) for kid in self.kids))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'versioninfo.VarFileInfo(%r)' % self.kids

class VarStruct:
    """
    WORD  wLength;        // length of the version resource
    WORD  wValueLength;   // length of the Value member in the current
                          // VS_VERSION_INFO structure
    WORD  wType;          // 1 means text, 0 means binary
    WCHAR szKey[];        // Contains the Unicode string 'Translation'
                          // or a user-defined key string value
    WORD  Padding[];      //
    WORD  Value[];        // list of one or more values that are language
                          // and code-page identifiers
    """

    def __init__(self, name=None, kids=None):
        if False:
            return 10
        self.name = name or ''
        self.kids = kids or []

    def fromRaw(self, data, i, limit):
        if False:
            for i in range(10):
                print('nop')
        (i, (self.sublen, self.wValueLength, self.wType, self.name)) = parseCommon(data, i)
        i = nextDWord(i)
        for j in range(0, self.wValueLength, 2):
            kid = struct.unpack('h', data[i:i + 2])[0]
            self.kids.append(kid)
            i += 2
        return i

    def toRaw(self):
        if False:
            return 10
        self.wValueLength = len(self.kids) * 2
        self.wType = 0
        raw_name = getRaw(self.name)
        sublen = 6 + len(raw_name) + 2
        pad = b''
        if sublen % 4:
            pad = b'\x00\x00'
        self.sublen = sublen + len(pad) + self.wValueLength
        tmp = b''.join([struct.pack('h', kid) for kid in self.kids])
        return struct.pack('hhh', self.sublen, self.wValueLength, self.wType) + raw_name + b'\x00\x00' + pad + tmp

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.toRaw() == other

    def __str__(self, indent=''):
        if False:
            for i in range(10):
                print('nop')
        return "VarStruct('%s', %r)" % (self.name, self.kids)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'versioninfo.VarStruct(%r, %r)' % (self.name, self.kids)

def load_version_info_from_text_file(filename):
    if False:
        return 10
    '\n    Load the `VSVersionInfo` structure from its string-based (`VSVersionInfo.__str__`) serialization by reading the\n    text from the file and running it through `eval()`.\n    '
    import PyInstaller.utils.misc as miscutils
    with open(filename, 'rb') as fp:
        text = miscutils.decode(fp.read())
    try:
        info = eval(text)
    except Exception as e:
        raise ValueError('Failed to deserialize VSVersionInfo from text-based representation!') from e
    assert isinstance(info, VSVersionInfo), f'Loaded incompatible structure type! Expected VSVersionInfo, got: {type(info)!r}'
    return info

def write_version_info_to_executable(exe_filename, info):
    if False:
        i = 10
        return i + 15
    assert isinstance(info, VSVersionInfo)
    pe = pefile.PE(exe_filename, fast_load=True)
    overlay_before = pe.get_overlay()
    pe.close()
    hdst = win32api.BeginUpdateResource(exe_filename, 0)
    win32api.UpdateResource(hdst, pefile.RESOURCE_TYPE['RT_VERSION'], 1, info.toRaw())
    win32api.EndUpdateResource(hdst, 0)
    if overlay_before:
        pe = pefile.PE(exe_filename, fast_load=True)
        overlay_after = pe.get_overlay()
        pe.close()
        if not overlay_after:
            with open(exe_filename, 'ab') as exef:
                exef.write(overlay_before)