""" Stamp a Win32 binary with version information.
"""
import glob
import optparse
import os
import struct
from win32api import BeginUpdateResource, EndUpdateResource, UpdateResource
VS_FFI_SIGNATURE = -17890115
VS_FFI_STRUCVERSION = 65536
VS_FFI_FILEFLAGSMASK = 63
VOS_NT_WINDOWS32 = 262148
null_byte = b'\x00'

def file_flags(debug):
    if False:
        for i in range(10):
            print('nop')
    if debug:
        return 3
    return 0

def file_type(is_dll):
    if False:
        i = 10
        return i + 15
    if is_dll:
        return 2
    return 1

def VS_FIXEDFILEINFO(maj, min, sub, build, debug=0, is_dll=1):
    if False:
        return 10
    return struct.pack('lllllllllllll', VS_FFI_SIGNATURE, VS_FFI_STRUCVERSION, maj << 16 | min, sub << 16 | build, maj << 16 | min, sub << 16 | build, VS_FFI_FILEFLAGSMASK, file_flags(debug), VOS_NT_WINDOWS32, file_type(is_dll), 0, 0, 0)

def nullterm(s):
    if False:
        i = 10
        return i + 15
    return (str(s) + '\x00').encode('utf-16le')

def pad32(s, extra=2):
    if False:
        i = 10
        return i + 15
    l = 4 - (len(s) + extra & 3)
    if l < 4:
        return s + null_byte * l
    return s

def addlen(s):
    if False:
        for i in range(10):
            print('nop')
    return struct.pack('h', len(s) + 2) + s

def String(key, value):
    if False:
        return 10
    key = nullterm(key)
    value = nullterm(value)
    result = struct.pack('hh', len(value) // 2, 1)
    result = result + key
    result = pad32(result) + value
    return addlen(result)

def StringTable(key, data):
    if False:
        for i in range(10):
            print('nop')
    key = nullterm(key)
    result = struct.pack('hh', 0, 1)
    result = result + key
    for (k, v) in data.items():
        result = result + String(k, v)
        result = pad32(result)
    return addlen(result)

def StringFileInfo(data):
    if False:
        return 10
    result = struct.pack('hh', 0, 1)
    result = result + nullterm('StringFileInfo')
    result = pad32(result) + StringTable('040904E4', data)
    return addlen(result)

def Var(key, value):
    if False:
        return 10
    result = struct.pack('hh', len(value), 0)
    result = result + nullterm(key)
    result = pad32(result) + value
    return addlen(result)

def VarFileInfo(data):
    if False:
        i = 10
        return i + 15
    result = struct.pack('hh', 0, 1)
    result = result + nullterm('VarFileInfo')
    result = pad32(result)
    for (k, v) in data.items():
        result = result + Var(k, v)
    return addlen(result)

def VS_VERSION_INFO(maj, min, sub, build, sdata, vdata, debug=0, is_dll=1):
    if False:
        for i in range(10):
            print('nop')
    ffi = VS_FIXEDFILEINFO(maj, min, sub, build, debug, is_dll)
    result = struct.pack('hh', len(ffi), 0)
    result = result + nullterm('VS_VERSION_INFO')
    result = pad32(result) + ffi
    result = pad32(result) + StringFileInfo(sdata) + VarFileInfo(vdata)
    return addlen(result)

def stamp(pathname, options):
    if False:
        while True:
            i = 10
    try:
        f = open(pathname, 'a+b')
        f.close()
    except OSError as why:
        print(f'WARNING: File {pathname} could not be opened - {why}')
    ver = options.version
    try:
        bits = [int(i) for i in ver.split('.')]
        (vmaj, vmin, vsub, vbuild) = bits
    except (IndexError, TypeError, ValueError):
        raise ValueError('--version must be a.b.c.d (all integers) - got %r' % ver)
    ifn = options.internal_name
    if not ifn:
        ifn = os.path.basename(pathname)
    ofn = options.original_filename
    if ofn is None:
        ofn = os.path.basename(pathname)
    sdata = {'Comments': options.comments, 'CompanyName': options.company, 'FileDescription': options.description, 'FileVersion': ver, 'InternalName': ifn, 'LegalCopyright': options.copyright, 'LegalTrademarks': options.trademarks, 'OriginalFilename': ofn, 'ProductName': options.product, 'ProductVersion': ver}
    vdata = {'Translation': struct.pack('hh', 1033, 1252)}
    is_dll = options.dll
    if is_dll is None:
        is_dll = os.path.splitext(pathname)[1].lower() in '.dll .pyd'.split()
    is_debug = options.debug
    if is_debug is None:
        is_debug = os.path.splitext(pathname)[0].lower().endswith('_d')
    for (k, v) in list(sdata.items()):
        if v is None:
            sdata[k] = ''
    vs = VS_VERSION_INFO(vmaj, vmin, vsub, vbuild, sdata, vdata, is_debug, is_dll)
    h = BeginUpdateResource(pathname, 0)
    UpdateResource(h, 16, 1, vs)
    EndUpdateResource(h, 0)
    if options.verbose:
        print('Stamped:', pathname)
if __name__ == '__main__':
    parser = optparse.OptionParser('%prog [options] filespec ...', description=__doc__)
    parser.add_option('-q', '--quiet', action='store_false', dest='verbose', default=True, help="don't print status messages to stdout")
    parser.add_option('', '--version', default='0.0.0.0', help='The version number as m.n.s.b')
    parser.add_option('', '--dll', help='Stamp the file as a DLL.  Default is to look at the\n                            file extension for .dll or .pyd.')
    parser.add_option('', '--debug', help='Stamp the file as a debug binary.')
    parser.add_option('', '--product', help='The product name to embed.')
    parser.add_option('', '--company', help='The company name to embed.')
    parser.add_option('', '--trademarks', help='The trademark string to embed.')
    parser.add_option('', '--comments', help='The comments string to embed.')
    parser.add_option('', '--copyright', help='The copyright message string to embed.')
    parser.add_option('', '--description', metavar='DESC', help='The description to embed.')
    parser.add_option('', '--internal-name', metavar='NAME', help='The internal filename to embed. If not specified\n                         the base filename is used.')
    parser.add_option('', '--original-filename', help='The original filename to embed. If not specified\n                            the base filename is used.')
    (options, args) = parser.parse_args()
    if not args:
        parser.error('You must supply a file to stamp.  Use --help for details.')
    for g in args:
        for f in glob.glob(g):
            stamp(f, options)