"""
1. Dump binary data to the following text format:

00000000: 00 00 00 5B 68 65 78 64  75 6D 70 5D 00 00 00 00  ...[hexdump]....
00000010: 00 11 22 33 44 55 66 77  88 99 AA BB CC DD EE FF  .."3DUfw........

It is similar to the one used by:
Scapy
00 00 00 5B 68 65 78 64 75 6D 70 5D 00 00 00 00  ...[hexdump]....
00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF  .."3DUfw........

Far Manager
000000000: 00 00 00 5B 68 65 78 64 ¦ 75 6D 70 5D 00 00 00 00     [hexdump]
000000010: 00 11 22 33 44 55 66 77 ¦ 88 99 AA BB CC DD EE FF   ?"3DUfw\x88\x99ª»ÌÝîÿ


2. Restore binary data from the formats above as well
   as from less exotic strings of raw hex

"""
__version__ = '3.3'
__author__ = 'anatoly techtonik <techtonik@gmail.com>'
__license__ = 'Public Domain'
__history__ = '\n3.3 (2015-01-22)\n * accept input from sys.stdin if "-" is specified\n   for both dump and restore (issue #1)\n * new normalize_py() helper to set sys.stdout to\n   binary mode on Windows\n\n3.2 (2015-07-02)\n * hexdump is now packaged as .zip on all platforms\n   (on Linux created archive was tar.gz)\n * .zip is executable! try `python hexdump-3.2.zip`\n * dump() now accepts configurable separator, patch\n   by Ian Land (PR #3)\n\n3.1 (2014-10-20)\n * implemented workaround against mysterious coding\n   issue with Python 3 (see revision 51302cf)\n * fix Python 3 installs for systems where UTF-8 is\n   not default (Windows), thanks to George Schizas\n   (the problem was caused by reading of README.txt)\n\n3.0 (2014-09-07)\n * remove unused int2byte() helper\n * add dehex(text) helper to convert hex string\n   to binary data\n * add \'size\' argument to dump() helper to specify\n   length of chunks\n\n2.0 (2014-02-02)\n * add --restore option to command line mode to get\n   binary data back from hex dump\n * support saving test output with `--test logfile`\n * restore() from hex strings without spaces\n * restore() now raises TypeError if input data is\n   not string\n * hexdump() and dumpgen() now don\'t return unicode\n   strings in Python 2.x when generator is requested\n\n1.0 (2013-12-30)\n * length of address is reduced from 10 to 8\n * hexdump() got new \'result\' keyword argument, it\n   can be either \'print\', \'generator\' or \'return\'\n * actual dumping logic is now in new dumpgen()\n   generator function\n * new dump(binary) function that takes binary data\n   and returns string like "66 6F 72 6D 61 74"\n * new genchunks(mixed, size) function that chunks\n   both sequences and file like objects\n\n0.5 (2013-06-10)\n * hexdump is now also a command line utility (no\n   restore yet)\n\n0.4 (2013-06-09)\n * fix installation with Python 3 for non English\n   versions of Windows, thanks to George Schizas\n\n0.3 (2013-04-29)\n * fully Python 3 compatible\n\n0.2 (2013-04-28)\n * restore() to recover binary data from a hex dump in\n   native, Far Manager and Scapy text formats (others\n   might work as well)\n * restore() is Python 3 compatible\n\n0.1 (2013-04-28)\n * working hexdump() function for Python 2\n'
import binascii
import sys
PY3K = sys.version_info >= (3, 0)

def normalize_py():
    if False:
        print('Hello World!')
    ' Problem 001 - sys.stdout in Python is by default opened in\n      text mode, and writes to this stdout produce corrupted binary\n      data on Windows\n\n          python -c "import sys; sys.stdout.write(\'_\n_\')" > file\n          python -c "print(repr(open(\'file\', \'rb\').read()))"\n  '
    if sys.platform == 'win32':
        import os, msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

def chunks(seq, size):
    if False:
        while True:
            i = 10
    'Generator that cuts sequence (bytes, memoryview, etc.)\n     into chunks of given size. If `seq` length is not multiply\n     of `size`, the length of the last chunk returned will be\n     less than requested.\n\n     >>> list( chunks([1,2,3,4,5,6,7], 3) )\n     [[1, 2, 3], [4, 5, 6], [7]]\n  '
    (d, m) = divmod(len(seq), size)
    for i in range(d):
        yield seq[i * size:(i + 1) * size]
    if m:
        yield seq[d * size:]

def chunkread(f, size):
    if False:
        while True:
            i = 10
    'Generator that reads from file like object. May return less\n     data than requested on the last read.'
    c = f.read(size)
    while len(c):
        yield c
        c = f.read(size)

def genchunks(mixed, size):
    if False:
        i = 10
        return i + 15
    'Generator to chunk binary sequences or file like objects.\n     The size of the last chunk returned may be less than\n     requested.'
    if hasattr(mixed, 'read'):
        return chunkread(mixed, size)
    else:
        return chunks(mixed, size)

def dehex(hextext):
    if False:
        i = 10
        return i + 15
    '\n  Convert from hex string to binary data stripping\n  whitespaces from `hextext` if necessary.\n  '
    if PY3K:
        return bytes.fromhex(hextext)
    else:
        hextext = ''.join(hextext.split())
        return hextext.decode('hex')

def dump(binary, size=2, sep=' '):
    if False:
        i = 10
        return i + 15
    "\n  Convert binary data (bytes in Python 3 and str in\n  Python 2) to hex string like '00 DE AD BE EF'.\n  `size` argument specifies length of text chunks\n  and `sep` sets chunk separator.\n  "
    hexstr = binascii.hexlify(binary)
    if PY3K:
        hexstr = hexstr.decode('ascii')
    return sep.join(chunks(hexstr.upper(), size))

def dumpgen(data, base_address):
    if False:
        for i in range(10):
            print('nop')
    "\n  Generator that produces strings:\n\n  '00000000: 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  ................'\n  "
    generator = genchunks(data, 16)
    for (addr, d) in enumerate(generator):
        line = '0x%016X: ' % (base_address + addr * 16)
        dumpstr = dump(d)
        line += dumpstr[:8 * 3]
        if len(d) > 8:
            line += ' ' + dumpstr[8 * 3:]
        pad = 2
        if len(d) < 16:
            pad += 3 * (16 - len(d))
        if len(d) <= 8:
            pad += 1
        line += ' ' * pad
        for byte in d:
            if not PY3K:
                byte = ord(byte)
            if 32 <= byte <= 126:
                line += chr(byte)
            else:
                line += '.'
        yield line

def hexdump(data, result='print', base_address=0):
    if False:
        for i in range(10):
            print('nop')
    "\n  Transform binary data to the hex dump text format:\n\n  00000000: 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  ................\n\n    [x] data argument as a binary string\n    [x] data argument as a file like object\n\n  Returns result depending on the `result` argument:\n    'print'     - prints line by line\n    'return'    - returns single string\n    'generator' - returns generator that produces lines\n  "
    if PY3K and type(data) == str:
        raise TypeError('Abstract unicode data (expected bytes sequence)')
    gen = dumpgen(data, base_address)
    if result == 'generator':
        return gen
    elif result == 'return':
        return '\n'.join(gen)
    elif result == 'print':
        for line in gen:
            print(line)
    else:
        raise ValueError('Unknown value of `result` argument')

def restore(dump):
    if False:
        while True:
            i = 10
    '\n  Restore binary data from a hex dump.\n    [x] dump argument as a string\n    [ ] dump argument as a line iterator\n\n  Supported formats:\n    [x] hexdump.hexdump\n    [x] Scapy\n    [x] Far Manager\n  '
    minhexwidth = 2 * 16
    bytehexwidth = 3 * 16 - 1
    result = bytes() if PY3K else ''
    if type(dump) != str:
        raise TypeError('Invalid data for restore')
    text = dump.strip()
    for line in text.split('\n'):
        addrend = line.find(':')
        if 0 < addrend < minhexwidth:
            line = line[addrend + 1:]
        line = line.lstrip()
        if line[2] == ' ':
            sepstart = (2 + 1) * 7 + 2
            sep = line[sepstart:sepstart + 3]
            if sep[:2] == '  ' and sep[2:] != ' ':
                hexdata = line[:bytehexwidth + 1]
            elif sep[2:] == ' ':
                hexdata = line[:sepstart] + line[sepstart + 3:bytehexwidth + 2]
            else:
                hexdata = line[:bytehexwidth]
            line = hexdata
        result += dehex(line)
    return result

def runtest(logfile=None):
    if False:
        while True:
            i = 10
    'Run hexdump tests. Requires hexfile.bin to be in the same\n     directory as hexdump.py itself'

    class TeeOutput(object):

        def __init__(self, stream1, stream2):
            if False:
                i = 10
                return i + 15
            self.outputs = [stream1, stream2]

        def write(self, data):
            if False:
                while True:
                    i = 10
            for stream in self.outputs:
                if PY3K:
                    if 'b' in stream.mode:
                        data = data.encode('utf-8')
                stream.write(data)
                stream.flush()

        def tell(self):
            if False:
                while True:
                    i = 10
            raise IOError

        def flush(self):
            if False:
                while True:
                    i = 10
            for stream in self.outputs:
                stream.flush()
    if logfile:
        openlog = open(logfile, 'wb')
        savedstd = (sys.stderr, sys.stdout)
        sys.stderr = TeeOutput(sys.stderr, openlog)
        sys.stdout = TeeOutput(sys.stdout, openlog)

    def echo(msg, linefeed=True):
        if False:
            return 10
        sys.stdout.write(msg)
        if linefeed:
            sys.stdout.write('\n')
    expected = '00000000: 00 00 00 5B 68 65 78 64  75 6D 70 5D 00 00 00 00  ...[hexdump]....\n00000010: 00 11 22 33 44 55 66 77  88 99 0A BB CC DD EE FF  .."3DUfw........'
    import pkgutil
    bin = pkgutil.get_data('hexdump', 'data/hexfile.bin')
    hexdump(b'zzzz' * 12)
    hexdump(b'o' * 17)
    hexdump(b'p' * 24)
    hexdump(b'q' * 26)
    hexdump(b'line\nfeed\r\ntest')
    hexdump(b'\x00\x00\x00[hexdump]\x00\x00\x00\x00\x00\x11"3DUfw\x88\x99\n\xbb\xcc\xdd\xee\xff')
    print('---')
    hexdump(bin)
    print('return output')
    hexout = hexdump(bin, result='return')
    assert hexout == expected, "returned hex didn't match"
    print('return generator')
    hexgen = hexdump(bin, result='generator')
    assert next(hexgen) == expected.split('\n')[0], "hex generator 1 didn't match"
    assert next(hexgen) == expected.split('\n')[1], "hex generator 2 didn't match"
    bindata = restore('\n00000000: 00 00 00 5B 68 65 78 64  75 6D 70 5D 00 00 00 00  ...[hexdump]....\n00000010: 00 11 22 33 44 55 66 77  88 99 0A BB CC DD EE FF  .."3DUfw........\n')
    echo('restore check ', linefeed=False)
    assert bin == bindata, 'restore check failed'
    echo('passed')
    far = '\n000000000: 00 00 00 5B 68 65 78 64 ¦ 75 6D 70 5D 00 00 00 00     [hexdump]\n000000010: 00 11 22 33 44 55 66 77 ¦ 88 99 0A BB CC DD EE FF   ?"3DUfw\x88\x99ª»ÌÝîÿ\n'
    echo('restore far format ', linefeed=False)
    assert bin == restore(far), 'far format check failed'
    echo('passed')
    scapy = '00 00 00 5B 68 65 78 64 75 6D 70 5D 00 00 00 00  ...[hexdump]....\n00 11 22 33 44 55 66 77 88 99 0A BB CC DD EE FF  .."3DUfw........\n'
    echo('restore scapy format ', linefeed=False)
    assert bin == restore(scapy), 'scapy format check failed'
    echo('passed')
    if not PY3K:
        assert restore('5B68657864756D705D') == '[hexdump]', 'no space check failed'
        assert dump('\\¡«\x1e', sep='').lower() == '5ca1ab1e'
    else:
        assert restore('5B68657864756D705D') == b'[hexdump]', 'no space check failed'
        assert dump(b'\\\xa1\xab\x1e', sep='').lower() == '5ca1ab1e'
    print('---[test file hexdumping]---')
    import os
    import tempfile
    hexfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        hexfile.write(bin)
        hexfile.close()
        hexdump(open(hexfile.name, 'rb'))
    finally:
        os.remove(hexfile.name)
    if logfile:
        (sys.stderr, sys.stdout) = savedstd
        openlog.close()

def main():
    if False:
        i = 10
        return i + 15
    from optparse import OptionParser
    parser = OptionParser(usage='\n  %prog [binfile|-]\n  %prog -r hexfile\n  %prog --test [logfile]', version=__version__)
    parser.add_option('-r', '--restore', action='store_true', help='restore binary from hex dump')
    parser.add_option('--test', action='store_true', help='run hexdump sanity checks')
    (options, args) = parser.parse_args()
    if options.test:
        if args:
            runtest(logfile=args[0])
        else:
            runtest()
    elif not args or len(args) > 1:
        parser.print_help()
        sys.exit(-1)
    elif not options.restore:
        if args[0] == '-':
            if not PY3K:
                hexdump(sys.stdin)
            else:
                hexdump(sys.stdin.buffer)
        else:
            hexdump(open(args[0], 'rb'))
    else:
        if args[0] == '-':
            instream = sys.stdin
        elif PY3K:
            instream = open(args[0])
        else:
            instream = open(args[0], 'rb')
        if PY3K:
            sys.stdout.buffer.write(restore(instream.read()))
        else:
            normalize_py()
            sys.stdout.write(restore(instream.read()))
if __name__ == '__main__':
    main()