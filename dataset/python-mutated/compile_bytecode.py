"""Compiles a single .py to a .pyc and writes it to stdout."""
import importlib.util
import marshal
import re
import sys
MAGIC = importlib.util.MAGIC_NUMBER
ENCODING_PATTERN = '^[ \t\x0b]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)'

def is_comment_only(line):
    if False:
        print('Hello World!')
    return re.match('[ \t\x0b]*#.*', line) is not None

def _write32(f, w):
    if False:
        for i in range(10):
            print('nop')
    f.write(bytearray([w >> 0 & 255, w >> 8 & 255, w >> 16 & 255, w >> 24 & 255]))

def write_pyc(f, codeobject, source_size=0, timestamp=0):
    if False:
        while True:
            i = 10
    f.write(MAGIC)
    f.write(b'\r\n\x00\x00')
    _write32(f, timestamp)
    _write32(f, source_size)
    f.write(marshal.dumps(codeobject))

def compile_to_pyc(data_file, filename, output, mode):
    if False:
        print('Hello World!')
    'Compile the source code to byte code.'
    with open(data_file, encoding='utf-8') as fi:
        src = fi.read()
    compile_src_to_pyc(src, filename, output, mode)

def strip_encoding(src):
    if False:
        print('Hello World!')
    'Strip encoding from a src string assumed to be read from a file.'
    if '\n' not in src:
        return src
    (l1, rest) = src.split('\n', 1)
    if re.match(ENCODING_PATTERN, l1.rstrip()):
        return '#\n' + rest
    elif '\n' not in rest:
        return src
    (l2, rest) = rest.split('\n', 1)
    if is_comment_only(l1) and re.match(ENCODING_PATTERN, l2.rstrip()):
        return '#\n#\n' + rest
    return src

def compile_src_to_pyc(src, filename, output, mode):
    if False:
        i = 10
        return i + 15
    'Compile a string of source code.'
    try:
        codeobject = compile(src, filename, mode)
    except Exception as err:
        output.write(b'\x01')
        output.write(str(err).encode('utf-8'))
    else:
        output.write(b'\x00')
        write_pyc(output, codeobject)

def main():
    if False:
        return 10
    if len(sys.argv) != 4:
        sys.exit(1)
    output = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout
    compile_to_pyc(data_file=sys.argv[1], filename=sys.argv[2], output=output, mode=sys.argv[3])
if __name__ == '__main__':
    main()