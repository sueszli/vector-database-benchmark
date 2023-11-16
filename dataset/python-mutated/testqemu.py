from __future__ import print_function
import os
import struct
import logging
from sys import stdout
from pdb import pm
try:
    stdout = stdout.buffer
except AttributeError:
    pass
from miasm.analysis.sandbox import Sandbox_Linux_x86_32
from miasm.jitter.jitload import log_func
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE
from miasm.core.locationdb import LocationDB

def parse_fmt(s):
    if False:
        print('Hello World!')
    fmt = s[:] + '\x00'
    out = []
    i = 0
    while i < len(fmt):
        c = fmt[i:i + 1]
        if c != '%':
            i += 1
            continue
        if fmt[i + 1:i + 2] == '%':
            i += 2
            continue
        j = 0
        i += 1
        while fmt[i + j:i + j + 1] in '0123456789$.-':
            j += 1
        if fmt[i + j:i + j + 1] in ['l']:
            j += 1
        if fmt[i + j:i + j + 1] == 'h':
            x = fmt[i + j:i + j + 2]
        else:
            x = fmt[i + j:i + j + 1]
        i += j
        out.append(x)
    return out
nb_tests = 1

def xxx___printf_chk(jitter):
    if False:
        while True:
            i = 10
    'Tiny implementation of printf_chk'
    global nb_tests
    (ret_ad, args) = jitter.func_args_systemv(['out', 'format'])
    if args.out != 1:
        raise RuntimeError('Not implemented')
    fmt = jitter.get_c_str(args.format)
    fmt = fmt.replace('llx', 'lx')
    fmt = fmt.replace('%016lx', '%016z')
    fmt_a = parse_fmt(fmt)
    esp = jitter.cpu.ESP
    args = []
    i = 0
    for x in fmt_a:
        a = jitter.vm.get_u32(esp + 8 + 4 * i)
        if x == 's':
            a = jitter.get_c_str(a)
        elif x in ('x', 'X', 'd'):
            pass
        elif x.lower() in ('f', 'l'):
            a2 = jitter.vm.get_u32(esp + 8 + 4 * (i + 1))
            a = struct.unpack('d', struct.pack('Q', a2 << 32 | a))[0]
            i += 1
        elif x.lower() == 'z':
            a2 = jitter.vm.get_u32(esp + 8 + 4 * (i + 1))
            a = a2 << 32 | a
            i += 1
        else:
            raise RuntimeError('Not implemented format')
        args.append(a)
        i += 1
    fmt = fmt.replace('%016z', '%016lx')
    output = fmt % tuple(args)
    output = output.replace('nan', '-nan')
    if '\n' not in output:
        raise RuntimeError('Format must end with a \\n')
    line = next(expected)
    if output != line:
        print('Expected:', line)
        print('Obtained:', output)
        raise RuntimeError('Bad semantic')
    stdout.write(b'[%d] %s' % (nb_tests, output.encode('utf8')))
    nb_tests += 1
    jitter.func_ret_systemv(ret_ad, 0)

def xxx_puts(jitter):
    if False:
        while True:
            i = 10
    '\n    #include <stdio.h>\n    int puts(const char *s);\n\n    writes the string s and a trailing newline to stdout.\n    '
    (ret_addr, args) = jitter.func_args_systemv(['target'])
    output = jitter.get_c_str(args.target)
    line = next(expected)
    if output != line.rstrip():
        print('Expected:', line)
        print('Obtained:', output)
        raise RuntimeError('Bad semantic')
    return jitter.func_ret_systemv(ret_addr, 1)
parser = Sandbox_Linux_x86_32.parser(description='ELF sandboxer')
parser.add_argument('filename', help='ELF Filename')
parser.add_argument('funcname', help="Targeted function's name")
parser.add_argument('expected', help='Expected output')
options = parser.parse_args()
expected = open(options.expected)
loc_db = LocationDB()
sb = Sandbox_Linux_x86_32(loc_db, options.filename, options, globals())
try:
    addr = sb.elf.getsectionbyname('.symtab')[options.funcname].value
except AttributeError:
    raise RuntimeError('The target binary must have a symtab section')
log_func.setLevel(logging.ERROR)
sb.jitter.cpu.set_segm_base(8, 2147418112)
sb.jitter.cpu.GS = 8
sb.jitter.vm.add_memory_page(2147418112 + 20, PAGE_READ | PAGE_WRITE, b'AAAA')
sb.run(addr)
assert sb.jitter.running is False