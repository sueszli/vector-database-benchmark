"""
This script reads in the given CMSIS device include file (eg stm32f405xx.h),
extracts relevant peripheral constants, and creates qstrs, mpz's and constants
for the stm module.
"""
from __future__ import print_function
import argparse
import re
import platform
if platform.python_version_tuple()[0] == '2':

    def convert_bytes_to_str(b):
        if False:
            return 10
        return b
elif platform.python_version_tuple()[0] == '3':

    def convert_bytes_to_str(b):
        if False:
            return 10
        try:
            return str(b, 'utf8')
        except ValueError:
            return ''.join((chr(l) for l in b if l <= 126))

def re_match_first(regexs, line):
    if False:
        i = 10
        return i + 15
    for (name, regex) in regexs:
        match = re.match(regex, line)
        if match:
            return (name, match)
    return (None, None)

class LexerError(Exception):

    def __init__(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.line = line

class Lexer:
    re_io_reg = '__IO uint(?P<bits>8|16|32)_t +(?P<reg>[A-Z0-9]+)'
    re_comment = '(?P<comment>[A-Za-z0-9 \\-/_()&:\\[\\]]+)'
    re_addr_offset = 'Address offset: (?P<offset>0x[0-9A-Z]{2,3})'
    regexs = (('#define hex', re.compile('#define +(?P<id>[A-Z0-9_]+) +\\(?(\\(uint32_t\\))?(?P<hex>0x[0-9A-F]+)U?L?\\)?($| */\\*)')), ('#define X', re.compile('#define +(?P<id>[A-Z0-9_]+) +(?P<id2>[A-Z0-9_]+)($| +/\\*)')), ('#define X+hex', re.compile('#define +(?P<id>[A-Za-z0-9_]+) +\\(?(?P<id2>[A-Z0-9_]+) \\+ (?P<hex>0x[0-9A-F]+)U?L?\\)?($| +/\\*)')), ('#define typedef', re.compile('#define +(?P<id>[A-Z0-9_]+(ext)?) +\\(\\([A-Za-z0-9_]+_(Global)?TypeDef \\*\\) (?P<id2>[A-Za-z0-9_]+)\\)($| +/\\*)')), ('typedef struct', re.compile('typedef struct$')), ('{', re.compile('{$')), ('}', re.compile('}$')), ('} _t', re.compile('} *([A-Za-z0-9_]+)_t;$')), ('} TypeDef', re.compile('} *(?P<id>[A-Z][A-Za-z0-9_]*)_(?P<global>([A-Za-z0-9_]+)?)TypeDef;$')), ('IO reg', re.compile(re_io_reg + ' *; */\\*!< *' + re_comment + ',? +' + re_addr_offset + ' *\\*/')), ('IO reg array', re.compile(re_io_reg + '\\[(?P<array>[2-8])\\] *; */\\*!< *' + re_comment + ',? +' + re_addr_offset + '-(0x[0-9A-Z]{2,3}) *\\*/')))

    def __init__(self, filename):
        if False:
            print('Hello World!')
        self.file = open(filename, 'rb')
        self.line_number = 0

    def next_match(self, strictly_next=False):
        if False:
            for i in range(10):
                print('nop')
        while True:
            line = self.file.readline()
            line = convert_bytes_to_str(line)
            self.line_number += 1
            if len(line) == 0:
                return ('EOF', None)
            match = re_match_first(Lexer.regexs, line.strip())
            if strictly_next or match[0] is not None:
                return match

    def must_match(self, kind):
        if False:
            i = 10
            return i + 15
        match = self.next_match(strictly_next=True)
        if match[0] != kind:
            raise LexerError(self.line_number)
        return match

def parse_file(filename):
    if False:
        for i in range(10):
            print('nop')
    lexer = Lexer(filename)
    reg_defs = {}
    consts = {}
    periphs = []
    while True:
        m = lexer.next_match()
        if m[0] == 'EOF':
            break
        elif m[0] == '#define hex':
            d = m[1].groupdict()
            consts[d['id']] = int(d['hex'], base=16)
        elif m[0] == '#define X':
            d = m[1].groupdict()
            if d['id2'] in consts:
                consts[d['id']] = consts[d['id2']]
        elif m[0] == '#define X+hex':
            d = m[1].groupdict()
            if d['id2'] in consts:
                consts[d['id']] = consts[d['id2']] + int(d['hex'], base=16)
        elif m[0] == '#define typedef':
            d = m[1].groupdict()
            if d['id2'] in consts:
                periphs.append((d['id'], consts[d['id2']]))
        elif m[0] == 'typedef struct':
            lexer.must_match('{')
            m = lexer.next_match()
            regs = []
            while m[0] in ('IO reg', 'IO reg array'):
                d = m[1].groupdict()
                reg = d['reg']
                offset = int(d['offset'], base=16)
                bits = int(d['bits'])
                comment = d['comment']
                if m[0] == 'IO reg':
                    regs.append((reg, offset, bits, comment))
                else:
                    for i in range(int(d['array'])):
                        regs.append((reg + str(i), offset + i * bits // 8, bits, comment))
                m = lexer.next_match()
            if m[0] in ('}', '} _t'):
                pass
            elif m[0] == '} TypeDef':
                d = m[1].groupdict()
                n = d['id']
                g = d['global']
                if n not in reg_defs or not g:
                    reg_defs[n] = regs
            else:
                raise LexerError(lexer.line_number)
    return (periphs, reg_defs)

def print_int_obj(val, needed_mpzs):
    if False:
        return 10
    if -1073741824 <= val < 1073741824:
        print('MP_ROM_INT(%#x)' % val, end='')
    else:
        print('MP_ROM_PTR(&mpz_%08x)' % val, end='')
        needed_mpzs.add(val)

def print_periph(periph_name, periph_val, needed_mpzs):
    if False:
        i = 10
        return i + 15
    qstr = periph_name.upper()
    print('{ MP_ROM_QSTR(MP_QSTR_%s), ' % qstr, end='')
    print_int_obj(periph_val, needed_mpzs)
    print(' },')

def print_regs(reg_name, reg_defs, needed_mpzs):
    if False:
        return 10
    reg_name = reg_name.upper()
    for r in reg_defs:
        qstr = reg_name + '_' + r[0]
        print('{ MP_ROM_QSTR(MP_QSTR_%s), ' % qstr, end='')
        print_int_obj(r[1], needed_mpzs)
        print(' }, // %s-bits, %s' % (r[2], r[3]))

def print_regs_as_submodules(reg_name, reg_defs, modules):
    if False:
        while True:
            i = 10
    mod_name_lower = reg_name.lower() + '_'
    mod_name_upper = mod_name_lower.upper()
    modules.append((mod_name_lower, mod_name_upper))
    print('\nSTATIC const mp_rom_map_elem_t stm_%s_globals_table[] = {\n    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_%s) },\n' % (mod_name_lower, mod_name_upper))
    for r in reg_defs:
        print('    { MP_ROM_QSTR(MP_QSTR_%s), MP_ROM_INT(%#x) }, // %s-bits, %s' % (r[0], r[1], r[2], r[3]))
    print('};\n\nSTATIC MP_DEFINE_CONST_DICT(stm_%s_globals, stm_%s_globals_table);\n\nconst mp_obj_module_t stm_%s_obj = {\n    .base = { &mp_type_module },\n    .name = MP_QSTR_%s,\n    .globals = (mp_obj_dict_t*)&stm_%s_globals,\n};\n' % (mod_name_lower, mod_name_lower, mod_name_lower, mod_name_upper, mod_name_lower))

def main():
    if False:
        i = 10
        return i + 15
    cmd_parser = argparse.ArgumentParser(description='Extract ST constants from a C header file.')
    cmd_parser.add_argument('file', nargs=1, help='input file')
    cmd_parser.add_argument('--mpz', dest='mpz_filename', default='build/stmconst_mpz.h', help='the destination file of the generated mpz header')
    args = cmd_parser.parse_args()
    (periphs, reg_defs) = parse_file(args.file[0])
    if 'GPIO' in reg_defs and 'stm32f4' in args.file[0]:
        reg_defs['GPIO'].append(['BSRRL', 24, 16, 'legacy register'])
        reg_defs['GPIO'].append(['BSRRH', 26, 16, 'legacy register'])
    needed_mpzs = set()
    print('// Automatically generated from %s by make-stmconst.py' % args.file[0])
    print('')
    for (periph_name, periph_val) in periphs:
        print_periph(periph_name, periph_val, needed_mpzs)
    for reg in ('ADC', 'FDCAN', 'CRC', 'DAC', 'DBGMCU', 'DMA_Stream', 'DMA', 'EXTI', 'FLASH', 'GPIO', 'SYSCFG', 'I2C', 'IWDG', 'PWR', 'RCC', 'RTC', 'SPI', 'TIM', 'USART', 'WWDG', 'RNG', 'IPCC'):
        if reg in reg_defs:
            print_regs(reg, reg_defs[reg], needed_mpzs)
    print('')
    with open(args.mpz_filename, 'wt') as mpz_file:
        for mpz in sorted(needed_mpzs):
            assert 0 <= mpz <= 4294967295
            print('STATIC const mp_obj_int_t mpz_%08x = {{&mp_type_int}, {.neg=0, .fixed_dig=1, .alloc=2, .len=2, .dig=(uint16_t*)(const uint16_t[]){%#x, %#x}}};' % (mpz, mpz & 65535, mpz >> 16 & 65535), file=mpz_file)
if __name__ == '__main__':
    main()