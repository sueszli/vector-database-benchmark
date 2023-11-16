from __future__ import print_function
import platform
if platform.python_version_tuple()[0] == '2':
    from binascii import hexlify as hexlify_py2
    str_cons = lambda val, enc=None: str(val)
    bytes_cons = lambda val, enc=None: bytearray(val)
    is_str_type = lambda o: isinstance(o, str)
    is_bytes_type = lambda o: type(o) is bytearray
    is_int_type = lambda o: isinstance(o, int) or isinstance(o, long)

    def hexlify_to_str(b):
        if False:
            i = 10
            return i + 15
        x = hexlify_py2(b)
        return ':'.join((x[i:i + 2] for i in range(0, len(x), 2)))
else:
    from binascii import hexlify
    str_cons = str
    bytes_cons = bytes
    is_str_type = lambda o: isinstance(o, str)
    is_bytes_type = lambda o: isinstance(o, bytes)
    is_int_type = lambda o: isinstance(o, int)

    def hexlify_to_str(b):
        if False:
            print('Hello World!')
        return str(hexlify(b, ':'), 'ascii')
import sys
import struct
sys.path.append(sys.path[0] + '/../py')
import makeqstrdata as qstrutil
PERSISTENT_STR_INTERN_THRESHOLD = 25

class MPYReadError(Exception):

    def __init__(self, filename, msg):
        if False:
            i = 10
            return i + 15
        self.filename = filename
        self.msg = msg

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s: %s' % (self.filename, self.msg)

class FreezeError(Exception):

    def __init__(self, rawcode, msg):
        if False:
            print('Hello World!')
        self.rawcode = rawcode
        self.msg = msg

    def __str__(self):
        if False:
            print('Hello World!')
        return 'error while freezing %s: %s' % (self.rawcode.source_file, self.msg)

class Config:
    MPY_VERSION = 6
    MPY_SUB_VERSION = 2
    MICROPY_LONGINT_IMPL_NONE = 0
    MICROPY_LONGINT_IMPL_LONGLONG = 1
    MICROPY_LONGINT_IMPL_MPZ = 2
config = Config()
MP_CODE_BYTECODE = 2
MP_CODE_NATIVE_PY = 3
MP_CODE_NATIVE_VIPER = 4
MP_CODE_NATIVE_ASM = 5
MP_NATIVE_ARCH_NONE = 0
MP_NATIVE_ARCH_X86 = 1
MP_NATIVE_ARCH_X64 = 2
MP_NATIVE_ARCH_ARMV6 = 3
MP_NATIVE_ARCH_ARMV6M = 4
MP_NATIVE_ARCH_ARMV7M = 5
MP_NATIVE_ARCH_ARMV7EM = 6
MP_NATIVE_ARCH_ARMV7EMSP = 7
MP_NATIVE_ARCH_ARMV7EMDP = 8
MP_NATIVE_ARCH_XTENSA = 9
MP_NATIVE_ARCH_XTENSAWIN = 10
MP_PERSISTENT_OBJ_FUN_TABLE = 0
MP_PERSISTENT_OBJ_NONE = 1
MP_PERSISTENT_OBJ_FALSE = 2
MP_PERSISTENT_OBJ_TRUE = 3
MP_PERSISTENT_OBJ_ELLIPSIS = 4
MP_PERSISTENT_OBJ_STR = 5
MP_PERSISTENT_OBJ_BYTES = 6
MP_PERSISTENT_OBJ_INT = 7
MP_PERSISTENT_OBJ_FLOAT = 8
MP_PERSISTENT_OBJ_COMPLEX = 9
MP_PERSISTENT_OBJ_TUPLE = 10
MP_SCOPE_FLAG_VIPERRELOC = 16
MP_SCOPE_FLAG_VIPERRODATA = 32
MP_SCOPE_FLAG_VIPERBSS = 64
MP_BC_MASK_EXTRA_BYTE = 158
MP_BC_FORMAT_BYTE = 0
MP_BC_FORMAT_QSTR = 1
MP_BC_FORMAT_VAR_UINT = 2
MP_BC_FORMAT_OFFSET = 3
mp_unary_op_method_name = ('__pos__', '__neg__', '__invert__', '<not>')
mp_binary_op_method_name = ('__lt__', '__gt__', '__eq__', '__le__', '__ge__', '__ne__', '<in>', '<is>', '<exception match>', '__ior__', '__ixor__', '__iand__', '__ilshift__', '__irshift__', '__iadd__', '__isub__', '__imul__', '__imatmul__', '__ifloordiv__', '__itruediv__', '__imod__', '__ipow__', '__or__', '__xor__', '__and__', '__lshift__', '__rshift__', '__add__', '__sub__', '__mul__', '__matmul__', '__floordiv__', '__truediv__', '__mod__', '__pow__')

class Opcode:
    MP_BC_BASE_RESERVED = 0
    MP_BC_BASE_QSTR_O = 16
    MP_BC_BASE_VINT_E = 32
    MP_BC_BASE_VINT_O = 48
    MP_BC_BASE_JUMP_E = 64
    MP_BC_BASE_BYTE_O = 80
    MP_BC_BASE_BYTE_E = 96
    MP_BC_LOAD_CONST_SMALL_INT_MULTI = 112
    MP_BC_LOAD_FAST_MULTI = 176
    MP_BC_STORE_FAST_MULTI = 192
    MP_BC_UNARY_OP_MULTI = 208
    MP_BC_BINARY_OP_MULTI = 215
    MP_BC_LOAD_CONST_SMALL_INT_MULTI_NUM = 64
    MP_BC_LOAD_CONST_SMALL_INT_MULTI_EXCESS = 16
    MP_BC_LOAD_FAST_MULTI_NUM = 16
    MP_BC_STORE_FAST_MULTI_NUM = 16
    MP_BC_UNARY_OP_MULTI_NUM = 4
    MP_BC_BINARY_OP_MULTI_NUM = 35
    MP_BC_LOAD_CONST_FALSE = MP_BC_BASE_BYTE_O + 0
    MP_BC_LOAD_CONST_NONE = MP_BC_BASE_BYTE_O + 1
    MP_BC_LOAD_CONST_TRUE = MP_BC_BASE_BYTE_O + 2
    MP_BC_LOAD_CONST_SMALL_INT = MP_BC_BASE_VINT_E + 2
    MP_BC_LOAD_CONST_STRING = MP_BC_BASE_QSTR_O + 0
    MP_BC_LOAD_CONST_OBJ = MP_BC_BASE_VINT_E + 3
    MP_BC_LOAD_NULL = MP_BC_BASE_BYTE_O + 3
    MP_BC_LOAD_FAST_N = MP_BC_BASE_VINT_E + 4
    MP_BC_LOAD_DEREF = MP_BC_BASE_VINT_E + 5
    MP_BC_LOAD_NAME = MP_BC_BASE_QSTR_O + 1
    MP_BC_LOAD_GLOBAL = MP_BC_BASE_QSTR_O + 2
    MP_BC_LOAD_ATTR = MP_BC_BASE_QSTR_O + 3
    MP_BC_LOAD_METHOD = MP_BC_BASE_QSTR_O + 4
    MP_BC_LOAD_SUPER_METHOD = MP_BC_BASE_QSTR_O + 5
    MP_BC_LOAD_BUILD_CLASS = MP_BC_BASE_BYTE_O + 4
    MP_BC_LOAD_SUBSCR = MP_BC_BASE_BYTE_O + 5
    MP_BC_STORE_FAST_N = MP_BC_BASE_VINT_E + 6
    MP_BC_STORE_DEREF = MP_BC_BASE_VINT_E + 7
    MP_BC_STORE_NAME = MP_BC_BASE_QSTR_O + 6
    MP_BC_STORE_GLOBAL = MP_BC_BASE_QSTR_O + 7
    MP_BC_STORE_ATTR = MP_BC_BASE_QSTR_O + 8
    MP_BC_STORE_SUBSCR = MP_BC_BASE_BYTE_O + 6
    MP_BC_DELETE_FAST = MP_BC_BASE_VINT_E + 8
    MP_BC_DELETE_DEREF = MP_BC_BASE_VINT_E + 9
    MP_BC_DELETE_NAME = MP_BC_BASE_QSTR_O + 9
    MP_BC_DELETE_GLOBAL = MP_BC_BASE_QSTR_O + 10
    MP_BC_DUP_TOP = MP_BC_BASE_BYTE_O + 7
    MP_BC_DUP_TOP_TWO = MP_BC_BASE_BYTE_O + 8
    MP_BC_POP_TOP = MP_BC_BASE_BYTE_O + 9
    MP_BC_ROT_TWO = MP_BC_BASE_BYTE_O + 10
    MP_BC_ROT_THREE = MP_BC_BASE_BYTE_O + 11
    MP_BC_UNWIND_JUMP = MP_BC_BASE_JUMP_E + 0
    MP_BC_JUMP = MP_BC_BASE_JUMP_E + 2
    MP_BC_POP_JUMP_IF_TRUE = MP_BC_BASE_JUMP_E + 3
    MP_BC_POP_JUMP_IF_FALSE = MP_BC_BASE_JUMP_E + 4
    MP_BC_JUMP_IF_TRUE_OR_POP = MP_BC_BASE_JUMP_E + 5
    MP_BC_JUMP_IF_FALSE_OR_POP = MP_BC_BASE_JUMP_E + 6
    MP_BC_SETUP_WITH = MP_BC_BASE_JUMP_E + 7
    MP_BC_SETUP_EXCEPT = MP_BC_BASE_JUMP_E + 8
    MP_BC_SETUP_FINALLY = MP_BC_BASE_JUMP_E + 9
    MP_BC_POP_EXCEPT_JUMP = MP_BC_BASE_JUMP_E + 10
    MP_BC_FOR_ITER = MP_BC_BASE_JUMP_E + 11
    MP_BC_WITH_CLEANUP = MP_BC_BASE_BYTE_O + 12
    MP_BC_END_FINALLY = MP_BC_BASE_BYTE_O + 13
    MP_BC_GET_ITER = MP_BC_BASE_BYTE_O + 14
    MP_BC_GET_ITER_STACK = MP_BC_BASE_BYTE_O + 15
    MP_BC_BUILD_TUPLE = MP_BC_BASE_VINT_E + 10
    MP_BC_BUILD_LIST = MP_BC_BASE_VINT_E + 11
    MP_BC_BUILD_MAP = MP_BC_BASE_VINT_E + 12
    MP_BC_STORE_MAP = MP_BC_BASE_BYTE_E + 2
    MP_BC_BUILD_SET = MP_BC_BASE_VINT_E + 13
    MP_BC_BUILD_SLICE = MP_BC_BASE_VINT_E + 14
    MP_BC_STORE_COMP = MP_BC_BASE_VINT_E + 15
    MP_BC_UNPACK_SEQUENCE = MP_BC_BASE_VINT_O + 0
    MP_BC_UNPACK_EX = MP_BC_BASE_VINT_O + 1
    MP_BC_RETURN_VALUE = MP_BC_BASE_BYTE_E + 3
    MP_BC_RAISE_LAST = MP_BC_BASE_BYTE_E + 4
    MP_BC_RAISE_OBJ = MP_BC_BASE_BYTE_E + 5
    MP_BC_RAISE_FROM = MP_BC_BASE_BYTE_E + 6
    MP_BC_YIELD_VALUE = MP_BC_BASE_BYTE_E + 7
    MP_BC_YIELD_FROM = MP_BC_BASE_BYTE_E + 8
    MP_BC_MAKE_FUNCTION = MP_BC_BASE_VINT_O + 2
    MP_BC_MAKE_FUNCTION_DEFARGS = MP_BC_BASE_VINT_O + 3
    MP_BC_MAKE_CLOSURE = MP_BC_BASE_VINT_E + 0
    MP_BC_MAKE_CLOSURE_DEFARGS = MP_BC_BASE_VINT_E + 1
    MP_BC_CALL_FUNCTION = MP_BC_BASE_VINT_O + 4
    MP_BC_CALL_FUNCTION_VAR_KW = MP_BC_BASE_VINT_O + 5
    MP_BC_CALL_METHOD = MP_BC_BASE_VINT_O + 6
    MP_BC_CALL_METHOD_VAR_KW = MP_BC_BASE_VINT_O + 7
    MP_BC_IMPORT_NAME = MP_BC_BASE_QSTR_O + 11
    MP_BC_IMPORT_FROM = MP_BC_BASE_QSTR_O + 12
    MP_BC_IMPORT_STAR = MP_BC_BASE_BYTE_E + 9
    ALL_OFFSET_SIGNED = (MP_BC_UNWIND_JUMP, MP_BC_JUMP, MP_BC_POP_JUMP_IF_TRUE, MP_BC_POP_JUMP_IF_FALSE)
    mapping = ['unknown' for _ in range(256)]
    for op_name in list(locals()):
        if op_name.startswith('MP_BC_'):
            mapping[locals()[op_name]] = op_name[len('MP_BC_'):]
    for i in range(MP_BC_LOAD_CONST_SMALL_INT_MULTI_NUM):
        name = 'LOAD_CONST_SMALL_INT %d' % (i - MP_BC_LOAD_CONST_SMALL_INT_MULTI_EXCESS)
        mapping[MP_BC_LOAD_CONST_SMALL_INT_MULTI + i] = name
    for i in range(MP_BC_LOAD_FAST_MULTI_NUM):
        mapping[MP_BC_LOAD_FAST_MULTI + i] = 'LOAD_FAST %d' % i
    for i in range(MP_BC_STORE_FAST_MULTI_NUM):
        mapping[MP_BC_STORE_FAST_MULTI + i] = 'STORE_FAST %d' % i
    for i in range(MP_BC_UNARY_OP_MULTI_NUM):
        mapping[MP_BC_UNARY_OP_MULTI + i] = 'UNARY_OP %d %s' % (i, mp_unary_op_method_name[i])
    for i in range(MP_BC_BINARY_OP_MULTI_NUM):
        mapping[MP_BC_BINARY_OP_MULTI + i] = 'BINARY_OP %d %s' % (i, mp_binary_op_method_name[i])

    def __init__(self, offset, fmt, opcode_byte, arg, extra_arg):
        if False:
            return 10
        self.offset = offset
        self.fmt = fmt
        self.opcode_byte = opcode_byte
        self.arg = arg
        self.extra_arg = extra_arg

def mp_small_int_fits(i):
    if False:
        while True:
            i = 10
    return -8192 <= i <= 8191

def mp_encode_uint(val, signed=False):
    if False:
        for i in range(10):
            print('nop')
    encoded = bytearray([val & 127])
    val >>= 7
    while val != 0 and val != -1:
        encoded.insert(0, 128 | val & 127)
        val >>= 7
    if signed:
        if val == -1 and encoded[0] & 64 == 0:
            encoded.insert(0, 255)
        elif val == 0 and encoded[0] & 64 != 0:
            encoded.insert(0, 128)
    return encoded

def mp_opcode_decode(bytecode, ip):
    if False:
        for i in range(10):
            print('nop')
    opcode = bytecode[ip]
    ip_start = ip
    f = 932 >> 2 * (opcode >> 4) & 3
    ip += 1
    arg = None
    extra_arg = None
    if f in (MP_BC_FORMAT_QSTR, MP_BC_FORMAT_VAR_UINT):
        arg = bytecode[ip] & 127
        if opcode == Opcode.MP_BC_LOAD_CONST_SMALL_INT and arg & 64 != 0:
            arg |= -1 << 7
        while bytecode[ip] & 128 != 0:
            ip += 1
            arg = arg << 7 | bytecode[ip] & 127
        ip += 1
    elif f == MP_BC_FORMAT_OFFSET:
        if bytecode[ip] & 128 == 0:
            arg = bytecode[ip]
            ip += 1
            if opcode in Opcode.ALL_OFFSET_SIGNED:
                arg -= 64
        else:
            arg = bytecode[ip] & 127 | bytecode[ip + 1] << 7
            ip += 2
            if opcode in Opcode.ALL_OFFSET_SIGNED:
                arg -= 16384
    if opcode & MP_BC_MASK_EXTRA_BYTE == 0:
        extra_arg = bytecode[ip]
        ip += 1
    return (f, ip - ip_start, arg, extra_arg)

def mp_opcode_encode(opcode):
    if False:
        i = 10
        return i + 15
    overflow = False
    encoded = bytearray([opcode.opcode_byte])
    if opcode.fmt in (MP_BC_FORMAT_QSTR, MP_BC_FORMAT_VAR_UINT):
        signed = opcode.opcode_byte == Opcode.MP_BC_LOAD_CONST_SMALL_INT
        encoded.extend(mp_encode_uint(opcode.arg, signed))
    elif opcode.fmt == MP_BC_FORMAT_OFFSET:
        is_signed = opcode.opcode_byte in Opcode.ALL_OFFSET_SIGNED
        bytecode_offset = opcode.target.offset - opcode.offset - 2
        if is_signed and -64 <= bytecode_offset <= 63 or (not is_signed and bytecode_offset <= 127):
            if is_signed:
                bytecode_offset += 64
            overflow = not 0 <= bytecode_offset <= 127
            encoded.append(bytecode_offset & 127)
        else:
            bytecode_offset -= 1
            if is_signed:
                bytecode_offset += 16384
            overflow = not 0 <= bytecode_offset <= 32767
            encoded.append(128 | bytecode_offset & 127)
            encoded.append(bytecode_offset >> 7 & 255)
    if opcode.extra_arg is not None:
        encoded.append(opcode.extra_arg)
    return (overflow, encoded)

def read_prelude_sig(read_byte):
    if False:
        while True:
            i = 10
    z = read_byte()
    S = z >> 3 & 15
    E = z >> 2 & 1
    F = 0
    A = z & 3
    K = 0
    D = 0
    n = 0
    while z & 128:
        z = read_byte()
        S |= (z & 48) << 2 * n
        E |= (z & 2) << n
        F |= (z & 64) >> 6 << n
        A |= (z & 4) << n
        K |= (z & 8) >> 3 << n
        D |= (z & 1) << n
        n += 1
    S += 1
    return (S, E, F, A, K, D)

def read_prelude_size(read_byte):
    if False:
        while True:
            i = 10
    I = 0
    C = 0
    n = 0
    while True:
        z = read_byte()
        I |= (z & 126) >> 1 << 6 * n
        C |= (z & 1) << n
        if not z & 128:
            break
        n += 1
    return (I, C)

def encode_prelude_size(I, C):
    if False:
        for i in range(10):
            print('nop')
    encoded = bytearray()
    while True:
        z = (I & 63) << 1 | C & 1
        C >>= 1
        I >>= 6
        if C | I:
            z |= 128
        encoded.append(z)
        if not C | I:
            return encoded

def extract_prelude(bytecode, ip):
    if False:
        return 10

    def local_read_byte():
        if False:
            while True:
                i = 10
        b = bytecode[ip_ref[0]]
        ip_ref[0] += 1
        return b
    ip_ref = [ip]
    (n_state, n_exc_stack, scope_flags, n_pos_args, n_kwonly_args, n_def_pos_args) = read_prelude_sig(local_read_byte)
    offset_prelude_size = ip_ref[0]
    (n_info, n_cell) = read_prelude_size(local_read_byte)
    offset_source_info = ip_ref[0]
    args = []
    for arg_num in range(1 + n_pos_args + n_kwonly_args):
        value = 0
        while True:
            b = local_read_byte()
            value = value << 7 | b & 127
            if b & 128 == 0:
                break
        args.append(value)
    offset_line_info = ip_ref[0]
    offset_closure_info = offset_source_info + n_info
    offset_opcodes = offset_source_info + n_info + n_cell
    return (offset_prelude_size, offset_source_info, offset_line_info, offset_closure_info, offset_opcodes, (n_state, n_exc_stack, scope_flags, n_pos_args, n_kwonly_args, n_def_pos_args), (n_info, n_cell), args)

class QStrType:

    def __init__(self, str):
        if False:
            for i in range(10):
                print('nop')
        self.str = str
        self.qstr_esc = qstrutil.qstr_escape(self.str)
        self.qstr_id = 'MP_QSTR_' + self.qstr_esc

class GlobalQStrList:

    def __init__(self):
        if False:
            return 10
        self.qstrs = [None]
        for n in qstrutil.static_qstr_list:
            self.qstrs.append(QStrType(n))

    def add(self, s):
        if False:
            return 10
        q = QStrType(s)
        self.qstrs.append(q)
        return q

    def get_by_index(self, i):
        if False:
            i = 10
            return i + 15
        return self.qstrs[i]

    def find_by_str(self, s):
        if False:
            return 10
        for q in self.qstrs:
            if q is not None and q.str == s:
                return q
        return None

class MPFunTable:

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'mp_fun_table'

class CompiledModule:

    def __init__(self, mpy_source_file, mpy_segments, header, qstr_table, obj_table, raw_code, qstr_table_file_offset, obj_table_file_offset, raw_code_file_offset, escaped_name):
        if False:
            i = 10
            return i + 15
        self.mpy_source_file = mpy_source_file
        self.mpy_segments = mpy_segments
        self.source_file = qstr_table[0]
        self.header = header
        self.qstr_table = qstr_table
        self.obj_table = obj_table
        self.raw_code = raw_code
        self.qstr_table_file_offset = qstr_table_file_offset
        self.obj_table_file_offset = obj_table_file_offset
        self.raw_code_file_offset = raw_code_file_offset
        self.escaped_name = escaped_name

    def hexdump(self):
        if False:
            print('Hello World!')
        with open(self.mpy_source_file, 'rb') as f:
            WIDTH = 16
            COL_OFF = '\x1b[0m'
            COL_TABLE = (('', ''), ('\x1b[0;31m', '\x1b[0;91m'), ('\x1b[0;32m', '\x1b[0;92m'), ('\x1b[0;34m', '\x1b[0;94m'))
            cur_col = ''
            cur_col_index = 0
            offset = 0
            segment_index = 0
            while True:
                data = bytes_cons(f.read(WIDTH))
                if not data:
                    break
                line_hex = cur_col
                line_chr = cur_col
                line_comment = ''
                for i in range(len(data)):
                    while segment_index < len(self.mpy_segments):
                        if offset + i == self.mpy_segments[segment_index].start:
                            cur_col = COL_TABLE[self.mpy_segments[segment_index].kind][cur_col_index]
                            cur_col_index = 1 - cur_col_index
                            line_hex += cur_col
                            line_chr += cur_col
                            line_comment += ' %s%s%s' % (cur_col, self.mpy_segments[segment_index].name, COL_OFF)
                        if offset + i == self.mpy_segments[segment_index].end:
                            cur_col = ''
                            line_hex += COL_OFF
                            line_chr += COL_OFF
                            segment_index += 1
                        else:
                            break
                    if i % 2 == 0:
                        line_hex += ' '
                    line_hex += '%02x' % data[i]
                    if 32 <= data[i] <= 126:
                        line_chr += '%s' % chr(data[i])
                    else:
                        line_chr += '.'
                if cur_col:
                    line_hex += COL_OFF
                    line_chr += COL_OFF
                pad = ' ' * ((WIDTH - len(data)) * 5 // 2)
                print('%08x:%s%s  %s %s' % (offset, line_hex, pad, line_chr, line_comment))
                offset += WIDTH

    def disassemble(self):
        if False:
            return 10
        print('mpy_source_file:', self.mpy_source_file)
        print('source_file:', self.source_file.str)
        print('header:', hexlify_to_str(self.header))
        print('qstr_table[%u]:' % len(self.qstr_table))
        for q in self.qstr_table:
            print('    %s' % q.str)
        print('obj_table:', self.obj_table)
        self.raw_code.disassemble()

    def freeze(self, compiled_module_index):
        if False:
            while True:
                i = 10
        print()
        print('/' * 80)
        print('// frozen module %s' % self.escaped_name)
        print('// - original source file: %s' % self.mpy_source_file)
        print('// - frozen file name: %s' % self.source_file.str)
        print('// - .mpy header: %s' % ':'.join(('%02x' % b for b in self.header)))
        print()
        self.raw_code.freeze()
        print()
        self.freeze_constants()
        print()
        print('static const mp_frozen_module_t frozen_module_%s = {' % self.escaped_name)
        print('    .constants = {')
        if len(self.qstr_table):
            print('        .qstr_table = (qstr_short_t *)&const_qstr_table_data_%s,' % self.escaped_name)
        else:
            print('        .qstr_table = NULL,')
        if len(self.obj_table):
            print('        .obj_table = (mp_obj_t *)&const_obj_table_data_%s,' % self.escaped_name)
        else:
            print('        .obj_table = NULL,')
        print('    },')
        print('    .rc = &raw_code_%s,' % self.raw_code.escaped_name)
        print('};')

    def freeze_constant_obj(self, obj_name, obj):
        if False:
            print('Hello World!')
        global const_str_content, const_int_content, const_obj_content
        if isinstance(obj, MPFunTable):
            return '&mp_fun_table'
        elif obj is None:
            return 'MP_ROM_NONE'
        elif obj is False:
            return 'MP_ROM_FALSE'
        elif obj is True:
            return 'MP_ROM_TRUE'
        elif obj is Ellipsis:
            return 'MP_ROM_PTR(&mp_const_ellipsis_obj)'
        elif is_str_type(obj) or is_bytes_type(obj):
            if len(obj) == 0:
                if is_str_type(obj):
                    return 'MP_ROM_QSTR(MP_QSTR_)'
                else:
                    return 'MP_ROM_PTR(&mp_const_empty_bytes_obj)'
            if is_str_type(obj):
                q = global_qstrs.find_by_str(obj)
                if q:
                    return 'MP_ROM_QSTR(%s)' % q.qstr_id
                obj = bytes_cons(obj, 'utf8')
                obj_type = 'mp_type_str'
            else:
                obj_type = 'mp_type_bytes'
            print('static const mp_obj_str_t %s = {{&%s}, %u, %u, (const byte*)"%s"};' % (obj_name, obj_type, qstrutil.compute_hash(obj, config.MICROPY_QSTR_BYTES_IN_HASH), len(obj), ''.join(('\\x%02x' % b for b in obj))))
            const_str_content += len(obj)
            const_obj_content += 4 * 4
            return 'MP_ROM_PTR(&%s)' % obj_name
        elif is_int_type(obj):
            if mp_small_int_fits(obj):
                return 'MP_ROM_INT(%d)' % obj
            elif config.MICROPY_LONGINT_IMPL == config.MICROPY_LONGINT_IMPL_NONE:
                raise FreezeError(self, 'target does not support long int')
            elif config.MICROPY_LONGINT_IMPL == config.MICROPY_LONGINT_IMPL_LONGLONG:
                raise FreezeError(self, 'freezing int to long-long is not implemented')
            elif config.MICROPY_LONGINT_IMPL == config.MICROPY_LONGINT_IMPL_MPZ:
                neg = 0
                if obj < 0:
                    obj = -obj
                    neg = 1
                bits_per_dig = config.MPZ_DIG_SIZE
                digs = []
                z = obj
                while z:
                    digs.append(z & (1 << bits_per_dig) - 1)
                    z >>= bits_per_dig
                ndigs = len(digs)
                digs = ','.join(('%#x' % d for d in digs))
                print('static const mp_obj_int_t %s = {{&mp_type_int}, {.neg=%u, .fixed_dig=1, .alloc=%u, .len=%u, .dig=(uint%u_t*)(const uint%u_t[]){%s}}};' % (obj_name, neg, ndigs, ndigs, bits_per_dig, bits_per_dig, digs))
                const_int_content += (digs.count(',') + 1) * bits_per_dig // 8
                const_obj_content += 4 * 4
                return 'MP_ROM_PTR(&%s)' % obj_name
        elif isinstance(obj, float):
            macro_name = '%s_macro' % obj_name
            print('#if MICROPY_OBJ_REPR == MICROPY_OBJ_REPR_A || MICROPY_OBJ_REPR == MICROPY_OBJ_REPR_B')
            print('static const mp_obj_float_t %s = {{&mp_type_float}, (mp_float_t)%.16g};' % (obj_name, obj))
            print('#define %s MP_ROM_PTR(&%s)' % (macro_name, obj_name))
            print('#elif MICROPY_OBJ_REPR == MICROPY_OBJ_REPR_C')
            n = struct.unpack('<I', struct.pack('<f', obj))[0]
            n = (n & ~3 | 2) + 2155872256
            print('#define %s ((mp_rom_obj_t)(0x%08x))' % (macro_name, n))
            print('#elif MICROPY_OBJ_REPR == MICROPY_OBJ_REPR_D')
            n = struct.unpack('<Q', struct.pack('<d', obj))[0]
            n += 9224497936761618432
            print('#define %s ((mp_rom_obj_t)(0x%016x))' % (macro_name, n))
            print('#endif')
            const_obj_content += 3 * 4
            return macro_name
        elif isinstance(obj, complex):
            print('static const mp_obj_complex_t %s = {{&mp_type_complex}, (mp_float_t)%.16g, (mp_float_t)%.16g};' % (obj_name, obj.real, obj.imag))
            return 'MP_ROM_PTR(&%s)' % obj_name
        elif type(obj) is tuple:
            if len(obj) == 0:
                return 'MP_ROM_PTR(&mp_const_empty_tuple_obj)'
            else:
                obj_refs = []
                for (i, sub_obj) in enumerate(obj):
                    sub_obj_name = '%s_%u' % (obj_name, i)
                    obj_refs.append(self.freeze_constant_obj(sub_obj_name, sub_obj))
                print('static const mp_rom_obj_tuple_t %s = {{&mp_type_tuple}, %d, {' % (obj_name, len(obj)))
                for ref in obj_refs:
                    print('    %s,' % ref)
                print('}};')
                return 'MP_ROM_PTR(&%s)' % obj_name
        else:
            raise FreezeError(self, 'freezing of object %r is not implemented' % (obj,))

    def freeze_constants(self):
        if False:
            i = 10
            return i + 15
        if len(self.qstr_table):
            print('static const qstr_short_t const_qstr_table_data_%s[%u] = {' % (self.escaped_name, len(self.qstr_table)))
            for q in self.qstr_table:
                print('    %s,' % q.qstr_id)
            print('};')
        if not len(self.obj_table):
            return
        print()
        print('// constants')
        obj_refs = []
        for (i, obj) in enumerate(self.obj_table):
            obj_name = 'const_obj_%s_%u' % (self.escaped_name, i)
            obj_refs.append(self.freeze_constant_obj(obj_name, obj))
        print()
        print('// constant table')
        print('static const mp_rom_obj_t const_obj_table_data_%s[%u] = {' % (self.escaped_name, len(self.obj_table)))
        for ref in obj_refs:
            print('    %s,' % ref)
        print('};')
        global const_table_ptr_content
        const_table_ptr_content += len(self.obj_table)

class RawCode(object):
    escaped_names = set()
    code_kind_str = {MP_CODE_BYTECODE: 'MP_CODE_BYTECODE', MP_CODE_NATIVE_PY: 'MP_CODE_NATIVE_PY', MP_CODE_NATIVE_VIPER: 'MP_CODE_NATIVE_VIPER', MP_CODE_NATIVE_ASM: 'MP_CODE_NATIVE_ASM'}

    def __init__(self, parent_name, qstr_table, fun_data, prelude_offset, code_kind):
        if False:
            while True:
                i = 10
        self.qstr_table = qstr_table
        self.fun_data = fun_data
        self.prelude_offset = prelude_offset
        self.code_kind = code_kind
        if code_kind in (MP_CODE_BYTECODE, MP_CODE_NATIVE_PY):
            (self.offset_prelude_size, self.offset_source_info, self.offset_line_info, self.offset_closure_info, self.offset_opcodes, self.prelude_signature, self.prelude_size, self.names) = extract_prelude(self.fun_data, prelude_offset)
            self.scope_flags = self.prelude_signature[2]
            self.n_pos_args = self.prelude_signature[3]
            self.simple_name = self.qstr_table[self.names[0]]
        else:
            self.simple_name = self.qstr_table[0]
        escaped_name = parent_name + '_' + self.simple_name.qstr_esc
        i = 2
        unique_escaped_name = escaped_name
        while unique_escaped_name in self.escaped_names:
            unique_escaped_name = escaped_name + str(i)
            i += 1
        self.escaped_names.add(unique_escaped_name)
        self.escaped_name = unique_escaped_name

    def disassemble_children(self):
        if False:
            i = 10
            return i + 15
        print('  children:', [rc.simple_name.str for rc in self.children])
        for rc in self.children:
            rc.disassemble()

    def freeze_children(self, prelude_ptr=None):
        if False:
            return 10
        if len(self.children):
            for rc in self.children:
                print('// child of %s' % self.escaped_name)
                rc.freeze()
                print()
            print('static const mp_raw_code_t *const children_%s[] = {' % self.escaped_name)
            for rc in self.children:
                print('    &raw_code_%s,' % rc.escaped_name)
            if prelude_ptr:
                print('    (void *)%s,' % prelude_ptr)
            print('};')
            print()

    def freeze_raw_code(self, prelude_ptr=None, type_sig=0):
        if False:
            while True:
                i = 10
        print('static const mp_raw_code_t raw_code_%s = {' % self.escaped_name)
        print('    .kind = %s,' % RawCode.code_kind_str[self.code_kind])
        print('    .scope_flags = 0x%02x,' % self.scope_flags)
        print('    .n_pos_args = %u,' % self.n_pos_args)
        print('    .fun_data = fun_data_%s,' % self.escaped_name)
        print('    #if MICROPY_PERSISTENT_CODE_SAVE || MICROPY_DEBUG_PRINTERS')
        print('    .fun_data_len = %u,' % len(self.fun_data))
        print('    #endif')
        if len(self.children):
            print('    .children = (void *)&children_%s,' % self.escaped_name)
        elif prelude_ptr:
            print('    .children = (void *)%s,' % prelude_ptr)
        else:
            print('    .children = NULL,')
        print('    #if MICROPY_PERSISTENT_CODE_SAVE')
        print('    .n_children = %u,' % len(self.children))
        if self.code_kind == MP_CODE_BYTECODE:
            print('    #if MICROPY_PY_SYS_SETTRACE')
            print('    .prelude = {')
            print('        .n_state = %u,' % self.prelude_signature[0])
            print('        .n_exc_stack = %u,' % self.prelude_signature[1])
            print('        .scope_flags = %u,' % self.prelude_signature[2])
            print('        .n_pos_args = %u,' % self.prelude_signature[3])
            print('        .n_kwonly_args = %u,' % self.prelude_signature[4])
            print('        .n_def_pos_args = %u,' % self.prelude_signature[5])
            print('        .qstr_block_name_idx = %u,' % self.names[0])
            print('        .line_info = fun_data_%s + %u,' % (self.escaped_name, self.offset_line_info))
            print('        .line_info_top = fun_data_%s + %u,' % (self.escaped_name, self.offset_closure_info))
            print('        .opcodes = fun_data_%s + %u,' % (self.escaped_name, self.offset_opcodes))
            print('    },')
            print('    .line_of_definition = %u,' % 0)
            print('    #endif')
        print('    #if MICROPY_EMIT_MACHINE_CODE')
        print('    .prelude_offset = %u,' % self.prelude_offset)
        print('    #endif')
        print('    #endif')
        print('    #if MICROPY_EMIT_MACHINE_CODE')
        print('    .type_sig = %u,' % type_sig)
        print('    #endif')
        print('};')
        global raw_code_count, raw_code_content
        raw_code_count += 1
        raw_code_content += 4 * 4

class RawCodeBytecode(RawCode):

    def __init__(self, parent_name, qstr_table, obj_table, fun_data):
        if False:
            for i in range(10):
                print('nop')
        self.obj_table = obj_table
        super(RawCodeBytecode, self).__init__(parent_name, qstr_table, fun_data, 0, MP_CODE_BYTECODE)

    def disassemble(self):
        if False:
            print('Hello World!')
        bc = self.fun_data
        print('simple_name:', self.simple_name.str)
        print('  raw bytecode:', len(bc), hexlify_to_str(bc))
        print('  prelude:', self.prelude_signature)
        print('  args:', [self.qstr_table[i].str for i in self.names[1:]])
        print('  line info:', hexlify_to_str(bc[self.offset_line_info:self.offset_opcodes]))
        ip = self.offset_opcodes
        while ip < len(bc):
            (fmt, sz, arg, _) = mp_opcode_decode(bc, ip)
            if bc[ip] == Opcode.MP_BC_LOAD_CONST_OBJ:
                arg = repr(self.obj_table[arg])
            if fmt == MP_BC_FORMAT_QSTR:
                arg = self.qstr_table[arg].str
            elif fmt in (MP_BC_FORMAT_VAR_UINT, MP_BC_FORMAT_OFFSET):
                pass
            else:
                arg = ''
            print('  %-11s %s %s' % (hexlify_to_str(bc[ip:ip + sz]), Opcode.mapping[bc[ip]], arg))
            ip += sz
        self.disassemble_children()

    def freeze(self):
        if False:
            return 10
        bc = self.fun_data
        print('// frozen bytecode for file %s, scope %s' % (self.qstr_table[0].str, self.escaped_name))
        print('static const byte fun_data_%s[%u] = {' % (self.escaped_name, len(bc)))
        print('    ', end='')
        for b in bc[:self.offset_source_info]:
            print('0x%02x,' % b, end='')
        print(' // prelude')
        print('    ', end='')
        for b in bc[self.offset_source_info:self.offset_line_info]:
            print('0x%02x,' % b, end='')
        print(' // names: %s' % ', '.join((self.qstr_table[i].str for i in self.names)))
        print('    ', end='')
        for b in bc[self.offset_line_info:self.offset_opcodes]:
            print('0x%02x,' % b, end='')
        print(' // code info')
        ip = self.offset_opcodes
        while ip < len(bc):
            (fmt, sz, arg, _) = mp_opcode_decode(bc, ip)
            opcode_name = Opcode.mapping[bc[ip]]
            if fmt == MP_BC_FORMAT_QSTR:
                opcode_name += ' ' + repr(self.qstr_table[arg].str)
            elif fmt in (MP_BC_FORMAT_VAR_UINT, MP_BC_FORMAT_OFFSET):
                opcode_name += ' %u' % arg
            print('    %s, // %s' % (','.join(('0x%02x' % b for b in bc[ip:ip + sz])), opcode_name))
            ip += sz
        print('};')
        self.freeze_children()
        self.freeze_raw_code()
        global bc_content
        bc_content += len(bc)

class RawCodeNative(RawCode):

    def __init__(self, parent_name, qstr_table, kind, fun_data, prelude_offset, scope_flags, n_pos_args, type_sig):
        if False:
            while True:
                i = 10
        super(RawCodeNative, self).__init__(parent_name, qstr_table, fun_data, prelude_offset, kind)
        if kind in (MP_CODE_NATIVE_VIPER, MP_CODE_NATIVE_ASM):
            self.scope_flags = scope_flags
            self.n_pos_args = n_pos_args
        self.type_sig = type_sig
        if config.native_arch in (MP_NATIVE_ARCH_X86, MP_NATIVE_ARCH_X64, MP_NATIVE_ARCH_XTENSA, MP_NATIVE_ARCH_XTENSAWIN):
            self.fun_data_attributes = '__attribute__((section(".text,\\"ax\\",@progbits # ")))'
        else:
            self.fun_data_attributes = '__attribute__((section(".text,\\"ax\\",%progbits @ ")))'
        if config.native_arch in (MP_NATIVE_ARCH_ARMV6, MP_NATIVE_ARCH_XTENSA, MP_NATIVE_ARCH_XTENSAWIN):
            self.fun_data_attributes += ' __attribute__ ((aligned (4)))'
        elif MP_NATIVE_ARCH_ARMV6M <= config.native_arch <= MP_NATIVE_ARCH_ARMV7EMDP:
            self.fun_data_attributes += ' __attribute__ ((aligned (2)))'

    def disassemble(self):
        if False:
            while True:
                i = 10
        fun_data = self.fun_data
        print('simple_name:', self.simple_name.str)
        print('  raw data:', len(fun_data), hexlify_to_str(fun_data[:32]), '...' if len(fun_data) > 32 else '')
        if self.code_kind != MP_CODE_NATIVE_PY:
            return
        print('  prelude:', self.prelude_signature)
        print('  args:', [self.qstr_table[i].str for i in self.names[1:]])
        print('  line info:', fun_data[self.offset_line_info:self.offset_opcodes])
        ip = 0
        while ip < self.prelude_offset:
            sz = 16
            print(' ', hexlify_to_str(fun_data[ip:min(ip + sz, self.prelude_offset)]))
            ip += sz
        self.disassemble_children()

    def freeze(self):
        if False:
            return 10
        if self.scope_flags & ~15:
            raise FreezeError('unable to freeze code with relocations')
        print()
        print('// frozen native code for file %s, scope %s' % (self.qstr_table[0].str, self.escaped_name))
        print('static const byte fun_data_%s[%u] %s = {' % (self.escaped_name, len(self.fun_data), self.fun_data_attributes))
        i_top = len(self.fun_data)
        i = 0
        while i < i_top:
            i16 = min(i + 16, i_top)
            print('   ', end='')
            for ii in range(i, i16):
                print(' 0x%02x,' % self.fun_data[ii], end='')
            print()
            i = i16
        print('};')
        prelude_ptr = None
        if self.code_kind == MP_CODE_NATIVE_PY:
            prelude_ptr = 'fun_data_%s_prelude_macro' % self.escaped_name
            print('#if MICROPY_EMIT_NATIVE_PRELUDE_SEPARATE_FROM_MACHINE_CODE')
            n = len(self.fun_data) - self.prelude_offset
            print('static const byte fun_data_%s_prelude[%u] = {' % (self.escaped_name, n), end='')
            for i in range(n):
                print(' 0x%02x,' % self.fun_data[self.prelude_offset + i], end='')
            print('};')
            print('#define %s &fun_data_%s_prelude[0]' % (prelude_ptr, self.escaped_name))
            print('#else')
            print('#define %s &fun_data_%s[%u]' % (prelude_ptr, self.escaped_name, self.prelude_offset))
            print('#endif')
        self.freeze_children(prelude_ptr)
        self.freeze_raw_code(prelude_ptr, self.type_sig)

class MPYSegment:
    META = 0
    QSTR = 1
    OBJ = 2
    CODE = 3

    def __init__(self, kind, name, start, end):
        if False:
            print('Hello World!')
        self.kind = kind
        self.name = name
        self.start = start
        self.end = end

class MPYReader:

    def __init__(self, filename, fileobj):
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename
        self.fileobj = fileobj

    def tell(self):
        if False:
            print('Hello World!')
        return self.fileobj.tell()

    def read_byte(self):
        if False:
            return 10
        return bytes_cons(self.fileobj.read(1))[0]

    def read_bytes(self, n):
        if False:
            i = 10
            return i + 15
        return bytes_cons(self.fileobj.read(n))

    def read_uint(self):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        while True:
            b = self.read_byte()
            i = i << 7 | b & 127
            if b & 128 == 0:
                break
        return i

def read_qstr(reader, segments):
    if False:
        return 10
    start_pos = reader.tell()
    ln = reader.read_uint()
    if ln & 1:
        q = global_qstrs.get_by_index(ln >> 1)
        segments.append(MPYSegment(MPYSegment.META, q.str, start_pos, start_pos))
        return q
    ln >>= 1
    start_pos = reader.tell()
    data = str_cons(reader.read_bytes(ln), 'utf8')
    reader.read_byte()
    segments.append(MPYSegment(MPYSegment.QSTR, data, start_pos, reader.tell()))
    return global_qstrs.add(data)

def read_obj(reader, segments):
    if False:
        i = 10
        return i + 15
    obj_type = reader.read_byte()
    if obj_type == MP_PERSISTENT_OBJ_FUN_TABLE:
        return MPFunTable()
    elif obj_type == MP_PERSISTENT_OBJ_NONE:
        return None
    elif obj_type == MP_PERSISTENT_OBJ_FALSE:
        return False
    elif obj_type == MP_PERSISTENT_OBJ_TRUE:
        return True
    elif obj_type == MP_PERSISTENT_OBJ_ELLIPSIS:
        return Ellipsis
    elif obj_type == MP_PERSISTENT_OBJ_TUPLE:
        ln = reader.read_uint()
        return tuple((read_obj(reader, segments) for _ in range(ln)))
    else:
        ln = reader.read_uint()
        start_pos = reader.tell()
        buf = reader.read_bytes(ln)
        if obj_type in (MP_PERSISTENT_OBJ_STR, MP_PERSISTENT_OBJ_BYTES):
            reader.read_byte()
        if obj_type == MP_PERSISTENT_OBJ_STR:
            obj = str_cons(buf, 'utf8')
            if len(obj) < PERSISTENT_STR_INTERN_THRESHOLD:
                if not global_qstrs.find_by_str(obj):
                    global_qstrs.add(obj)
        elif obj_type == MP_PERSISTENT_OBJ_BYTES:
            obj = buf
        elif obj_type == MP_PERSISTENT_OBJ_INT:
            obj = int(str_cons(buf, 'ascii'), 10)
        elif obj_type == MP_PERSISTENT_OBJ_FLOAT:
            obj = float(str_cons(buf, 'ascii'))
        elif obj_type == MP_PERSISTENT_OBJ_COMPLEX:
            obj = complex(str_cons(buf, 'ascii'))
        else:
            raise MPYReadError(reader.filename, 'corrupt .mpy file')
        segments.append(MPYSegment(MPYSegment.OBJ, obj, start_pos, reader.tell()))
        return obj

def read_raw_code(reader, parent_name, qstr_table, obj_table, segments):
    if False:
        print('Hello World!')
    kind_len = reader.read_uint()
    kind = (kind_len & 3) + MP_CODE_BYTECODE
    has_children = kind_len >> 2 & 1
    fun_data_len = kind_len >> 3
    file_offset = reader.tell()
    fun_data = reader.read_bytes(fun_data_len)
    segments_len = len(segments)
    if kind == MP_CODE_BYTECODE:
        rc = RawCodeBytecode(parent_name, qstr_table, obj_table, fun_data)
    else:
        native_scope_flags = 0
        native_n_pos_args = 0
        native_type_sig = 0
        if kind == MP_CODE_NATIVE_PY:
            prelude_offset = reader.read_uint()
        else:
            prelude_offset = 0
            native_scope_flags = reader.read_uint()
            if kind == MP_CODE_NATIVE_VIPER:
                if native_scope_flags & MP_SCOPE_FLAG_VIPERRODATA:
                    rodata_size = reader.read_uint()
                if native_scope_flags & MP_SCOPE_FLAG_VIPERBSS:
                    reader.read_uint()
                if native_scope_flags & MP_SCOPE_FLAG_VIPERRODATA:
                    reader.read_bytes(rodata_size)
                if native_scope_flags & MP_SCOPE_FLAG_VIPERRELOC:
                    while True:
                        op = reader.read_byte()
                        if op == 255:
                            break
                        if op & 1:
                            reader.read_uint()
                        op >>= 1
                        if op <= 5 and op & 1:
                            reader.read_uint()
            else:
                assert kind == MP_CODE_NATIVE_ASM
                native_n_pos_args = reader.read_uint()
                native_type_sig = reader.read_uint()
        rc = RawCodeNative(parent_name, qstr_table, kind, fun_data, prelude_offset, native_scope_flags, native_n_pos_args, native_type_sig)
    segments.insert(segments_len, MPYSegment(MPYSegment.CODE, rc.simple_name.str, file_offset, file_offset + fun_data_len))
    rc.children = []
    if has_children:
        if not rc.escaped_name.endswith('_lt_module_gt_'):
            parent_name = rc.escaped_name
        n_children = reader.read_uint()
        for _ in range(n_children):
            rc.children.append(read_raw_code(reader, parent_name, qstr_table, obj_table, segments))
    return rc

def read_mpy(filename):
    if False:
        while True:
            i = 10
    with open(filename, 'rb') as fileobj:
        reader = MPYReader(filename, fileobj)
        segments = []
        header = reader.read_bytes(4)
        if header[0] != ord('M'):
            raise MPYReadError(filename, 'not a valid .mpy file')
        if header[1] != config.MPY_VERSION:
            raise MPYReadError(filename, 'incompatible .mpy version')
        feature_byte = header[2]
        mpy_native_arch = feature_byte >> 2
        if mpy_native_arch != MP_NATIVE_ARCH_NONE:
            mpy_sub_version = feature_byte & 3
            if mpy_sub_version != config.MPY_SUB_VERSION:
                raise MPYReadError(filename, 'incompatible .mpy sub-version')
            if config.native_arch == MP_NATIVE_ARCH_NONE:
                config.native_arch = mpy_native_arch
            elif config.native_arch != mpy_native_arch:
                raise MPYReadError(filename, 'native architecture mismatch')
        config.mp_small_int_bits = header[3]
        n_qstr = reader.read_uint()
        n_obj = reader.read_uint()
        qstr_table_file_offset = reader.tell()
        qstr_table = []
        for i in range(n_qstr):
            qstr_table.append(read_qstr(reader, segments))
        obj_table_file_offset = reader.tell()
        obj_table = []
        for i in range(n_obj):
            obj_table.append(read_obj(reader, segments))
        cm_escaped_name = qstr_table[0].str.replace('/', '_')[:-3]
        raw_code_file_offset = reader.tell()
        raw_code = read_raw_code(reader, cm_escaped_name, qstr_table, obj_table, segments)
    return CompiledModule(filename, segments, header, qstr_table, obj_table, raw_code, qstr_table_file_offset, obj_table_file_offset, raw_code_file_offset, cm_escaped_name)

def hexdump_mpy(compiled_modules):
    if False:
        while True:
            i = 10
    for cm in compiled_modules:
        cm.hexdump()

def disassemble_mpy(compiled_modules):
    if False:
        while True:
            i = 10
    for cm in compiled_modules:
        cm.disassemble()

def freeze_mpy(firmware_qstr_idents, compiled_modules):
    if False:
        print('Hello World!')
    new = {}
    for q in global_qstrs.qstrs:
        if q is None or q.qstr_esc in firmware_qstr_idents or q.qstr_esc in new:
            continue
        new[q.qstr_esc] = (len(new), q.qstr_esc, q.str, bytes_cons(q.str, 'utf8'))
    new = sorted(new.values(), key=lambda x: x[2])
    print('#include "py/mpconfig.h"')
    print('#include "py/objint.h"')
    print('#include "py/objstr.h"')
    print('#include "py/emitglue.h"')
    print('#include "py/nativeglue.h"')
    print()
    print('#if MICROPY_LONGINT_IMPL != %u' % config.MICROPY_LONGINT_IMPL)
    print('#error "incompatible MICROPY_LONGINT_IMPL"')
    print('#endif')
    print()
    if config.MICROPY_LONGINT_IMPL == config.MICROPY_LONGINT_IMPL_MPZ:
        print('#if MPZ_DIG_SIZE != %u' % config.MPZ_DIG_SIZE)
        print('#error "incompatible MPZ_DIG_SIZE"')
        print('#endif')
        print()
    print('#if MICROPY_PY_BUILTINS_FLOAT')
    print('typedef struct _mp_obj_float_t {')
    print('    mp_obj_base_t base;')
    print('    mp_float_t value;')
    print('} mp_obj_float_t;')
    print('#endif')
    print()
    print('#if MICROPY_PY_BUILTINS_COMPLEX')
    print('typedef struct _mp_obj_complex_t {')
    print('    mp_obj_base_t base;')
    print('    mp_float_t real;')
    print('    mp_float_t imag;')
    print('} mp_obj_complex_t;')
    print('#endif')
    print()
    if len(new) > 0:
        print('enum {')
        for i in range(len(new)):
            if i == 0:
                print('    MP_QSTR_%s = MP_QSTRnumber_of,' % new[i][1])
            else:
                print('    MP_QSTR_%s,' % new[i][1])
        print('};')
    qstr_pool_alloc = min(len(new), 10)
    global bc_content, const_str_content, const_int_content, const_obj_content, const_table_qstr_content, const_table_ptr_content, raw_code_count, raw_code_content
    qstr_content = 0
    bc_content = 0
    const_str_content = 0
    const_int_content = 0
    const_obj_content = 0
    const_table_qstr_content = 0
    const_table_ptr_content = 0
    raw_code_count = 0
    raw_code_content = 0
    print()
    print('const qstr_hash_t mp_qstr_frozen_const_hashes[] = {')
    qstr_size = {'metadata': 0, 'data': 0}
    for (_, _, _, qbytes) in new:
        qhash = qstrutil.compute_hash(qbytes, config.MICROPY_QSTR_BYTES_IN_HASH)
        print('    %d,' % qhash)
    print('};')
    print()
    print('const qstr_len_t mp_qstr_frozen_const_lengths[] = {')
    for (_, _, _, qbytes) in new:
        print('    %d,' % len(qbytes))
        qstr_size['metadata'] += config.MICROPY_QSTR_BYTES_IN_LEN + config.MICROPY_QSTR_BYTES_IN_HASH
        qstr_size['data'] += len(qbytes)
    print('};')
    print()
    print('extern const qstr_pool_t mp_qstr_const_pool;')
    print('const qstr_pool_t mp_qstr_frozen_const_pool = {')
    print('    &mp_qstr_const_pool, // previous pool')
    print('    MP_QSTRnumber_of, // previous pool size')
    print('    true, // is_sorted')
    print('    %u, // allocated entries' % qstr_pool_alloc)
    print('    %u, // used entries' % len(new))
    print('    (qstr_hash_t *)mp_qstr_frozen_const_hashes,')
    print('    (qstr_len_t *)mp_qstr_frozen_const_lengths,')
    print('    {')
    for (_, _, qstr, qbytes) in new:
        print('        "%s",' % qstrutil.escape_bytes(qstr, qbytes))
        qstr_content += config.MICROPY_QSTR_BYTES_IN_LEN + config.MICROPY_QSTR_BYTES_IN_HASH + len(qbytes) + 1
    print('    },')
    print('};')
    for (idx, cm) in enumerate(compiled_modules):
        cm.freeze(idx)
    print()
    print('/' * 80)
    print('// collection of all frozen modules')
    print()
    print('const char mp_frozen_names[] = {')
    print('    #ifdef MP_FROZEN_STR_NAMES')
    print('    MP_FROZEN_STR_NAMES')
    print('    #endif')
    mp_frozen_mpy_names_content = 1
    for cm in compiled_modules:
        module_name = cm.source_file.str
        print('    "%s\\0"' % module_name)
        mp_frozen_mpy_names_content += len(cm.source_file.str) + 1
    print('    "\\0"')
    print('};')
    print()
    print('const mp_frozen_module_t *const mp_frozen_mpy_content[] = {')
    for cm in compiled_modules:
        print('    &frozen_module_%s,' % cm.escaped_name)
    print('};')
    mp_frozen_mpy_content_size = len(compiled_modules * 4)
    print()
    print('#ifdef MICROPY_FROZEN_LIST_ITEM')
    for cm in compiled_modules:
        module_name = cm.source_file.str
        if module_name.endswith('/__init__.py'):
            short_name = module_name[:-len('/__init__.py')]
        else:
            short_name = module_name[:-len('.py')]
        print('MICROPY_FROZEN_LIST_ITEM("%s", "%s")' % (short_name, module_name))
    print('#endif')
    print()
    print('/*')
    print('byte sizes:')
    print('qstr content: %d unique, %d bytes' % (len(new), qstr_content))
    print('bc content: %d' % bc_content)
    print('const str content: %d' % const_str_content)
    print('const int content: %d' % const_int_content)
    print('const obj content: %d' % const_obj_content)
    print('const table qstr content: %d entries, %d bytes' % (const_table_qstr_content, const_table_qstr_content * 4))
    print('const table ptr content: %d entries, %d bytes' % (const_table_ptr_content, const_table_ptr_content * 4))
    print('raw code content: %d * 4 = %d' % (raw_code_count, raw_code_content))
    print('mp_frozen_mpy_names_content: %d' % mp_frozen_mpy_names_content)
    print('mp_frozen_mpy_content_size: %d' % mp_frozen_mpy_content_size)
    print('total: %d' % (qstr_content + bc_content + const_str_content + const_int_content + const_obj_content + const_table_qstr_content * 4 + const_table_ptr_content * 4 + raw_code_content + mp_frozen_mpy_names_content + mp_frozen_mpy_content_size))
    print('*/')

def adjust_bytecode_qstr_obj_indices(bytecode_in, qstr_table_base, obj_table_base):
    if False:
        for i in range(10):
            print('nop')
    opcodes = []
    labels = {}
    ip = 0
    while ip < len(bytecode_in):
        (fmt, sz, arg, extra_arg) = mp_opcode_decode(bytecode_in, ip)
        opcode = Opcode(ip, fmt, bytecode_in[ip], arg, extra_arg)
        labels[ip] = opcode
        opcodes.append(opcode)
        ip += sz
        if fmt == MP_BC_FORMAT_OFFSET:
            opcode.arg += ip
    for opcode in opcodes:
        if opcode.fmt == MP_BC_FORMAT_OFFSET:
            opcode.target = labels[opcode.arg]
    for opcode in opcodes:
        if opcode.fmt == MP_BC_FORMAT_QSTR:
            opcode.arg += qstr_table_base
        elif opcode.opcode_byte == Opcode.MP_BC_LOAD_CONST_OBJ:
            opcode.arg += obj_table_base
    offset_changed = True
    while offset_changed:
        offset_changed = False
        overflow = False
        bytecode_out = b''
        for opcode in opcodes:
            ip = len(bytecode_out)
            if opcode.offset != ip:
                offset_changed = True
                opcode.offset = ip
            (opcode_overflow, encoded_opcode) = mp_opcode_encode(opcode)
            if opcode_overflow:
                overflow = True
            bytecode_out += encoded_opcode
    if overflow:
        raise Exception('bytecode overflow')
    return bytecode_out

def rewrite_raw_code(rc, qstr_table_base, obj_table_base):
    if False:
        for i in range(10):
            print('nop')
    if rc.code_kind != MP_CODE_BYTECODE:
        raise Exception('can only rewrite bytecode')
    source_info = bytearray()
    for arg in rc.names:
        source_info.extend(mp_encode_uint(qstr_table_base + arg))
    closure_info = rc.fun_data[rc.offset_closure_info:rc.offset_opcodes]
    bytecode_in = memoryview(rc.fun_data)[rc.offset_opcodes:]
    bytecode_out = adjust_bytecode_qstr_obj_indices(bytecode_in, qstr_table_base, obj_table_base)
    prelude_signature = rc.fun_data[:rc.offset_prelude_size]
    prelude_size = encode_prelude_size(len(source_info), len(closure_info))
    fun_data = prelude_signature + prelude_size + source_info + closure_info + bytecode_out
    output = mp_encode_uint(len(fun_data) << 3 | bool(len(rc.children)) << 2)
    output += fun_data
    if rc.children:
        output += mp_encode_uint(len(rc.children))
        for child in rc.children:
            output += rewrite_raw_code(child, qstr_table_base, obj_table_base)
    return output

def merge_mpy(compiled_modules, output_file):
    if False:
        for i in range(10):
            print('nop')
    merged_mpy = bytearray()
    if len(compiled_modules) == 1:
        with open(compiled_modules[0].mpy_source_file, 'rb') as f:
            merged_mpy.extend(f.read())
    else:
        main_cm_idx = None
        for (idx, cm) in enumerate(compiled_modules):
            feature_byte = cm.header[2]
            mpy_native_arch = feature_byte >> 2
            if mpy_native_arch:
                if main_cm_idx is not None:
                    raise Exception("can't merge files when more than one contains native code")
                main_cm_idx = idx
        if main_cm_idx is not None:
            compiled_modules.insert(0, compiled_modules.pop(main_cm_idx))
        header = bytearray(4)
        header[0] = ord('M')
        header[1] = config.MPY_VERSION
        header[2] = config.native_arch << 2 | config.MPY_SUB_VERSION if config.native_arch else 0
        header[3] = config.mp_small_int_bits
        merged_mpy.extend(header)
        n_qstr = 0
        n_obj = 0
        for cm in compiled_modules:
            n_qstr += len(cm.qstr_table)
            n_obj += len(cm.obj_table)
        merged_mpy.extend(mp_encode_uint(n_qstr))
        merged_mpy.extend(mp_encode_uint(n_obj))

        def copy_section(file, offset, offset2):
            if False:
                while True:
                    i = 10
            with open(file, 'rb') as f:
                f.seek(offset)
                merged_mpy.extend(f.read(offset2 - offset))
        for cm in compiled_modules:
            copy_section(cm.mpy_source_file, cm.qstr_table_file_offset, cm.obj_table_file_offset)
        for cm in compiled_modules:
            copy_section(cm.mpy_source_file, cm.obj_table_file_offset, cm.raw_code_file_offset)
        bytecode = bytearray()
        bytecode.append(0)
        bytecode.append(2)
        bytecode.extend(b'\x00')
        for idx in range(len(compiled_modules)):
            bytecode.append(50)
            bytecode.append(idx)
            bytecode.extend(b'4\x00Y')
        bytecode.extend(b'Qc')
        merged_mpy.extend(mp_encode_uint(len(bytecode) << 3 | 1 << 2))
        merged_mpy.extend(bytecode)
        merged_mpy.extend(mp_encode_uint(len(compiled_modules)))
        qstr_table_base = 0
        obj_table_base = 0
        for cm in compiled_modules:
            if qstr_table_base == 0 and obj_table_base == 0:
                with open(cm.mpy_source_file, 'rb') as f:
                    f.seek(cm.raw_code_file_offset)
                    merged_mpy.extend(f.read())
            else:
                merged_mpy.extend(rewrite_raw_code(cm.raw_code, qstr_table_base, obj_table_base))
            qstr_table_base += len(cm.qstr_table)
            obj_table_base += len(cm.obj_table)
    if output_file is None:
        sys.stdout.buffer.write(merged_mpy)
    else:
        with open(output_file, 'wb') as f:
            f.write(merged_mpy)

def main():
    if False:
        for i in range(10):
            print('nop')
    global global_qstrs
    import argparse
    cmd_parser = argparse.ArgumentParser(description='A tool to work with MicroPython .mpy files.')
    cmd_parser.add_argument('-x', '--hexdump', action='store_true', help='output an annotated hex dump of files')
    cmd_parser.add_argument('-d', '--disassemble', action='store_true', help='output disassembled contents of files')
    cmd_parser.add_argument('-f', '--freeze', action='store_true', help='freeze files')
    cmd_parser.add_argument('--merge', action='store_true', help='merge multiple .mpy files into one')
    cmd_parser.add_argument('-q', '--qstr-header', help='qstr header file to freeze against')
    cmd_parser.add_argument('-mlongint-impl', choices=['none', 'longlong', 'mpz'], default='mpz', help='long-int implementation used by target (default mpz)')
    cmd_parser.add_argument('-mmpz-dig-size', metavar='N', type=int, default=16, help='mpz digit size used by target (default 16)')
    cmd_parser.add_argument('-o', '--output', default=None, help='output file')
    cmd_parser.add_argument('files', nargs='+', help='input .mpy files')
    args = cmd_parser.parse_args()
    config.MICROPY_LONGINT_IMPL = {'none': config.MICROPY_LONGINT_IMPL_NONE, 'longlong': config.MICROPY_LONGINT_IMPL_LONGLONG, 'mpz': config.MICROPY_LONGINT_IMPL_MPZ}[args.mlongint_impl]
    config.MPZ_DIG_SIZE = args.mmpz_dig_size
    config.native_arch = MP_NATIVE_ARCH_NONE
    if args.qstr_header:
        (qcfgs, extra_qstrs) = qstrutil.parse_input_headers([args.qstr_header])
        firmware_qstr_idents = set(qstrutil.static_qstr_list_ident) | set(extra_qstrs.keys())
        config.MICROPY_QSTR_BYTES_IN_LEN = int(qcfgs['BYTES_IN_LEN'])
        config.MICROPY_QSTR_BYTES_IN_HASH = int(qcfgs['BYTES_IN_HASH'])
    else:
        config.MICROPY_QSTR_BYTES_IN_LEN = 1
        config.MICROPY_QSTR_BYTES_IN_HASH = 1
        firmware_qstr_idents = set(qstrutil.static_qstr_list)
    global_qstrs = GlobalQStrList()
    try:
        compiled_modules = [read_mpy(file) for file in args.files]
    except MPYReadError as er:
        print(er, file=sys.stderr)
        sys.exit(1)
    if args.hexdump:
        hexdump_mpy(compiled_modules)
    if args.disassemble:
        if args.hexdump:
            print()
        disassemble_mpy(compiled_modules)
    if args.freeze:
        try:
            freeze_mpy(firmware_qstr_idents, compiled_modules)
        except FreezeError as er:
            print(er, file=sys.stderr)
            sys.exit(1)
    if args.merge:
        merge_mpy(compiled_modules, args.output)
if __name__ == '__main__':
    main()