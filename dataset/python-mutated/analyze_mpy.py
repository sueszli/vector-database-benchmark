import sys
import binascii
import io
bytecode_format_sizes = {'MP_OPCODE_BYTE': 1, 'MP_OPCODE_QSTR': 3, 'MP_OPCODE_VAR_UINT': None, 'MP_OPCODE_OFFSET': 3, 'MP_OPCODE_BYTE_EXTRA': 2, 'MP_OPCODE_VAR_UINT_EXTRA': None, 'MP_OPCODE_OFFSET_EXTRA': 4}
bytecodes = {0: {'name': 'MP_BC_LOAD_FAST_MULTI', 'format': 'MP_OPCODE_BYTE'}, 16: {'name': 'MP_BC_LOAD_CONST_FALSE', 'format': 'MP_OPCODE_BYTE'}, 17: {'name': 'MP_BC_LOAD_CONST_NONE', 'format': 'MP_OPCODE_BYTE'}, 18: {'name': 'MP_BC_LOAD_CONST_TRUE', 'format': 'MP_OPCODE_BYTE'}, 20: {'name': 'MP_BC_LOAD_CONST_SMALL_INT', 'format': 'MP_OPCODE_VAR_UINT'}, 22: {'name': 'MP_BC_LOAD_CONST_STRING', 'format': 'MP_OPCODE_QSTR'}, 23: {'name': 'MP_BC_LOAD_CONST_OBJ', 'format': 'MP_OPCODE_VAR_UINT'}, 24: {'name': 'MP_BC_LOAD_NULL', 'format': 'MP_OPCODE_BYTE'}, 26: {'name': 'MP_BC_LOAD_DEREF', 'format': 'MP_OPCODE_VAR_UINT'}, 27: {'name': 'MP_BC_LOAD_NAME', 'format': 'MP_OPCODE_QSTR'}, 28: {'name': 'MP_BC_LOAD_GLOBAL', 'format': 'MP_OPCODE_QSTR'}, 29: {'name': 'MP_BC_LOAD_ATTR', 'format': 'MP_OPCODE_QSTR'}, 30: {'name': 'MP_BC_LOAD_METHOD', 'format': 'MP_OPCODE_QSTR'}, 31: {'name': 'MP_BC_LOAD_SUPER_METHOD', 'format': 'MP_OPCODE_QSTR'}, 32: {'name': 'MP_BC_LOAD_BUILD_CLASS', 'format': 'MP_OPCODE_BYTE'}, 33: {'name': 'MP_BC_LOAD_SUBSCR', 'format': 'MP_OPCODE_BYTE'}, 36: {'name': 'MP_BC_STORE_NAME', 'format': 'MP_OPCODE_QSTR'}, 37: {'name': 'MP_BC_STORE_GLOBAL', 'format': 'MP_OPCODE_QSTR'}, 38: {'name': 'MP_BC_STORE_ATTR', 'format': 'MP_OPCODE_QSTR'}, 39: {'name': 'MP_BC_LOAD_SUBSCR', 'format': 'MP_OPCODE_BYTE'}, 40: {'name': 'MP_BC_DELETE_FAST', 'format': 'MP_OPCODE_VAR_UINT'}, 48: {'name': 'MP_BC_DUP_TOP', 'format': 'MP_OPCODE_BYTE'}, 50: {'name': 'MP_BC_POP_TOP', 'format': 'MP_OPCODE_BYTE'}, 51: {'name': 'MP_BC_ROT_TWO', 'format': 'MP_OPCODE_BYTE'}, 52: {'name': 'MP_BC_ROT_THREE', 'format': 'MP_OPCODE_BYTE'}, 53: {'name': 'MP_BC_JUMP', 'format': 'MP_OPCODE_OFFSET'}, 54: {'name': 'MP_BC_POP_JUMP_IF_TRUE', 'format': 'MP_OPCODE_OFFSET'}, 55: {'name': 'MP_BC_POP_JUMP_IF_FALSE', 'format': 'MP_OPCODE_OFFSET'}, 67: {'name': 'MP_BC_FOR_ITER', 'format': 'MP_OPCODE_OFFSET'}, 68: {'name': 'MP_BC_POP_BLOCK', 'format': 'MP_OPCODE_BYTE'}, 71: {'name': 'MP_BC_GET_ITER_STACK', 'format': 'MP_OPCODE_BYTE'}, 80: {'name': 'MP_BC_BUILD_TUPLE', 'format': 'MP_OPCODE_VAR_UINT'}, 81: {'name': 'MP_BC_BUILD_LIST', 'format': 'MP_OPCODE_VAR_UINT'}, 83: {'name': 'MP_BC_BUILD_MAP', 'format': 'MP_OPCODE_VAR_UINT'}, 84: {'name': 'MP_BC_STORE_MAP', 'format': 'MP_OPCODE_BYTE'}, 87: {'name': 'MP_BC_STORE_COMP', 'format': 'MP_OPCODE_VAR_UINT'}, 91: {'name': 'MP_BC_RETURN_VALUE', 'format': 'MP_OPCODE_BYTE'}, 92: {'name': 'MP_BC_RAISE_VARARGS', 'format': 'MP_OPCODE_BYTE_EXTRA'}, 96: {'name': 'MP_BC_MAKE_FUNCTION', 'format': 'MP_OPCODE_VAR_UINT'}, 97: {'name': 'MP_BC_MAKE_FUNCTION_DEFARGS', 'format': 'MP_OPCODE_VAR_UINT'}, 98: {'name': 'MP_BC_MAKE_CLOSURE', 'format': 'MP_OPCODE_VAR_UINT_EXTRA'}, 99: {'name': 'MP_BC_MAKE_CLOSURE', 'format': 'MP_OPCODE_VAR_UINT_EXTRA'}, 100: {'name': 'MP_BC_CALL_FUNCTION', 'format': 'MP_OPCODE_VAR_UINT'}, 101: {'name': 'MP_BC_CALL_FUNCTION_VAR_KW', 'format': 'MP_OPCODE_VAR_UINT'}, 102: {'name': 'MP_BC_CALL_METHOD', 'format': 'MP_OPCODE_VAR_UINT'}, 103: {'name': 'MP_BC_CALL_METHOD_VAR_KW', 'format': 'MP_OPCODE_VAR_UINT'}, 104: {'name': 'MP_BC_IMPORT_NAME', 'format': 'MP_OPCODE_QSTR'}, 105: {'name': 'MP_BC_IMPORT_FROM', 'format': 'MP_OPCODE_QSTR'}, 127: {'name': 'MP_BC_LOAD_CONST_SMALL_INT_MULTI -1', 'format': 'MP_OPCODE_BYTE'}, 128: {'name': 'MP_BC_LOAD_CONST_SMALL_INT_MULTI 0', 'format': 'MP_OPCODE_BYTE'}, 129: {'name': 'MP_BC_LOAD_CONST_SMALL_INT_MULTI 1', 'format': 'MP_OPCODE_BYTE'}, 130: {'name': 'MP_BC_LOAD_CONST_SMALL_INT_MULTI 2', 'format': 'MP_OPCODE_BYTE'}, 131: {'name': 'MP_BC_LOAD_CONST_SMALL_INT_MULTI 3', 'format': 'MP_OPCODE_BYTE'}, 132: {'name': 'MP_BC_LOAD_CONST_SMALL_INT_MULTI 4', 'format': 'MP_OPCODE_BYTE'}, 176: {'name': 'MP_BC_LOAD_FAST_MULTI 0', 'format': 'MP_OPCODE_BYTE'}, 177: {'name': 'MP_BC_LOAD_FAST_MULTI 1', 'format': 'MP_OPCODE_BYTE'}, 178: {'name': 'MP_BC_LOAD_FAST_MULTI 2', 'format': 'MP_OPCODE_BYTE'}, 179: {'name': 'MP_BC_LOAD_FAST_MULTI 3', 'format': 'MP_OPCODE_BYTE'}, 180: {'name': 'MP_BC_LOAD_FAST_MULTI 4', 'format': 'MP_OPCODE_BYTE'}, 181: {'name': 'MP_BC_LOAD_FAST_MULTI 5', 'format': 'MP_OPCODE_BYTE'}, 182: {'name': 'MP_BC_LOAD_FAST_MULTI 6', 'format': 'MP_OPCODE_BYTE'}, 183: {'name': 'MP_BC_LOAD_FAST_MULTI 7', 'format': 'MP_OPCODE_BYTE'}, 184: {'name': 'MP_BC_LOAD_FAST_MULTI 8', 'format': 'MP_OPCODE_BYTE'}, 192: {'name': 'MP_BC_STORE_FAST_MULTI 0', 'format': 'MP_OPCODE_BYTE'}, 193: {'name': 'MP_BC_STORE_FAST_MULTI 1', 'format': 'MP_OPCODE_BYTE'}, 194: {'name': 'MP_BC_STORE_FAST_MULTI 2', 'format': 'MP_OPCODE_BYTE'}, 195: {'name': 'MP_BC_STORE_FAST_MULTI 3', 'format': 'MP_OPCODE_BYTE'}, 196: {'name': 'MP_BC_STORE_FAST_MULTI 4', 'format': 'MP_OPCODE_BYTE'}, 197: {'name': 'MP_BC_STORE_FAST_MULTI 5', 'format': 'MP_OPCODE_BYTE'}, 198: {'name': 'MP_BC_STORE_FAST_MULTI 6', 'format': 'MP_OPCODE_BYTE'}, 199: {'name': 'MP_BC_STORE_FAST_MULTI 7', 'format': 'MP_OPCODE_BYTE'}, 215: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_LESS', 'format': 'MP_OPCODE_BYTE'}, 216: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_MORE', 'format': 'MP_OPCODE_BYTE'}, 217: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_EQUAL', 'format': 'MP_OPCODE_BYTE'}, 218: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_LESS_EQUAL', 'format': 'MP_OPCODE_BYTE'}, 219: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_MORE_EQUAL', 'format': 'MP_OPCODE_BYTE'}, 220: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_NOT_EQUAL', 'format': 'MP_OPCODE_BYTE'}, 229: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_INPLACE_ADD', 'format': 'MP_OPCODE_BYTE'}, 230: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_INPLACE_SUBTRACT', 'format': 'MP_OPCODE_BYTE'}, 241: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_ADD', 'format': 'MP_OPCODE_BYTE'}, 242: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_SUBTRACT', 'format': 'MP_OPCODE_BYTE'}, 243: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_MULTIPLY', 'format': 'MP_OPCODE_BYTE'}, 244: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_FLOOR_DIVIDE', 'format': 'MP_OPCODE_BYTE'}, 245: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_TRUE_DIVIDE', 'format': 'MP_OPCODE_BYTE'}, 246: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_MODULO', 'format': 'MP_OPCODE_BYTE'}, 247: {'name': 'MP_BC_BINARY_OP_MULTI MP_BINARY_OP_POWER', 'format': 'MP_OPCODE_BYTE'}}

def read_uint(encoded_uint, peek=False):
    if False:
        while True:
            i = 10
    unum = 0
    i = 0
    while True:
        if peek:
            b = encoded_uint.peek()[i]
        else:
            b = encoded_uint.read(1)[0]
        unum = unum << 7 | b & 127
        if b & 128 == 0:
            break
        i += 1
    return unum

class Prelude:

    def __init__(self, encoded_prelude):
        if False:
            print('Hello World!')
        self.n_state = read_uint(encoded_prelude)
        self.n_exc_stack = read_uint(encoded_prelude)
        self.scope_flags = encoded_prelude.read(1)[0]
        self.n_pos_args = encoded_prelude.read(1)[0]
        self.n_kwonly_args = encoded_prelude.read(1)[0]
        self.n_def_pos_args = encoded_prelude.read(1)[0]
        self.code_info_size = read_uint(encoded_prelude, peek=True)

class RawCode:

    def __init__(self, encoded_raw_code):
        if False:
            i = 10
            return i + 15
        bc_len = read_uint(encoded_raw_code)
        bc = encoded_raw_code.read(bc_len)
        bc = io.BufferedReader(io.BytesIO(bc))
        prelude = Prelude(bc)
        encoded_code_info = bc.read(prelude.code_info_size)
        bc.read(1)
        while bc.peek(1)[0] == 255:
            bc.read(1)
        bc = bytearray(bc.read())
        self.qstrs = []
        self.simple_name = self._load_qstr(encoded_raw_code)
        self.source_file = self._load_qstr(encoded_raw_code)
        self._load_bytecode_qstrs(encoded_raw_code, bc)
        n_obj = read_uint(encoded_raw_code)
        n_raw_code = read_uint(encoded_raw_code)
        self.const_table = []
        for i in range(prelude.n_pos_args + prelude.n_kwonly_args):
            self.const_table.append(self._load_qstr(encoded_raw_code))
            print('load args', self.const_table[-1])
        for i in range(n_obj):
            self.const_table.append(self._load_obj(encoded_raw_code))
            print('load obj', self.const_table[-1])
        for i in range(n_raw_code):
            print('load raw code')
            self.const_table.append(RawCode(encoded_raw_code))
        print(self.qstrs[self.simple_name], self.qstrs[self.source_file])

    def _load_qstr(self, encoded_qstr):
        if False:
            for i in range(10):
                print('nop')
        string_len = read_uint(encoded_qstr)
        string = encoded_qstr.read(string_len).decode('utf-8')
        print(string)
        if string in self.qstrs:
            return self.qstrs.index(string)
        new_index = len(self.qstrs)
        self.qstrs.append(string)
        return new_index

    def _load_obj(self, encoded_obj):
        if False:
            print('Hello World!')
        obj_type = encoded_obj.read(1)
        if obj_type == b'e':
            return '...'
        else:
            str_len = read_uint(encoded_obj)
            s = encoded_obj.read(str_len)
            if obj_type == b's':
                return s.decode('utf-8')
            elif obj_type == b'b':
                return s
            elif obj_type == b'i':
                return int(s)
            elif obj_type == b'f':
                return float(s)
            elif obj_type == b'c':
                return float(s)
        raise RuntimeError('Unknown object type {}'.format(obj_type))

    def _load_bytecode_qstrs(self, encoded_raw_code, bytecode):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        while i < len(bytecode):
            bc = bytecode[i]
            if bc not in bytecodes:
                raise RuntimeError('missing code 0x{:x} at {}'.format(bc, i))
                return
            bc = bytecodes[bc]
            opcode = bc['name']
            print(opcode)
            opcode_size = bytecode_format_sizes[bc['format']]
            if bc['format'] == 'MP_OPCODE_QSTR':
                qstr_index = self._load_qstr(encoded_raw_code)
                bytecode[i + 1] = qstr_index
                bytecode[i + 2] = qstr_index >> 8
            if not opcode_size:
                i += 2
                while bytecode[i] & 128 != 0:
                    i += 1
                if bc['format'] == 'MP_OPCODE_VAR_UINT_EXTRA':
                    i += 1
            else:
                i += opcode_size

class mpyFile:

    def __init__(self, encoded_mpy):
        if False:
            return 10
        first_byte = encoded_mpy.read(1)
        if first_byte != b'C':
            raise ValueError("Not a valid first byte. Should be 'C' but is {}".format(first_byte))
        self.version = encoded_mpy.read(1)[0]
        self.feature_flags = encoded_mpy.read(1)[0]
        self.small_int_bits = encoded_mpy.read(1)[0]
        self.raw_code = RawCode(encoded_mpy)
if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        mpy = mpyFile(f)
        print(mpy.version)
        print(mpy.feature_flags)
        print(mpy.small_int_bits)