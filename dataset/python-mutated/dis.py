"""Disassembler of Python byte code into mnemonics."""
import sys
import types
import collections
import io
from opcode import *
from opcode import __all__ as _opcodes_all
__all__ = ['code_info', 'dis', 'disassemble', 'distb', 'disco', 'findlinestarts', 'findlabels', 'show_code', 'get_instructions', 'Instruction', 'Bytecode'] + _opcodes_all
del _opcodes_all
_have_code = (types.MethodType, types.FunctionType, types.CodeType, classmethod, staticmethod, type)
FORMAT_VALUE = opmap['FORMAT_VALUE']
FORMAT_VALUE_CONVERTERS = ((None, ''), (str, 'str'), (repr, 'repr'), (ascii, 'ascii'))
MAKE_FUNCTION = opmap['MAKE_FUNCTION']
MAKE_FUNCTION_FLAGS = ('defaults', 'kwdefaults', 'annotations', 'closure')

def _try_compile(source, name):
    if False:
        i = 10
        return i + 15
    'Attempts to compile the given source, first as an expression and\n       then as a statement if the first approach fails.\n\n       Utility function to accept strings in functions that otherwise\n       expect code objects\n    '
    try:
        c = compile(source, name, 'eval')
    except SyntaxError:
        c = compile(source, name, 'exec')
    return c

def dis(x=None, *, file=None, depth=None):
    if False:
        i = 10
        return i + 15
    'Disassemble classes, methods, functions, and other compiled objects.\n\n    With no argument, disassemble the last traceback.\n\n    Compiled objects currently include generator objects, async generator\n    objects, and coroutine objects, all of which store their code object\n    in a special attribute.\n    '
    if x is None:
        distb(file=file)
        return
    if hasattr(x, '__func__'):
        x = x.__func__
    if hasattr(x, '__code__'):
        x = x.__code__
    elif hasattr(x, 'gi_code'):
        x = x.gi_code
    elif hasattr(x, 'ag_code'):
        x = x.ag_code
    elif hasattr(x, 'cr_code'):
        x = x.cr_code
    if hasattr(x, '__dict__'):
        items = sorted(x.__dict__.items())
        for (name, x1) in items:
            if isinstance(x1, _have_code):
                print('Disassembly of %s:' % name, file=file)
                try:
                    dis(x1, file=file, depth=depth)
                except TypeError as msg:
                    print('Sorry:', msg, file=file)
                print(file=file)
    elif hasattr(x, 'co_code'):
        _disassemble_recursive(x, file=file, depth=depth)
    elif isinstance(x, (bytes, bytearray)):
        _disassemble_bytes(x, file=file)
    elif isinstance(x, str):
        _disassemble_str(x, file=file, depth=depth)
    else:
        raise TypeError("don't know how to disassemble %s objects" % type(x).__name__)

def distb(tb=None, *, file=None):
    if False:
        for i in range(10):
            print('nop')
    'Disassemble a traceback (default: last traceback).'
    if tb is None:
        try:
            tb = sys.last_traceback
        except AttributeError:
            raise RuntimeError('no last traceback to disassemble') from None
        while tb.tb_next:
            tb = tb.tb_next
    disassemble(tb.tb_frame.f_code, tb.tb_lasti, file=file)
COMPILER_FLAG_NAMES = {1: 'OPTIMIZED', 2: 'NEWLOCALS', 4: 'VARARGS', 8: 'VARKEYWORDS', 16: 'NESTED', 32: 'GENERATOR', 64: 'NOFREE', 128: 'COROUTINE', 256: 'ITERABLE_COROUTINE', 512: 'ASYNC_GENERATOR', 1073741824: 'SUPPRESS_JIT'}

def pretty_flags(flags):
    if False:
        while True:
            i = 10
    'Return pretty representation of code flags.'
    names = []
    for i in range(32):
        flag = 1 << i
        if flags & flag:
            names.append(COMPILER_FLAG_NAMES.get(flag, hex(flag)))
            flags ^= flag
            if not flags:
                break
    else:
        names.append(hex(flags))
    return ', '.join(names)

def _get_code_object(x):
    if False:
        i = 10
        return i + 15
    'Helper to handle methods, compiled or raw code objects, and strings.'
    if hasattr(x, '__func__'):
        x = x.__func__
    if hasattr(x, '__code__'):
        x = x.__code__
    elif hasattr(x, 'gi_code'):
        x = x.gi_code
    elif hasattr(x, 'ag_code'):
        x = x.ag_code
    elif hasattr(x, 'cr_code'):
        x = x.cr_code
    if isinstance(x, str):
        x = _try_compile(x, '<disassembly>')
    if hasattr(x, 'co_code'):
        return x
    raise TypeError("don't know how to disassemble %s objects" % type(x).__name__)

def code_info(x):
    if False:
        i = 10
        return i + 15
    'Formatted details of methods, functions, or code.'
    return _format_code_info(_get_code_object(x))

def _format_code_info(co):
    if False:
        return 10
    lines = []
    lines.append('Name:              %s' % co.co_name)
    lines.append('Filename:          %s' % co.co_filename)
    lines.append('Argument count:    %s' % co.co_argcount)
    lines.append('Positional-only arguments: %s' % co.co_posonlyargcount)
    lines.append('Kw-only arguments: %s' % co.co_kwonlyargcount)
    lines.append('Number of locals:  %s' % co.co_nlocals)
    lines.append('Stack size:        %s' % co.co_stacksize)
    lines.append('Flags:             %s' % pretty_flags(co.co_flags))
    if co.co_consts:
        lines.append('Constants:')
        for i_c in enumerate(co.co_consts):
            lines.append('%4d: %r' % i_c)
    if co.co_names:
        lines.append('Names:')
        for i_n in enumerate(co.co_names):
            lines.append('%4d: %s' % i_n)
    if co.co_varnames:
        lines.append('Variable names:')
        for i_n in enumerate(co.co_varnames):
            lines.append('%4d: %s' % i_n)
    if co.co_freevars:
        lines.append('Free variables:')
        for i_n in enumerate(co.co_freevars):
            lines.append('%4d: %s' % i_n)
    if co.co_cellvars:
        lines.append('Cell variables:')
        for i_n in enumerate(co.co_cellvars):
            lines.append('%4d: %s' % i_n)
    return '\n'.join(lines)

def show_code(co, *, file=None):
    if False:
        print('Hello World!')
    'Print details of methods, functions, or code to *file*.\n\n    If *file* is not provided, the output is printed on stdout.\n    '
    print(code_info(co), file=file)
_Instruction = collections.namedtuple('_Instruction', 'opname opcode arg argval argrepr offset starts_line is_jump_target')
_Instruction.opname.__doc__ = 'Human readable name for operation'
_Instruction.opcode.__doc__ = 'Numeric code for operation'
_Instruction.arg.__doc__ = 'Numeric argument to operation (if any), otherwise None'
_Instruction.argval.__doc__ = 'Resolved arg value (if known), otherwise same as arg'
_Instruction.argrepr.__doc__ = 'Human readable description of operation argument'
_Instruction.offset.__doc__ = 'Start index of operation within bytecode sequence'
_Instruction.starts_line.__doc__ = 'Line started by this opcode (if any), otherwise None'
_Instruction.is_jump_target.__doc__ = 'True if other code jumps to here, otherwise False'
_OPNAME_WIDTH = 20
_OPARG_WIDTH = 5

class Instruction(_Instruction):
    """Details for a bytecode operation

       Defined fields:
         opname - human readable name for operation
         opcode - numeric code for operation
         arg - numeric argument to operation (if any), otherwise None
         argval - resolved arg value (if known), otherwise same as arg
         argrepr - human readable description of operation argument
         offset - start index of operation within bytecode sequence
         starts_line - line started by this opcode (if any), otherwise None
         is_jump_target - True if other code jumps to here, otherwise False
    """

    def _disassemble(self, lineno_width=3, mark_as_current=False, offset_width=4):
        if False:
            i = 10
            return i + 15
        "Format instruction details for inclusion in disassembly output\n\n        *lineno_width* sets the width of the line number field (0 omits it)\n        *mark_as_current* inserts a '-->' marker arrow as part of the line\n        *offset_width* sets the width of the instruction offset field\n        "
        fields = []
        if lineno_width:
            if self.starts_line is not None:
                lineno_fmt = '%%%dd' % lineno_width
                fields.append(lineno_fmt % self.starts_line)
            else:
                fields.append(' ' * lineno_width)
        if mark_as_current:
            fields.append('-->')
        else:
            fields.append('   ')
        if self.is_jump_target:
            fields.append('>>')
        else:
            fields.append('  ')
        fields.append(repr(self.offset).rjust(offset_width))
        fields.append(self.opname.ljust(_OPNAME_WIDTH))
        if self.arg is not None:
            fields.append(repr(self.arg).rjust(_OPARG_WIDTH))
            if self.argrepr:
                fields.append('(' + self.argrepr + ')')
        return ' '.join(fields).rstrip()

def get_instructions(x, *, first_line=None):
    if False:
        while True:
            i = 10
    'Iterator for the opcodes in methods, functions or code\n\n    Generates a series of Instruction named tuples giving the details of\n    each operations in the supplied code.\n\n    If *first_line* is not None, it indicates the line number that should\n    be reported for the first source line in the disassembled code.\n    Otherwise, the source line information (if any) is taken directly from\n    the disassembled code object.\n    '
    co = _get_code_object(x)
    cell_names = co.co_cellvars + co.co_freevars
    linestarts = dict(findlinestarts(co))
    if first_line is not None:
        line_offset = first_line - co.co_firstlineno
    else:
        line_offset = 0
    return _get_instructions_bytes(co.co_code, co.co_varnames, co.co_names, co.co_consts, cell_names, linestarts, line_offset)

def _get_const_info(const_index, const_list):
    if False:
        i = 10
        return i + 15
    'Helper to get optional details about const references\n\n       Returns the dereferenced constant and its repr if the constant\n       list is defined.\n       Otherwise returns the constant index and its repr().\n    '
    argval = const_index
    if const_list is not None:
        argval = const_list[const_index]
    return (argval, repr(argval))

def _get_name_info(name_index, name_list):
    if False:
        for i in range(10):
            print('nop')
    'Helper to get optional details about named references\n\n       Returns the dereferenced name as both value and repr if the name\n       list is defined.\n       Otherwise returns the name index and its repr().\n    '
    argval = name_index
    if name_list is not None:
        argval = name_list[name_index]
        argrepr = argval
    else:
        argrepr = repr(argval)
    return (argval, argrepr)

def _get_instructions_bytes(code, varnames=None, names=None, constants=None, cells=None, linestarts=None, line_offset=0):
    if False:
        return 10
    "Iterate over the instructions in a bytecode string.\n\n    Generates a sequence of Instruction namedtuples giving the details of each\n    opcode.  Additional information about the code's runtime environment\n    (e.g. variable names, constants) can be specified using optional\n    arguments.\n\n    "
    labels = findlabels(code)
    starts_line = None
    for (offset, op, arg) in _unpack_opargs(code):
        if linestarts is not None:
            starts_line = linestarts.get(offset, None)
            if starts_line is not None:
                starts_line += line_offset
        is_jump_target = offset in labels
        argval = None
        argrepr = ''
        if arg is not None:
            argval = arg
            if op in hasconst:
                (argval, argrepr) = _get_const_info(arg, constants)
            elif op in hasname:
                (argval, argrepr) = _get_name_info(arg, names)
            elif op in hasjabs:
                argval = arg * 2
                argrepr = 'to ' + repr(argval)
            elif op in hasjrel:
                argval = offset + 2 + arg * 2
                argrepr = 'to ' + repr(argval)
            elif op in haslocal:
                (argval, argrepr) = _get_name_info(arg, varnames)
            elif op in hascompare:
                argval = cmp_op[arg]
                argrepr = argval
            elif op in hasfree:
                (argval, argrepr) = _get_name_info(arg, cells)
            elif op == FORMAT_VALUE:
                (argval, argrepr) = FORMAT_VALUE_CONVERTERS[arg & 3]
                argval = (argval, bool(arg & 4))
                if argval[1]:
                    if argrepr:
                        argrepr += ', '
                    argrepr += 'with format'
            elif op == MAKE_FUNCTION:
                argrepr = ', '.join((s for (i, s) in enumerate(MAKE_FUNCTION_FLAGS) if arg & 1 << i))
        yield Instruction(opname[op], op, arg, argval, argrepr, offset, starts_line, is_jump_target)

def disassemble(co, lasti=-1, *, file=None):
    if False:
        while True:
            i = 10
    'Disassemble a code object.'
    cell_names = co.co_cellvars + co.co_freevars
    linestarts = dict(findlinestarts(co))
    _disassemble_bytes(co.co_code, lasti, co.co_varnames, co.co_names, co.co_consts, cell_names, linestarts, file=file)

def _disassemble_recursive(co, *, file=None, depth=None):
    if False:
        for i in range(10):
            print('nop')
    disassemble(co, file=file)
    if depth is None or depth > 0:
        if depth is not None:
            depth = depth - 1
        for x in co.co_consts:
            if hasattr(x, 'co_code'):
                print(file=file)
                print('Disassembly of %r:' % (x,), file=file)
                _disassemble_recursive(x, file=file, depth=depth)

def _disassemble_bytes(code, lasti=-1, varnames=None, names=None, constants=None, cells=None, linestarts=None, *, file=None, line_offset=0):
    if False:
        return 10
    show_lineno = bool(linestarts)
    if show_lineno:
        maxlineno = max(linestarts.values()) + line_offset
        if maxlineno >= 1000:
            lineno_width = len(str(maxlineno))
        else:
            lineno_width = 3
    else:
        lineno_width = 0
    maxoffset = len(code) - 2
    if maxoffset >= 10000:
        offset_width = len(str(maxoffset))
    else:
        offset_width = 4
    for instr in _get_instructions_bytes(code, varnames, names, constants, cells, linestarts, line_offset=line_offset):
        new_source_line = show_lineno and instr.starts_line is not None and (instr.offset > 0)
        if new_source_line:
            print(file=file)
        is_current_instr = instr.offset == lasti
        print(instr._disassemble(lineno_width, is_current_instr, offset_width), file=file)

def _disassemble_str(source, **kwargs):
    if False:
        i = 10
        return i + 15
    'Compile the source string, then disassemble the code object.'
    _disassemble_recursive(_try_compile(source, '<dis>'), **kwargs)
disco = disassemble

def _unpack_opargs(code):
    if False:
        while True:
            i = 10
    extended_arg = 0
    for i in range(0, len(code), 2):
        op = code[i]
        if op >= HAVE_ARGUMENT:
            arg = code[i + 1] | extended_arg
            extended_arg = arg << 8 if op == EXTENDED_ARG else 0
        else:
            arg = None
            extended_arg = 0
        yield (i, op, arg)

def findlabels(code):
    if False:
        print('Hello World!')
    'Detect all offsets in a byte code which are jump targets.\n\n    Return the list of offsets.\n\n    '
    labels = []
    for (offset, op, arg) in _unpack_opargs(code):
        if arg is not None:
            if op in hasjrel:
                label = offset + 2 + arg * 2
            elif op in hasjabs:
                label = arg * 2
            else:
                continue
            if label not in labels:
                labels.append(label)
    return labels

def findlinestarts(code):
    if False:
        print('Hello World!')
    'Find the offsets in a byte code which are start of lines in the source.\n\n    Generate pairs (offset, lineno)\n    '
    lastline = None
    for (start, end, line) in code.co_lines():
        if line is not None and line != lastline:
            lastline = line
            yield (start, line)
    return

class Bytecode:
    """The bytecode operations of a piece of code

    Instantiate this with a function, method, other compiled object, string of
    code, or a code object (as returned by compile()).

    Iterating over this yields the bytecode operations as Instruction instances.
    """

    def __init__(self, x, *, first_line=None, current_offset=None):
        if False:
            print('Hello World!')
        self.codeobj = co = _get_code_object(x)
        if first_line is None:
            self.first_line = co.co_firstlineno
            self._line_offset = 0
        else:
            self.first_line = first_line
            self._line_offset = first_line - co.co_firstlineno
        self._cell_names = co.co_cellvars + co.co_freevars
        self._linestarts = dict(findlinestarts(co))
        self._original_object = x
        self.current_offset = current_offset

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        co = self.codeobj
        return _get_instructions_bytes(co.co_code, co.co_varnames, co.co_names, co.co_consts, self._cell_names, self._linestarts, line_offset=self._line_offset)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '{}({!r})'.format(self.__class__.__name__, self._original_object)

    @classmethod
    def from_traceback(cls, tb):
        if False:
            print('Hello World!')
        ' Construct a Bytecode from the given traceback '
        while tb.tb_next:
            tb = tb.tb_next
        return cls(tb.tb_frame.f_code, current_offset=tb.tb_lasti)

    def info(self):
        if False:
            return 10
        'Return formatted information about the code object.'
        return _format_code_info(self.codeobj)

    def dis(self):
        if False:
            print('Hello World!')
        'Return a formatted view of the bytecode operations.'
        co = self.codeobj
        if self.current_offset is not None:
            offset = self.current_offset
        else:
            offset = -1
        with io.StringIO() as output:
            _disassemble_bytes(co.co_code, varnames=co.co_varnames, names=co.co_names, constants=co.co_consts, cells=self._cell_names, linestarts=self._linestarts, line_offset=self._line_offset, file=output, lasti=offset)
            return output.getvalue()

def _test():
    if False:
        print('Hello World!')
    'Simple test program to disassemble a file.'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('rb'), nargs='?', default='-')
    args = parser.parse_args()
    with args.infile as infile:
        source = infile.read()
    code = compile(source, args.infile.name, 'exec')
    dis(code)
if __name__ == '__main__':
    _test()