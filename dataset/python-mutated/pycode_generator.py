from __future__ import annotations
import random
import sys
import types
from functools import cached_property
from typing import TYPE_CHECKING
import opcode
import paddle
from ...utils import FallbackError, InnerError, OrderedSet, ResumeFnNameFactory, is_clean_code, list_contain_by_id, list_find_index_by_id, no_eval_frame
from ..instruction_utils import analysis_inputs, calc_stack_effect, gen_instr, get_instructions, instrs_info, modify_instrs, modify_vars
from ..instruction_utils.opcode_info import PYOPCODE_CACHE_SIZE, UNCONDITIONAL_JUMP, JumpDirection, PopJumpCond
from .instr_flag import CALL_FUNCTION_EX_FLAG
CODE_NAME_RNG = random.Random(2023)
if TYPE_CHECKING:
    from typing import Any
    from ..instruction_utils import Instruction

def get_pycode_attributes() -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of attribute names for PyCodeObject.\n    NOTE(SigureMo): The order should consistent with signature specified in code_doc\n    3.8: https://github.com/python/cpython/blob/3.8/Objects/codeobject.c#L416-L421\n    3.10: https://github.com/python/cpython/blob/3.10/Objects/codeobject.c#L523-L543\n    3.11: https://github.com/python/cpython/blob/3.11/Objects/codeobject.c#L1494-L1516\n\n    Returns:\n        list[str]: The attribute names for PyCodeObject.\n    '
    pycode_attributes = ['co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals', 'co_stacksize', 'co_flags', 'co_code', 'co_consts', 'co_names', 'co_varnames', 'co_filename', 'co_name']
    if sys.version_info >= (3, 11):
        pycode_attributes.append('co_qualname')
    pycode_attributes.append('co_firstlineno')
    if sys.version_info >= (3, 10):
        pycode_attributes.append('co_linetable')
    else:
        pycode_attributes.append('co_lnotab')
    if sys.version_info >= (3, 11):
        pycode_attributes.append('co_exceptiontable')
    pycode_attributes += ['co_freevars', 'co_cellvars']
    return pycode_attributes
PYCODE_ATTRIBUTES = get_pycode_attributes()

def gen_code_options(code: types.CodeType) -> dict[str, Any]:
    if False:
        i = 10
        return i + 15
    '\n    Generates a dictionary of code options for the given code object.\n\n    Args:\n        code (types.CodeType): The code object.\n\n    Returns:\n        dict[str, any]: The code options.\n    '
    code_options = {}
    for k in PYCODE_ATTRIBUTES:
        val = getattr(code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val
    return code_options

def gen_new_opcode(instrs: list[Instruction], code_options: dict[str, Any], keys: list[str]) -> types.CodeType:
    if False:
        return 10
    '\n    Generates a new code object with the given instructions, code options, and keys.\n\n    Args:\n        instrs (list[Instruction]): The instructions for the new code object.\n        code_options (dict[str, any]): The code options for the new code object.\n        keys (list[str]): The keys to specify the order of code options.\n\n    Returns:\n        types.CodeType: The new code object.\n    '
    (bytecode, linetable) = assemble(instrs, code_options['co_firstlineno'])
    if sys.version_info >= (3, 10):
        code_options['co_linetable'] = linetable
    else:
        code_options['co_lnotab'] = linetable
    code_options['co_code'] = bytecode
    code_options['co_nlocals'] = len(code_options['co_varnames'])
    code_options['co_stacksize'] = stacksize(instrs)
    if sys.version_info >= (3, 11):
        code_options['co_exceptiontable'] = bytes([])
    for (key, val) in code_options.items():
        if isinstance(val, list):
            code_options[key] = tuple(val)
    return types.CodeType(*[code_options[k] for k in keys])

def assemble(instructions: list[Instruction], firstlineno: int) -> tuple[bytes, bytes]:
    if False:
        while True:
            i = 10
    '\n    Assembles a list of instructions into bytecode and lnotab.\n\n    Args:\n        instructions (list[Instruction]): The list of instructions to assemble.\n        firstlineno (int): The starting line number.\n\n    Returns:\n        tuple[bytes, bytes]: The assembled bytecode and lnotab.\n    '
    code = []
    linetable = []
    (calc_linetable, update_cursor) = create_linetable_calculator(firstlineno)
    for instr in instructions:
        if instr.starts_line is not None or sys.version_info >= (3, 11):
            linetable.extend(calc_linetable(instr.starts_line, len(code)))
            update_cursor(instr.starts_line, len(code))
        arg = instr.arg or 0
        code.extend((instr.opcode, arg & 255))
        for _ in range(get_instruction_size(instr) // 2 - 1):
            code.extend((0, 0))
    if sys.version_info >= (3, 11):
        linetable.extend(calc_linetable(None, len(code)))
    elif sys.version_info >= (3, 10):
        linetable.extend(calc_linetable(0, len(code)))
    return (bytes(code), bytes(linetable))

def to_byte(num):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts a negative number to an unsigned byte.\n\n    Args:\n        num (int): The number to convert.\n\n    Returns:\n        int: The converted unsigned byte.\n    '
    if num < 0:
        num += 256
    return num

def get_instruction_size(instr: Instruction) -> int:
    if False:
        i = 10
        return i + 15
    cache_size = 0
    if sys.version_info >= (3, 11):
        cache_size = PYOPCODE_CACHE_SIZE.get(instr.opname, 0)
    return 2 * (cache_size + 1)

def create_linetable_calculator(firstlineno: int):
    if False:
        while True:
            i = 10
    '\n    Creates a line table calculator function.\n\n    Args:\n        firstlineno (int): The starting line number.\n\n    Returns:\n        Callable: The line table calculator function.\n    '
    cur_lineno = firstlineno
    cur_bytecode = 0
    line_offset = 0

    def update_cursor(starts_line: int | None, code_length: int):
        if False:
            i = 10
            return i + 15
        nonlocal cur_lineno, cur_bytecode
        cur_bytecode = code_length
        if starts_line is not None:
            cur_lineno = starts_line

    def calc_lnotab(starts_line: int, code_length: int):
        if False:
            print('Hello World!')
        '\n        Calculates the lnotab for Python 3.8 and 3.9.\n        https://github.com/python/cpython/blob/3.9/Objects/lnotab_notes.txt\n\n        Args:\n            starts_line (int): The line number where the instruction starts.\n            code_length (int): The length of the code.\n\n        Returns:\n            list[int]: The lnotab.\n        '
        nonlocal cur_lineno, cur_bytecode
        line_offset = starts_line - cur_lineno
        byte_offset = code_length - cur_bytecode
        result = []
        while line_offset or byte_offset:
            line_offset_step = min(max(line_offset, -128), 127)
            byte_offset_step = min(max(byte_offset, 0), 255)
            result.extend((byte_offset_step, to_byte(line_offset_step)))
            line_offset -= line_offset_step
            byte_offset -= byte_offset_step
        return result

    def calc_linetable_py310(starts_line: int, code_length: int):
        if False:
            while True:
                i = 10
        '\n        Calculates the linetable for Python 3.10.\n        https://github.com/python/cpython/blob/3.10/Objects/lnotab_notes.txt\n\n        Args:\n            starts_line (int): The line number where the instruction starts.\n            code_length (int): The length of the code.\n\n        Returns:\n            list[int]: The linetable.\n        '
        nonlocal cur_lineno, cur_bytecode, line_offset
        byte_offset = code_length - cur_bytecode
        result = []
        while line_offset or byte_offset:
            line_offset_step = min(max(line_offset, -127), 127)
            byte_offset_step = min(max(byte_offset, 0), 254)
            result.extend((byte_offset_step, to_byte(line_offset_step)))
            line_offset -= line_offset_step
            byte_offset -= byte_offset_step
        line_offset = starts_line - cur_lineno
        return result

    def _encode_varint(num: int):
        if False:
            return 10
        '\n        Encode unsigned integer into variable-length format.\n        '
        continue_flag = 1 << 6
        stop_flag = 0 << 6
        while num >= 64:
            yield (num & 63 | continue_flag)
            num >>= 6
        yield (num | stop_flag)

    def _encode_svarint(num: int):
        if False:
            i = 10
            return i + 15
        '\n        Encode signed integer into variable-length format.\n        '
        unsigned_value = -num << 1 | 1 if num < 0 else num << 1
        yield from _encode_varint(unsigned_value)

    def _encode_bytecode_to_entries_py311(line_offset: int, byte_offset: int):
        if False:
            print('Hello World!')
        if not byte_offset:
            return []
        if 0 < byte_offset <= 8:
            entry_head = 232 | byte_offset - 1
            return [entry_head, *list(_encode_svarint(line_offset))]
        return [*_encode_bytecode_to_entries_py311(line_offset, 8), *_encode_bytecode_to_entries_py311(line_offset, byte_offset - 8)]

    def calc_linetable_py311(starts_line: int | None, code_length: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the linetable for Python 3.11.\n        https://github.com/python/cpython/blob/3.11/Objects/locations.md\n\n        Args:\n            starts_line (int): The line number where the instruction starts.\n            code_length (int): The length of the code.\n\n        Returns:\n            list[int]: The linetable.\n        '
        nonlocal cur_lineno, cur_bytecode
        line_offset = starts_line - cur_lineno if starts_line is not None else 0
        byte_offset = (code_length - cur_bytecode) // 2
        return _encode_bytecode_to_entries_py311(line_offset, byte_offset)
    if sys.version_info >= (3, 11):
        return (calc_linetable_py311, update_cursor)
    elif sys.version_info >= (3, 10):
        return (calc_linetable_py310, update_cursor)
    else:
        return (calc_lnotab, update_cursor)

def compile_exception_table():
    if False:
        print('Hello World!')
    'Compile the exception table, it is used for Python 3.11+.\n    See https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt\n    '
    ...

def stacksize(instructions: list[Instruction]) -> float:
    if False:
        print('Hello World!')
    '\n    Calculates the maximum stack size before each opcode is called.\n\n    Args:\n        instructions (list[Instruction]): The list of instructions.\n\n    Returns:\n        int: The maximum stack size.\n    '
    max_stack = [float('-inf')] * len(instructions)
    max_stack[0] = 0
    queue = []
    queue.append(0)

    def update_stacksize(lasti: int, nexti: int, stack_effect: int):
        if False:
            return 10
        '\n        Updates the maximum stack size.\n\n        Args:\n            lasti (int): The index of the last instruction.\n            nexti (int): The index of the next instruction.\n            stack_effect (int): The effect on the stack size.\n\n        Returns:\n            None\n        '
        old_max = max_stack[nexti]
        max_stack[nexti] = max(max_stack[nexti], max_stack[lasti] + stack_effect)
        if old_max != max_stack[nexti]:
            if nexti not in queue:
                queue.append(nexti)
    while len(queue) > 0:
        idx = queue[0]
        del queue[0]
        instr = instructions[idx]
        opname = instr.opname
        if idx + 1 < len(instructions) and instr.opname not in UNCONDITIONAL_JUMP:
            stack_effect = calc_stack_effect(instr, jump=False)
            update_stacksize(idx, idx + 1, stack_effect)
        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = calc_stack_effect(instr, jump=True)
            target_idx = instructions.index(instr.jump_to)
            update_stacksize(idx, target_idx, stack_effect)
    return max(max_stack)

class PyCodeGen:
    """Helper to create new code object"""

    def __init__(self, frame: types.FrameType, disable_eval_frame: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes a PyCodeGen object.\n\n        Args:\n            frame: The frame to be translated.\n            disable_eval_frame (bool): Whether to disable the evaluation frame. Defaults to False.\n        '
        self._frame = frame
        self._origin_code = frame.f_code
        self._code_options = gen_code_options(self._origin_code)
        self.update_code_name('', is_resumed_fn=False)
        self._f_globals = frame.f_globals
        self._instructions = []
        self.disable_eval_frame = disable_eval_frame
        self.hooks = []
        if self.disable_eval_frame:
            self.gen_disable_eval_frame()

    def insert_prefix_instructions(self):
        if False:
            return 10
        '\n        Insert prefix instructions to the instruction list.\n        In Python 3.11+, we need to insert MAKE_CELL and COPY_FREE_VARS before the\n        first instruction.\n        The implementation is based on cpython implementation:\n        https://github.com/python/cpython/blob/f45ef5edabb1cc0748f3326e7114b8aaa0424392/Python/compile.c#L8177\n        '
        prefixes = []
        if sys.version_info >= (3, 11):
            if self._code_options['co_cellvars']:
                name_map = list(OrderedSet(self._code_options['co_varnames']) | OrderedSet(self._code_options['co_cellvars']))
                for i in self._code_options['co_cellvars']:
                    idx: int = name_map.index(i)
                    prefixes.append(gen_instr('MAKE_CELL', arg=idx, argval=i))
            if self._code_options['co_freevars']:
                n_freevars = len(self._code_options['co_freevars'])
                prefixes.append(gen_instr('COPY_FREE_VARS', arg=n_freevars, argval=n_freevars))
            prefixes.append(gen_instr('RESUME', arg=0, argval=0))
        self._instructions[:] = prefixes + self._instructions

    def update_code_name(self, fn_name, is_resumed_fn):
        if False:
            while True:
                i = 10
        if is_resumed_fn:
            self._code_options['co_name'] = f"${fn_name}@{self._code_options['co_name'][1:]}"
        elif self._code_options['co_name'].startswith('$'):
            self._code_options['co_name'] = f"#{self._code_options['co_name']}"
        elif not self._code_options['co_name'].startswith('#'):
            random_number = int(CODE_NAME_RNG.random() * 100000000)
            self._code_options['co_name'] = f"#{self._code_options['co_name']}_{hex(random_number & 1048575)[2:]:0>5}"

    def gen_pycode(self) -> types.CodeType:
        if False:
            print('Hello World!')
        '\n        Generates a new pycode that is runnable.\n\n        Returns:\n            CodeType: The generated code object.\n        '
        for hook in self.hooks:
            hook()
        self.hooks.clear()
        self.insert_prefix_instructions()
        modify_instrs(self._instructions)
        modify_vars(self._instructions, self._code_options)
        new_code = gen_new_opcode(self._instructions, self._code_options, PYCODE_ATTRIBUTES)
        return new_code

    def gen_resume_fn_at(self, index: int, stack_size: int) -> tuple[None | types.FunctionType, OrderedSet[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Generates a resume function at the specified index in the instruction list.\n\n        Args:\n            index (int): The index in the instruction list to generate the resume function.\n            stack_size (int): The size of the stack. Defaults to 0.\n\n        Returns:\n            tuple: The resume function object and the inputs to the function.\n\n        '
        self._instructions = get_instructions(self._origin_code)
        if self._instructions[index].opname == 'RETURN_VALUE':
            return (None, OrderedSet())
        inputs = analysis_inputs(self._instructions, index)
        fn_name = ResumeFnNameFactory().next()
        stack_arg_str = fn_name + '_stack_{}'
        self._instructions = [gen_instr('LOAD_FAST', argval=stack_arg_str.format(i)) for i in range(stack_size)] + [gen_instr('JUMP_FORWARD', jump_to=self._instructions[index])] + self._instructions
        self._code_options['co_argcount'] = len(inputs) + stack_size
        self._code_options['co_varnames'] = list([stack_arg_str.format(i) for i in range(stack_size)] + list(inputs) + [var_name for var_name in self._code_options['co_varnames'] if var_name not in inputs])
        self.update_code_name(fn_name, is_resumed_fn=True)
        new_code = self.gen_pycode()
        if len(new_code.co_freevars) + len(new_code.co_cellvars) > 0:
            raise FallbackError('Break graph in closure is not support.')
        fn = types.FunctionType(new_code, self._f_globals, new_code.co_name)
        return (fn, inputs)

    @cached_property
    def global_null_variable(self):
        if False:
            return 10
        from .variables.basic import NullVariable
        return NullVariable()

    def gen_disable_eval_frame(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates instructions to disable the evaluation frame.\n        '
        if is_clean_code():
            return
        self.gen_load_object(paddle.framework.core.set_eval_frame, 'paddle_set_eval_frame_fn')
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast('___old_eval_frame')

    def gen_enable_eval_frame(self):
        if False:
            return 10
        '\n        Generates instructions to enable the evaluation frame.\n        '
        if is_clean_code():
            return
        self.gen_load_object(paddle.framework.core.set_eval_frame, 'paddle_set_eval_frame_fn')
        self.gen_load_fast('___old_eval_frame')
        self.gen_call_function(1)
        self.gen_pop_top()

    def gen_outputs_and_return(self, outputs):
        if False:
            i = 10
            return i + 15
        for name in outputs:
            self.gen_load(name)
        self.gen_build_tuple(len(outputs))
        self.gen_return()

    def create_fn_with_inputs(self, inputs: list) -> types.FunctionType:
        if False:
            return 10
        '\n        Creates a function with specific input and output variables.\n\n        Args:\n            inputs (list): The input variables.\n\n        Returns:\n            function: The created function object.\n        '
        self._code_options['co_argcount'] = len(inputs)
        self._code_options['co_varnames'] = list(list(inputs) + [var_name for var_name in self._origin_code.co_varnames if var_name not in inputs])
        fn_name = ResumeFnNameFactory().next()
        self.update_code_name(fn_name, is_resumed_fn=True)
        new_code = self.gen_pycode()
        if len(new_code.co_freevars) + len(new_code.co_cellvars) > 0:
            raise FallbackError('Break graph in closure is not support.')
        fn = types.FunctionType(new_code, self._f_globals, new_code.co_name)
        return fn

    def gen_load_const(self, value: Any):
        if False:
            return 10
        '\n        Generates instructions to load a constant value.\n        '
        if not list_contain_by_id(self._code_options['co_consts'], value):
            self._code_options['co_consts'].append(value)
        idx = list_find_index_by_id(self._code_options['co_consts'], value)
        self._add_instr('LOAD_CONST', arg=idx, argval=value)

    def gen_print_log(self, message):
        if False:
            print('Hello World!')
        'print a log'
        import paddle
        self.gen_load_object(paddle.framework.core.set_eval_frame, 'dbg_set_eval_frame')
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast('old_eval_frame')
        self.gen_load_global('print', push_null=True)
        self.gen_load_const(message)
        self.gen_call_function(1)
        self.gen_pop_top()
        self.gen_load_object(paddle.framework.core.set_eval_frame, 'dbg_set_eval_frame')
        self.gen_load_fast('old_eval_frame')
        self.gen_call_function(1)
        self.gen_pop_top()

    def gen_dbg_function(self, dbg_fun):
        if False:
            return 10
        'debug bytecode helper function.\n        Usage like:\n        def dbg_func():\n            import inspect\n            import dis\n            print("dbg here.")\n            print(locals())\n            frame = inspect.currentframe().f_back\n            code = (inspect.currentframe().f_back.f_code)\n            breakpoint()\n            print(inspect.currentframe().f_back.f_locals[\'y\'])\n\n        self.pycode_gen.gen_dbg_function(dbg_func)\n        '
        import paddle
        self.gen_load_object(paddle.framework.core.set_eval_frame, 'dbg_set_eval_frame')
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast('old_eval_frame')
        self.gen_load_object(dbg_fun, 'dbg1')
        self.gen_call_function(0)
        self.gen_pop_top()
        self.gen_load_object(paddle.framework.core.set_eval_frame, 'dbg_set_eval_frame')
        self.gen_load_fast('old_eval_frame')
        self.gen_call_function(1)
        self.gen_pop_top()

    @property
    def cell_free_storage(self):
        if False:
            return 10
        return self._code_options['co_cellvars'] + self._code_options['co_freevars']

    def gen_load(self, name):
        if False:
            return 10
        if name in self.cell_free_storage:
            self.gen_load_deref(name)
        elif name in self._code_options['co_varnames']:
            self.gen_load_fast(name)
        elif name in self._code_options['co_names']:
            self.gen_load_global(name, push_null=False)
        else:
            raise InnerError(f'Want gen_load, but {name} can not found in code object.')

    def gen_store(self, name, code):
        if False:
            i = 10
            return i + 15
        "\n        Generate the bytecode for storing a variable identified by 'name'\n        in the corresponding symbol table and generate the appropriate\n        store code based on the symbol table analysis.\n\n        Args:\n            name (str): The name of the variable.\n        "
        if name in code.co_freevars + code.co_cellvars:
            self.gen_store_deref(name)
        elif name in code.co_varnames:
            self.gen_store_fast(name)
        elif name in code.co_names:
            self.gen_store_global(name)
        else:
            raise InnerError(f'Want gen_store, but {name} can not found in code object.')

    def gen_load_global(self, name, push_null=False):
        if False:
            while True:
                i = 10
        '\n        Generate the bytecode for loading a global variable.\n\n        Args:\n            name (str): The name of the global variable.\n        '
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        if sys.version_info >= (3, 11):
            idx <<= 1
            if push_null:
                idx |= 1
        self._add_instr('LOAD_GLOBAL', arg=idx, argval=name)

    def gen_load_object(self, obj, obj_name: str, push_null: bool=True):
        if False:
            print('Hello World!')
        '\n        Generate the bytecode for loading an object.\n\n        Args:\n            obj (Any): The object to load.\n            obj_name (str): The name of the object.\n        '
        if obj_name not in self._f_globals:
            self._f_globals[obj_name] = obj
        self.gen_load_global(obj_name, push_null=push_null)

    def gen_load_null_variable(self):
        if False:
            while True:
                i = 10
        '\n        Generate the bytecode for loading a null variable.\n        '
        null_var = self.global_null_variable
        self.gen_load_object(null_var, '___null_var', push_null=False)

    def gen_load_fast(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate the bytecode for loading a local variable.\n\n        Args:\n            name (str): The name of the local variable.\n        '
        if name not in self._code_options['co_varnames']:
            self._code_options['co_varnames'].append(name)
        idx = self._code_options['co_varnames'].index(name)
        self._add_instr('LOAD_FAST', arg=idx, argval=name)

    def gen_load_deref(self, name):
        if False:
            while True:
                i = 10
        if name not in self.cell_free_storage:
            self._code_options['co_freevars'].append(name)
        if sys.version_info >= (3, 11):
            idx = (self._code_options['co_varnames'] + self._code_options['co_freevars']).index(name)
        else:
            idx = self.cell_free_storage.index(name)
        self._add_instr('LOAD_DEREF', arg=idx, argval=name)

    def gen_load_attr(self, name: str):
        if False:
            print('Hello World!')
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        self._add_instr('LOAD_ATTR', arg=idx, argval=name)

    def gen_store_attr(self, name: str):
        if False:
            return 10
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        self._add_instr('STORE_ATTR', arg=idx, argval=name)

    def gen_delete_attr(self, name: str):
        if False:
            i = 10
            return i + 15
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        self._add_instr('DELETE_ATTR', arg=idx, argval=name)

    def gen_load_method(self, name: str):
        if False:
            while True:
                i = 10
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        self._add_instr('LOAD_METHOD', arg=idx, argval=name)

    def gen_delete_global(self, name: str):
        if False:
            print('Hello World!')
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        self._add_instr('DELETE_GLOBAL', arg=idx, argval=name)

    def gen_import_name(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        self._add_instr('IMPORT_NAME', arg=idx, argval=name)

    def gen_push_null(self):
        if False:
            return 10
        if sys.version_info >= (3, 11):
            self._add_instr('PUSH_NULL')
        else:
            self.gen_load_const(0)
            self.gen_load_const(None)
            self.gen_import_name('sys')
            self.gen_store_fast('sys')
            self.gen_load_fast('sys')
            self.gen_load_method('getsizeof')
            self.gen_pop_top()

    def gen_store_fast(self, name):
        if False:
            while True:
                i = 10
        if name not in self._code_options['co_varnames']:
            self._code_options['co_varnames'].append(name)
        idx = self._code_options['co_varnames'].index(name)
        self._add_instr('STORE_FAST', arg=idx, argval=name)

    def gen_store_global(self, name):
        if False:
            while True:
                i = 10
        if name not in self._code_options['co_names']:
            self._code_options['co_names'].append(name)
        idx = self._code_options['co_names'].index(name)
        self._add_instr('STORE_GLOBAL', arg=idx, argval=name)

    def gen_store_deref(self, name):
        if False:
            return 10
        if name not in self.cell_free_storage:
            self._code_options['co_freevars'].append(name)
        if sys.version_info >= (3, 11):
            idx = (self._code_options['co_varnames'] + self._code_options['co_freevars']).index(name)
        else:
            idx = self.cell_free_storage.index(name)
        self._add_instr('STORE_DEREF', arg=idx, argval=name)

    def gen_store_subscr(self):
        if False:
            i = 10
            return i + 15
        self._add_instr('STORE_SUBSCR')

    def gen_subscribe(self):
        if False:
            return 10
        self._add_instr('BINARY_SUBSCR')

    def gen_build_tuple(self, count):
        if False:
            for i in range(10):
                print('nop')
        self._add_instr('BUILD_TUPLE', arg=count, argval=count)

    def gen_build_list(self, count):
        if False:
            for i in range(10):
                print('nop')
        self._add_instr('BUILD_LIST', arg=count, argval=count)

    def gen_build_map(self, count):
        if False:
            return 10
        self._add_instr('BUILD_MAP', arg=count, argval=count)

    def gen_build_slice(self, argc):
        if False:
            while True:
                i = 10
        self._add_instr('BUILD_SLICE', arg=argc, argval=argc)

    def gen_unpack_sequence(self, count):
        if False:
            for i in range(10):
                print('nop')
        self._add_instr('UNPACK_SEQUENCE', arg=count, argval=count)

    def gen_call_function(self, argc=0):
        if False:
            i = 10
            return i + 15
        if sys.version_info >= (3, 11):
            self._add_instr('PRECALL', arg=argc, argval=argc)
            self._add_instr('CALL', arg=argc, argval=argc)
        else:
            self._add_instr('CALL_FUNCTION', arg=argc, argval=argc)

    def gen_call_function_ex(self, has_kwargs):
        if False:
            return 10
        flag = 0
        if has_kwargs:
            flag |= CALL_FUNCTION_EX_FLAG.CFE_HAS_KWARGS
        self._add_instr('CALL_FUNCTION_EX', arg=flag, argval=flag)

    def gen_call_method(self, argc=0):
        if False:
            while True:
                i = 10
        if sys.version_info >= (3, 11):
            self._add_instr('PRECALL', arg=argc, argval=argc)
            self._add_instr('CALL', arg=argc, argval=argc)
        else:
            self._add_instr('CALL_METHOD', arg=argc, argval=argc)

    def gen_kw_names(self, kw_names: tuple[str, ...] | None):
        if False:
            i = 10
            return i + 15
        if kw_names is None:
            return
        if sys.version_info < (3, 11):
            raise InnerError('gen_kw_names is not supported before python3.11')
        if kw_names not in self._code_options['co_consts']:
            self._code_options['co_consts'].append(kw_names)
        idx = self._code_options['co_consts'].index(kw_names)
        self._add_instr('KW_NAMES', arg=idx, argval=kw_names)

    def gen_pop_top(self):
        if False:
            for i in range(10):
                print('nop')
        self._add_instr('POP_TOP')

    def gen_rot_n(self, n):
        if False:
            while True:
                i = 10
        if n <= 1:
            return
        if sys.version_info >= (3, 11):
            for i in range(n, 1, -1):
                self._add_instr('SWAP', arg=i)
        elif sys.version_info >= (3, 10):
            self._add_instr('ROT_N', arg=n)
        elif n <= 4:
            self._add_instr('ROT_' + ['TWO', 'THREE', 'FOUR'][n - 2])
        else:

            def rot_n_fn(n):
                if False:
                    return 10
                vars = [f'var{i}' for i in range(n)]
                rotated = reversed(vars[-1:] + vars[:-1])
                fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
                fn = no_eval_frame(fn)
                fn.__name__ = f'rot_{n}_fn'
                return fn
            self.gen_build_tuple(n)
            self.gen_load_const(rot_n_fn(n))
            self.gen_rot_n(2)
            self._add_instr('CALL_FUNCTION_EX', arg=0)
            self.gen_unpack_sequence(n)

    def gen_shift_n(self, s: int, n: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate the bytecode for shifting the stack.\n\n        Args:\n            s (int): Steps to shift.\n            n (int): The number of elements to shift.\n        '
        if s == 0 or n <= 1:
            return
        if abs(s) > n // 2:
            new_s = s - n if s > 0 else s + n
            self.gen_shift_n(new_s, n)
            return
        if s > 0:
            if s == 1:
                self.gen_rot_n(n)
            else:
                self.gen_rot_n(n)
                self.gen_shift_n(s - 1, n)
        elif sys.version_info >= (3, 11):
            if s == -1:
                for i in range(2, n + 1):
                    self._add_instr('SWAP', arg=i)
            else:
                self.gen_shift_n(-1, n)
                self.gen_shift_n(s + 1, n)
        else:
            raise NotImplementedError('shift_n is not supported before python3.11')

    def gen_swap(self, n):
        if False:
            i = 10
            return i + 15
        if sys.version_info >= (3, 11):
            self._add_instr('SWAP', arg=n)
        else:
            raise NotImplementedError('swap is not supported before python3.11')

    def gen_jump(self, jump_to: Instruction | None=None, *, direction: JumpDirection=JumpDirection.FORWARD) -> Instruction:
        if False:
            i = 10
            return i + 15
        if sys.version_info >= (3, 11):
            return self._add_instr(f'JUMP_{direction.value}', jump_to=jump_to)
        else:
            return self._add_instr('JUMP_ABSOLUTE', jump_to=jump_to)

    def gen_pop_jump(self, jump_to: Instruction | None=None, *, direction: JumpDirection=JumpDirection.FORWARD, suffix: PopJumpCond=PopJumpCond.NONE) -> Instruction:
        if False:
            for i in range(10):
                print('nop')
        if sys.version_info >= (3, 11):
            return self._add_instr(f'POP_JUMP_{direction.value}_IF_{suffix.value}', jump_to=jump_to)
        else:
            return self._add_instr(f'POP_JUMP_IF_{suffix.value}', jump_to=jump_to)

    def gen_return(self):
        if False:
            while True:
                i = 10
        self._add_instr('RETURN_VALUE')

    def gen_get_iter(self):
        if False:
            return 10
        self._add_instr('GET_ITER')

    def _add_instr(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        instr = gen_instr(*args, **kwargs)
        self._instructions.append(instr)
        return instr

    def _insert_instr(self, index, *args, **kwargs):
        if False:
            while True:
                i = 10
        instr = gen_instr(*args, **kwargs)
        self._instructions.insert(index, instr)

    def pprint(self):
        if False:
            i = 10
            return i + 15
        print('\n'.join(instrs_info(self._instructions)))

    def extend_instrs(self, instrs):
        if False:
            print('Hello World!')
        self._instructions.extend(instrs)

    def pop_instr(self):
        if False:
            return 10
        self._instructions.pop()

    def replace_null_variable(self):
        if False:
            return 10
        '\n        Replace all NullVariables in the bytecode.\n\n        Returns:\n            Optional[Tuple[Any, Callable]]: The new code object and its guard function, or None if no dummy variables are found.\n        '
        from .variables.basic import NullVariable
        instructions = get_instructions(self._origin_code)
        has_null_variable = False
        for instr in instructions:
            if instr.opname == 'LOAD_FAST' and instr.argval in self._frame.f_locals.keys() and isinstance(self._frame.f_locals[instr.argval], NullVariable):
                has_null_variable = True
                self._frame.f_locals[instr.argval].reconstruct(self)
            elif instr.opname == 'LOAD_GLOBAL' and instr.argval in self._frame.f_globals.keys() and isinstance(self._frame.f_globals[instr.argval], NullVariable):
                has_null_variable = True
                self._frame.f_globals[instr.argval].reconstruct(self)
            else:
                self.extend_instrs([instr])
        if has_null_variable:
            new_code = self.gen_pycode()
            return new_code
        else:
            return None