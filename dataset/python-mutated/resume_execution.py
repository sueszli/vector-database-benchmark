import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import create_call_function, create_call_method, create_dup_top, create_instruction, create_jump_absolute, Instruction, InstructionExnTabEntry, transform_code_object, unique_id
from .utils import ExactWeakKeyDictionary
CO_OPTIMIZED = 1
CO_NEWLOCALS = 2
CO_VARARGS = 4
CO_VARKEYWORDS = 8
CO_NESTED = 16
CO_GENERATOR = 32
CO_NOFREE = 64
CO_COROUTINE = 128
CO_ITERABLE_COROUTINE = 256
CO_ASYNC_GENERATOR = 512

@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int
    target_values: Optional[Tuple[Any, ...]] = None

    def try_except(self, code_options, cleanup: List[Instruction]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Codegen based off of:\n        load args\n        enter context\n        try:\n            (rest)\n        finally:\n            exit context\n        '
        load_args = []
        if self.target_values:
            load_args = [create_instruction('LOAD_CONST', argval=val) for val in self.target_values]
        ctx_name = unique_id(f'___context_manager_{self.stack_index}')
        if ctx_name not in code_options['co_varnames']:
            code_options['co_varnames'] += (ctx_name,)
        for name in ['__enter__', '__exit__']:
            if name not in code_options['co_names']:
                code_options['co_names'] += (name,)
        except_jump_target = create_instruction('NOP' if sys.version_info < (3, 11) else 'PUSH_EXC_INFO')
        cleanup_complete_jump_target = create_instruction('NOP')
        setup_finally = [*load_args, *create_call_function(len(load_args), True), create_instruction('STORE_FAST', argval=ctx_name), create_instruction('LOAD_FAST', argval=ctx_name), create_instruction('LOAD_METHOD', argval='__enter__'), *create_call_method(0), create_instruction('POP_TOP')]
        if sys.version_info < (3, 11):
            setup_finally.append(create_instruction('SETUP_FINALLY', target=except_jump_target))
        else:
            exn_tab_begin = create_instruction('NOP')
            exn_tab_end = create_instruction('NOP')
            exn_tab_begin.exn_tab_entry = InstructionExnTabEntry(exn_tab_begin, exn_tab_end, except_jump_target, self.stack_index + 1, False)
            setup_finally.append(exn_tab_begin)

        def create_reset():
            if False:
                return 10
            return [create_instruction('LOAD_FAST', argval=ctx_name), create_instruction('LOAD_METHOD', argval='__exit__'), create_instruction('LOAD_CONST', argval=None), create_dup_top(), create_dup_top(), *create_call_method(3), create_instruction('POP_TOP')]
        if sys.version_info < (3, 9):
            epilogue = [create_instruction('POP_BLOCK'), create_instruction('BEGIN_FINALLY'), except_jump_target, *create_reset(), create_instruction('END_FINALLY')]
        elif sys.version_info < (3, 11):
            epilogue = [create_instruction('POP_BLOCK'), *create_reset(), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), except_jump_target, *create_reset(), create_instruction('RERAISE'), cleanup_complete_jump_target]
        else:
            finally_exn_tab_end = create_instruction('RERAISE', arg=0)
            finally_exn_tab_target = create_instruction('COPY', arg=3)
            except_jump_target.exn_tab_entry = InstructionExnTabEntry(except_jump_target, finally_exn_tab_end, finally_exn_tab_target, self.stack_index + 2, True)
            epilogue = [exn_tab_end, *create_reset(), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), except_jump_target, *create_reset(), finally_exn_tab_end, finally_exn_tab_target, create_instruction('POP_EXCEPT'), create_instruction('RERAISE', arg=1), cleanup_complete_jump_target]
        cleanup[:] = epilogue + cleanup
        return setup_finally

    def __call__(self, code_options, cleanup):
        if False:
            return 10
        '\n        Codegen based off of:\n        with ctx(args):\n            (rest)\n        '
        load_args = []
        if self.target_values:
            load_args = [create_instruction('LOAD_CONST', argval=val) for val in self.target_values]
        if sys.version_info < (3, 9):
            with_cleanup_start = create_instruction('WITH_CLEANUP_START')
            begin_finally = create_instruction('BEGIN_FINALLY')
            cleanup[:] = [create_instruction('POP_BLOCK'), begin_finally, with_cleanup_start, create_instruction('WITH_CLEANUP_FINISH'), create_instruction('END_FINALLY')] + cleanup
            return ([*load_args, create_instruction('CALL_FUNCTION', arg=len(load_args)), create_instruction('SETUP_WITH', target=with_cleanup_start), create_instruction('POP_TOP')], None)
        elif sys.version_info < (3, 11):
            with_except_start = create_instruction('WITH_EXCEPT_START')
            pop_top_after_with_except_start = create_instruction('POP_TOP')
            cleanup_complete_jump_target = create_instruction('NOP')
            cleanup[:] = [create_instruction('POP_BLOCK'), create_instruction('LOAD_CONST', argval=None), create_instruction('DUP_TOP'), create_instruction('DUP_TOP'), create_instruction('CALL_FUNCTION', arg=3), create_instruction('POP_TOP'), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), with_except_start, create_instruction('POP_JUMP_IF_TRUE', target=pop_top_after_with_except_start), create_instruction('RERAISE'), pop_top_after_with_except_start, create_instruction('POP_TOP'), create_instruction('POP_TOP'), create_instruction('POP_EXCEPT'), create_instruction('POP_TOP'), cleanup_complete_jump_target] + cleanup
            return ([*load_args, create_instruction('CALL_FUNCTION', arg=len(load_args)), create_instruction('SETUP_WITH', target=with_except_start), create_instruction('POP_TOP')], None)
        else:
            pop_top_after_with_except_start = create_instruction('POP_TOP')
            cleanup_complete_jump_target = create_instruction('NOP')

            def create_load_none():
                if False:
                    print('Hello World!')
                return create_instruction('LOAD_CONST', argval=None)
            exn_tab_1_begin = create_instruction('POP_TOP')
            exn_tab_1_end = create_instruction('NOP')
            exn_tab_1_target = create_instruction('PUSH_EXC_INFO')
            exn_tab_2_end = create_instruction('RERAISE', arg=2)
            exn_tab_2_target = create_instruction('COPY', arg=3)
            exn_tab_1_begin.exn_tab_entry = InstructionExnTabEntry(exn_tab_1_begin, exn_tab_1_end, exn_tab_1_target, self.stack_index + 1, True)
            exn_tab_1_target.exn_tab_entry = InstructionExnTabEntry(exn_tab_1_target, exn_tab_2_end, exn_tab_2_target, self.stack_index + 3, True)
            pop_top_after_with_except_start.exn_tab_entry = InstructionExnTabEntry(pop_top_after_with_except_start, pop_top_after_with_except_start, exn_tab_2_target, self.stack_index + 3, True)
            cleanup[:] = [exn_tab_1_end, create_load_none(), create_load_none(), create_load_none(), *create_call_function(2, False), create_instruction('POP_TOP'), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), exn_tab_1_target, create_instruction('WITH_EXCEPT_START'), create_instruction('POP_JUMP_FORWARD_IF_TRUE', target=pop_top_after_with_except_start), exn_tab_2_end, exn_tab_2_target, create_instruction('POP_EXCEPT'), create_instruction('RERAISE', arg=1), pop_top_after_with_except_start, create_instruction('POP_EXCEPT'), create_instruction('POP_TOP'), create_instruction('POP_TOP'), cleanup_complete_jump_target] + cleanup
            return ([*load_args, *create_call_function(len(load_args), True), create_instruction('BEFORE_WITH'), exn_tab_1_begin], exn_tab_1_target)

@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: List[Instruction] = dataclasses.field(default_factory=list)
    prefix_block_target_offset_remap: List[int] = dataclasses.field(default_factory=list)
    block_target_offset_remap: Optional[Dict[int, int]] = None

def _filter_iter(l1, l2, cond):
    if False:
        while True:
            i = 10
    '\n    Two-pointer conditional filter.\n    e.g. _filter_iter(insts, sorted_offsets, lambda i, o: i.offset == o)\n    returns the instructions with offsets in sorted_offsets\n    '
    it = iter(l2)
    res = []
    try:
        cur = next(it)
        for val in l1:
            if cond(val, cur):
                res.append(val)
                cur = next(it)
    except StopIteration:
        pass
    return res

class ContinueExecutionCache:
    cache = ExactWeakKeyDictionary()
    generated_code_metadata = ExactWeakKeyDictionary()

    @classmethod
    def lookup(cls, code, lineno, *key):
        if False:
            for i in range(10):
                print('nop')
        if code not in cls.cache:
            cls.cache[code] = dict()
        key = tuple(key)
        if key not in cls.cache[code]:
            cls.cache[code][key] = cls.generate(code, lineno, *key)
        return cls.cache[code][key]

    @classmethod
    def generate(cls, code, lineno, offset: int, setup_fn_target_offsets: Tuple[int], nstack: int, argnames: Tuple[str], setup_fns: Tuple[ReenterWith], null_idxes: Tuple[int]) -> types.CodeType:
        if False:
            i = 10
            return i + 15
        assert offset is not None
        assert not code.co_flags & (CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR)
        assert code.co_flags & CO_OPTIMIZED
        if code in ContinueExecutionCache.generated_code_metadata:
            return cls.generate_based_on_original_code_object(code, lineno, offset, setup_fn_target_offsets, nstack, argnames, setup_fns, null_idxes)
        is_py311_plus = sys.version_info >= (3, 11)
        meta = ResumeFunctionMetadata(code)

        def update(instructions: List[Instruction], code_options: Dict[str, Any]):
            if False:
                return 10
            meta.instructions = copy.deepcopy(instructions)
            args = [f'___stack{i}' for i in range(nstack)]
            args.extend((v for v in argnames if v not in args))
            freevars = tuple(code_options['co_cellvars'] or []) + tuple(code_options['co_freevars'] or [])
            code_options['co_name'] = f"resume_in_{code_options['co_name']}"
            if is_py311_plus:
                code_options['co_qualname'] = f"resume_in_{code_options['co_qualname']}"
            code_options['co_firstlineno'] = lineno
            code_options['co_cellvars'] = tuple()
            code_options['co_freevars'] = freevars
            code_options['co_argcount'] = len(args)
            code_options['co_posonlyargcount'] = 0
            code_options['co_kwonlyargcount'] = 0
            code_options['co_varnames'] = tuple(args + [v for v in code_options['co_varnames'] if v not in args])
            code_options['co_flags'] = code_options['co_flags'] & ~(CO_VARARGS | CO_VARKEYWORDS)
            target = next((i for i in instructions if i.offset == offset))
            prefix = []
            if is_py311_plus:
                if freevars:
                    prefix.append(create_instruction('COPY_FREE_VARS', arg=len(freevars)))
                prefix.append(create_instruction('RESUME', arg=0))
            cleanup: List[Instruction] = []
            hooks = {fn.stack_index: fn for fn in setup_fns}
            hook_target_offsets = {fn.stack_index: setup_fn_target_offsets[i] for (i, fn) in enumerate(setup_fns)}
            offset_to_inst = {inst.offset: inst for inst in instructions}
            old_hook_target_remap = {}
            null_idxes_i = 0
            for i in range(nstack):
                while null_idxes_i < len(null_idxes) and null_idxes[null_idxes_i] == i + null_idxes_i:
                    prefix.append(create_instruction('PUSH_NULL'))
                    null_idxes_i += 1
                prefix.append(create_instruction('LOAD_FAST', argval=f'___stack{i}'))
                if i in hooks:
                    hook = hooks.pop(i)
                    (hook_insts, exn_target) = hook(code_options, cleanup)
                    prefix.extend(hook_insts)
                    if is_py311_plus:
                        hook_target_offset = hook_target_offsets.pop(i)
                        old_hook_target = offset_to_inst[hook_target_offset]
                        meta.prefix_block_target_offset_remap.append(hook_target_offset)
                        old_hook_target_remap[old_hook_target] = exn_target
            if is_py311_plus:
                meta.prefix_block_target_offset_remap = list(reversed(meta.prefix_block_target_offset_remap))
            assert not hooks
            prefix.append(create_jump_absolute(target))
            for inst in instructions:
                if inst.offset == target.offset:
                    break
                inst.starts_line = None
                if sys.version_info >= (3, 11):
                    inst.positions = None
            if cleanup:
                prefix.extend(cleanup)
                prefix.extend(cls.unreachable_codes(code_options))
            if old_hook_target_remap:
                assert is_py311_plus
                for inst in instructions:
                    if inst.exn_tab_entry and inst.exn_tab_entry.target in old_hook_target_remap:
                        inst.exn_tab_entry.target = old_hook_target_remap[inst.exn_tab_entry.target]
            instructions[:] = prefix + instructions
        new_code = transform_code_object(code, update)
        ContinueExecutionCache.generated_code_metadata[new_code] = meta
        return new_code

    @staticmethod
    def unreachable_codes(code_options) -> List[Instruction]:
        if False:
            while True:
                i = 10
        'Codegen a `raise None` to make analysis work for unreachable code'
        return [create_instruction('LOAD_CONST', argval=None), create_instruction('RAISE_VARARGS', arg=1)]

    @classmethod
    def generate_based_on_original_code_object(cls, code, lineno, offset: int, setup_fn_target_offsets: Tuple[int, ...], *args):
        if False:
            while True:
                i = 10
        '\n        This handles the case of generating a resume into code generated\n        to resume something else.  We want to always generate starting\n        from the original code object so that if control flow paths\n        converge we only generated 1 resume function (rather than 2^n\n        resume functions).\n        '
        meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[code]
        new_offset = None

        def find_new_offset(instructions: List[Instruction], code_options: Dict[str, Any]):
            if False:
                print('Hello World!')
            nonlocal new_offset
            (target,) = (i for i in instructions if i.offset == offset)
            (new_target,) = (i2 for (i1, i2) in zip(reversed(instructions), reversed(meta.instructions)) if i1 is target)
            assert target.opcode == new_target.opcode
            new_offset = new_target.offset
        transform_code_object(code, find_new_offset)
        if sys.version_info >= (3, 11):
            if not meta.block_target_offset_remap:
                block_target_offset_remap = meta.block_target_offset_remap = {}

                def remap_block_offsets(instructions: List[Instruction], code_options: Dict[str, Any]):
                    if False:
                        return 10
                    prefix_blocks: List[Instruction] = []
                    for inst in instructions:
                        if len(prefix_blocks) == len(meta.prefix_block_target_offset_remap):
                            break
                        if inst.opname == 'PUSH_EXC_INFO':
                            prefix_blocks.append(inst)
                    for (inst, o) in zip(prefix_blocks, meta.prefix_block_target_offset_remap):
                        block_target_offset_remap[cast(int, inst.offset)] = o
                    old_start_offset = cast(int, prefix_blocks[-1].offset) if prefix_blocks else -1
                    old_inst_offsets = sorted((n for n in setup_fn_target_offsets if n > old_start_offset))
                    targets = _filter_iter(instructions, old_inst_offsets, lambda inst, o: inst.offset == o)
                    new_targets = _filter_iter(zip(reversed(instructions), reversed(meta.instructions)), targets, lambda v1, v2: v1[0] is v2)
                    for (new, old) in zip(new_targets, targets):
                        block_target_offset_remap[old.offset] = new[1].offset
                transform_code_object(code, remap_block_offsets)
            setup_fn_target_offsets = tuple((block_target_offset_remap[n] for n in setup_fn_target_offsets))
        return ContinueExecutionCache.lookup(meta.code, lineno, new_offset, setup_fn_target_offsets, *args)
'\n# partially finished support for with statements\n\ndef convert_locals_to_cells(\n        instructions: List[Instruction],\n        code_options: Dict[str, Any]):\n\n    code_options["co_cellvars"] = tuple(\n        var\n        for var in code_options["co_varnames"]\n        if var not in code_options["co_freevars"]\n        and not var.startswith("___stack")\n    )\n    cell_and_free = code_options["co_cellvars"] + code_options["co_freevars"]\n    for inst in instructions:\n        if str(inst.argval).startswith("___stack"):\n            continue\n        elif inst.opname == "LOAD_FAST":\n            inst.opname = "LOAD_DEREF"\n        elif inst.opname == "STORE_FAST":\n            inst.opname = "STORE_DEREF"\n        elif inst.opname == "DELETE_FAST":\n            inst.opname = "DELETE_DEREF"\n        else:\n            continue\n        inst.opcode = dis.opmap[inst.opname]\n        assert inst.argval in cell_and_free, inst.argval\n        inst.arg = cell_and_free.index(inst.argval)\n\ndef patch_setup_with(\n    instructions: List[Instruction],\n    code_options: Dict[str, Any]\n):\n    nonlocal need_skip\n    need_skip = True\n    target_index = next(\n        idx for idx, i in enumerate(instructions) if i.offset == offset\n    )\n    assert instructions[target_index].opname == "SETUP_WITH"\n    convert_locals_to_cells(instructions, code_options)\n\n    stack_depth_before = nstack + stack_effect(instructions[target_index].opcode,\n                                               instructions[target_index].arg)\n\n    inside_with = []\n    inside_with_resume_at = None\n    stack_depth = stack_depth_before\n    idx = target_index + 1\n    for idx in range(idx, len(instructions)):\n        inst = instructions[idx]\n        if inst.opname == "BEGIN_FINALLY":\n            inside_with_resume_at = inst\n            break\n        elif inst.target is not None:\n            unimplemented("jump from with not supported")\n        elif inst.opname in ("BEGIN_FINALLY", "WITH_CLEANUP_START", "WITH_CLEANUP_FINISH", "END_FINALLY",\n                             "POP_FINALLY", "POP_EXCEPT",\n                             "POP_BLOCK", "END_ASYNC_FOR"):\n            unimplemented("block ops not supported")\n        inside_with.append(inst)\n        stack_depth += stack_effect(inst.opcode, inst.arg)\n    assert inside_with_resume_at\n\n    instructions = [\n        create_instruction("LOAD_FAST", f"___stack{i}") for i in range(nstack)\n    ] + [\n        create_instruction("SETUP_WITH", target=instructions[target_index].target)\n        ... call the function ...\n        unpack_tuple\n    ] + [\n        create_instruction("JUMP_ABSOLUTE", target=inside_with_resume_at)\n    ]\n'