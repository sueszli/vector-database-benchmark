from vyper.codegen.core import STORE, add_ofst, get_dyn_array_count, get_element_ptr, is_tuple_like, make_setter, zero_pad
from vyper.codegen.ir_node import IRnode
from vyper.evm.address_space import MEMORY
from vyper.exceptions import CompilerPanic
from vyper.semantics.types import DArrayT, SArrayT, _BytestringT
from vyper.semantics.types.shortcuts import UINT256_T

def _is_complex_type(typ):
    if False:
        while True:
            i = 10
    return is_tuple_like(typ) or isinstance(typ, SArrayT)

def _deconstruct_complex_type(ir_node):
    if False:
        return 10
    ir_t = ir_node.typ
    assert _is_complex_type(ir_t)
    if is_tuple_like(ir_t):
        ks = ir_t.tuple_keys()
    else:
        ks = [IRnode.from_list(i, UINT256_T) for i in range(ir_t.count)]
    ret = []
    for k in ks:
        ret.append(get_element_ptr(ir_node, k, array_bounds_check=False))
    return ret

def _encode_child_helper(buf, child, static_ofst, dyn_ofst, context):
    if False:
        i = 10
        return i + 15
    child_abi_t = child.typ.abi_type
    static_loc = add_ofst(IRnode.from_list(buf), static_ofst)
    ret = ['seq']
    if not child_abi_t.is_dynamic():
        _bufsz = child_abi_t.size_bound()
        ret.append(abi_encode(static_loc, child, context, _bufsz, returns_len=False))
    else:
        ret.append(['mstore', static_loc, dyn_ofst])
        child_dst = ['add', buf, dyn_ofst]
        _bufsz = child_abi_t.size_bound()
        child_len = abi_encode(child_dst, child, context, _bufsz, returns_len=True)
        ret.append(['set', dyn_ofst, ['add', dyn_ofst, child_len]])
    return ret

def _encode_dyn_array_helper(dst, ir_node, context):
    if False:
        return 10
    if ir_node.value == 'multi':
        buf = context.new_internal_variable(dst.typ)
        buf = IRnode.from_list(buf, typ=dst.typ, location=MEMORY)
        _bufsz = dst.typ.abi_type.size_bound()
        return ['seq', make_setter(buf, ir_node), ['set', 'dyn_ofst', abi_encode(dst, buf, context, _bufsz, returns_len=True)]]
    subtyp = ir_node.typ.value_type
    child_abi_t = subtyp.abi_type
    ret = ['seq']
    len_ = get_dyn_array_count(ir_node)
    with len_.cache_when_complex('len') as (b, len_):
        ret.append(STORE(dst, len_))
        t = UINT256_T
        i = IRnode.from_list(context.fresh_varname('ix'), typ=t)
        child_location = get_element_ptr(ir_node, i, array_bounds_check=False)
        dst = add_ofst(dst, 32)
        static_elem_size = child_abi_t.embedded_static_size()
        static_ofst = ['mul', i, static_elem_size]
        loop_body = _encode_child_helper(dst, child_location, static_ofst, 'dyn_child_ofst', context)
        loop = ['repeat', i, 0, len_, ir_node.typ.count, loop_body]
        x = ['seq', loop, 'dyn_child_ofst']
        start_dyn_ofst = ['mul', len_, static_elem_size]
        run_children = ['with', 'dyn_child_ofst', start_dyn_ofst, x]
        new_dyn_ofst = ['add', 'dyn_ofst', run_children]
        new_dyn_ofst = ['add', 32, new_dyn_ofst]
        ret.append(['set', 'dyn_ofst', new_dyn_ofst])
        return b.resolve(ret)

def abi_encoding_matches_vyper(typ):
    if False:
        for i in range(10):
            print('nop')
    "\n    returns True if the ABI encoding matches vyper's memory encoding of\n    a type, otherwise False\n    "
    return not typ.abi_type.is_dynamic()

def abi_encode(dst, ir_node, context, bufsz, returns_len=False):
    if False:
        return 10
    dst = IRnode.from_list(dst, typ=ir_node.typ, location=MEMORY)
    abi_t = dst.typ.abi_type
    size_bound = abi_t.size_bound()
    assert isinstance(bufsz, int)
    if bufsz < size_bound:
        raise CompilerPanic('buffer provided to abi_encode not large enough')
    if size_bound < dst.typ.memory_bytes_required:
        raise CompilerPanic('Bad ABI size calc')
    annotation = f'abi_encode {ir_node.typ}'
    ir_ret = ['seq']
    if abi_encoding_matches_vyper(ir_node.typ):
        ir_ret.append(make_setter(dst, ir_node))
        if returns_len:
            assert abi_t.embedded_static_size() == ir_node.typ.memory_bytes_required
            ir_ret.append(abi_t.embedded_static_size())
        return IRnode.from_list(ir_ret, annotation=annotation)
    with ir_node.cache_when_complex('to_encode') as (b1, ir_node), dst.cache_when_complex('dst') as (b2, dst):
        dyn_ofst = 'dyn_ofst'
        if ir_node.typ._is_prim_word:
            ir_ret.append(make_setter(dst, ir_node))
        elif isinstance(ir_node.typ, _BytestringT):
            ir_ret.append(make_setter(dst, ir_node))
            ir_ret.append(zero_pad(dst))
        elif isinstance(ir_node.typ, DArrayT):
            ir_ret.append(_encode_dyn_array_helper(dst, ir_node, context))
        elif _is_complex_type(ir_node.typ):
            static_ofst = 0
            elems = _deconstruct_complex_type(ir_node)
            for e in elems:
                encode_ir = _encode_child_helper(dst, e, static_ofst, dyn_ofst, context)
                ir_ret.extend(encode_ir)
                static_ofst += e.typ.abi_type.embedded_static_size()
        else:
            raise CompilerPanic(f'unencodable type: {ir_node.typ}')
        if returns_len:
            if not abi_t.is_dynamic():
                ir_ret.append(abi_t.embedded_static_size())
            elif isinstance(ir_node.typ, _BytestringT):
                calc_len = ['ceil32', ['add', 32, ['mload', dst]]]
                ir_ret.append(calc_len)
            elif abi_t.is_complex_type():
                ir_ret.append('dyn_ofst')
            else:
                raise CompilerPanic(f'unknown type {ir_node.typ}')
        if abi_t.is_dynamic() and abi_t.is_complex_type():
            dyn_section_start = abi_t.static_size()
            ir_ret = ['with', dyn_ofst, dyn_section_start, ir_ret]
        else:
            pass
        return b1.resolve(b2.resolve(IRnode.from_list(ir_ret, annotation=annotation)))