import logging
import typing
from collections import OrderedDict
import paddle
from paddle.base import framework
from paddle.base.core import ops_contain_none, prim_config
from paddle.base.framework import Operator, default_main_program
from paddle.incubate.autograd.utils import as_tensors
from .composite_rules import _composite
from .primops import add, fill_const
from .primreg import lookup_composite, lookup_orig2prim, lookup_prim2orig, op_position_inputs, op_position_output
from .primrules import _jvp, _orig2prim, _prim2orig, _transpose
from .utils import flatten, flatten_and_remove_none, get_input_var_list, get_output_var_list, map_output_for_composite, prepare_python_api_arguments

def topo_path(xs, ys, block=None):
    if False:
        while True:
            i = 10
    'Returns the list of ops on the path from `xs` to `ys` in topological\n    order.\n\n    TODO(Tongxin): supporting control flow and nested blocks.\n    Args:\n        xs: a list|tuple of vars as source\n        ys: a list|tuple of vars as sink\n        block: the program block containing the path, optional\n    Returns:\n        (path, unused_xs, unreached_ys): a tuple comprised of the resulting op\n        path, the unused variables in `xs`, and the unreached variables in `ys`\n    '
    block = default_main_program().current_block() if block is None else block
    path = []
    backpath = []
    reached_vars = OrderedDict()
    used_vars = OrderedDict()
    for x in xs:
        assert x is None or x.block == block, 'x is not None and x.block != block'
        reached_vars[id(x)] = x
    reaching = lambda op: any((id(v) in reached_vars for v in flatten_and_remove_none(get_input_var_list(op))))
    for op in block.ops:
        if reaching(op):
            path.append(op)
            for var in flatten_and_remove_none(get_output_var_list(op)):
                reached_vars[id(var)] = var
    used_vars = OrderedDict(((id(y), y) for y in ys if id(y) in reached_vars))
    back_reaching = lambda op: any((id(out) in used_vars for out in flatten_and_remove_none(get_output_var_list(op))))
    for op in reversed(path):
        if back_reaching(op):
            backpath.append(op)
            for var in flatten_and_remove_none(get_input_var_list(op)):
                used_vars[id(var)] = var
    unused_xs = [x for x in xs if id(x) not in used_vars]
    unreached_ys = [y for y in ys if id(y) not in reached_vars]
    return (list(reversed(backpath)), unused_xs, unreached_ys)

def output_vars_on_path(path):
    if False:
        for i in range(10):
            print('nop')
    'Returns the output variables of all the ops on the path from `xs`\n    to `ys`.\n\n    Args:\n        path: a list of ops on which to find the output variables\n\n    Returns:\n        vars: the output vars\n    '
    vars = OrderedDict()
    for op in path:
        for out in flatten_and_remove_none(get_output_var_list(op)):
            vars[id(out)] = out
    return vars

class VarMap:
    """A general map data structure for linking variables to variables.

    An example is linking variables to their gradients.
    """
    __slots__ = ['name', 'varset', 'tab']

    def __init__(self, name, varset):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.varset = varset
        self.tab = OrderedDict()

    def add(self, key_var, value_var):
        if False:
            for i in range(10):
                print('nop')
        self.tab[id(key_var)] = id(value_var)

    def add_rec(self, key_vars, value_vars):
        if False:
            return 10
        if value_vars is None:
            return
        if isinstance(key_vars, paddle.base.framework.Variable):
            if not isinstance(value_vars, paddle.base.framework.Variable):
                raise TypeError(f'value_vars must be Variable, but got {type(value_vars)}')
            self.tab[id(key_vars)] = id(value_vars)
        else:
            assert len(key_vars) == len(value_vars), f'len(key_vars) shoule be equal to len(value_vars), but len(key_vars)={len(key_vars)} and len(value_vars)={len(value_vars)}.'
            for (key_var, value_var) in zip(key_vars, value_vars):
                self.add_rec(key_var, value_var)

    def lookup(self, key_var):
        if False:
            for i in range(10):
                print('nop')
        value_id = self.tab.get(id(key_var))
        if value_id is not None:
            return self.varset.get(value_id)
        else:
            return None

    def delete(self, key_var):
        if False:
            while True:
                i = 10
        varid = id(key_var)
        if varid in self.tab:
            del self.tab[id(key_var)]

    def delete_keyvars(self, key_vars):
        if False:
            return 10
        for var in key_vars:
            varid = id(var)
            if varid in self.tab:
                del self.tab[varid]

    def delete_valuevars(self, value_vars):
        if False:
            for i in range(10):
                print('nop')
        ids = [id(v) for v in value_vars]
        keys = [k for (k, v) in self.tab.items() if v in ids]
        for k in keys:
            del self.tab[k]

    def contain_var(self, key_var):
        if False:
            return 10
        return self.tab.__contains__(id(key_var))

    def contain_value(self, value_var):
        if False:
            for i in range(10):
                print('nop')
        return id(value_var) in self.tab.values()

class Transform:
    """An object that maintains the state of transformations applied to a
    primitve program."""

    def __init__(self, block):
        if False:
            for i in range(10):
                print('nop')
        assert block == default_main_program().current_block(), 'only support transform on current block of main program.'
        self.block = block
        self.vars = self.init_vars(block)
        self.var2dot = VarMap('var2dot', self.vars)
        self.dot2bar = VarMap('dot2var', self.vars)

    def init_vars(self, block):
        if False:
            for i in range(10):
                print('nop')
        vars = OrderedDict()
        for (_, var) in block.vars.items():
            vars[id(var)] = var
        return vars

    def add_vars(self, new_vars):
        if False:
            while True:
                i = 10
        self.vars.update({id(v): v for v in new_vars if v is not None})

    def add_vars_rec(self, new_vars):
        if False:
            for i in range(10):
                print('nop')
        if new_vars is None:
            return
        if isinstance(new_vars, paddle.base.framework.Variable):
            self.vars.update({id(new_vars): new_vars})
            return
        if not isinstance(new_vars, list):
            raise TypeError(f'new_vars must be list, but got {type(new_vars)}')
        for var in new_vars:
            self.add_vars_rec(var)

    def erase_ops(self, ordered_indexes):
        if False:
            i = 10
            return i + 15
        block = self.block
        for op_index in reversed(ordered_indexes):
            block.desc._remove_op(op_index, op_index + 1)
        for op_index in reversed(ordered_indexes):
            del block.ops[op_index]
        block._sync_with_cpp()

    def erase_dots(self, vars_to_erase):
        if False:
            while True:
                i = 10
        for var in vars_to_erase:
            if id(var) in self.vars:
                del self.vars[id(var)]
        self.dot2bar.delete_keyvars(vars_to_erase)
        self.var2dot.delete_valuevars(vars_to_erase)
        block = self.block
        for var in vars_to_erase:
            name = var.name
            block.desc._remove_var(name.encode())
            del block.vars[name]
        block._sync_with_cpp()

    def var2dot_rec(self, vars):
        if False:
            print('Hello World!')
        'Lookup var2dot recursively.'
        if isinstance(vars, paddle.base.framework.Variable):
            dot = self.var2dot.lookup(vars)
            return dot
        dots = [self.var2dot_rec(var) for var in vars]
        return dots

    def dot2bar_rec(self, dots):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(dots, paddle.base.framework.Variable):
            bar = self.dot2bar.lookup(dots)
            assert bar is not None, 'bar must be not None'
            return bar
        bars = [self.dot2bar_rec(dot) for dot in dots]
        return bars

    def linearize(self, xs, ys, xs_dot=None):
        if False:
            return 10
        'Performs the linearization transform, a.k.a, forward mode AD\n        transform, on a primitive lowered program.\n\n        Args:\n            xs: a list of input variables\n            ys: a list of output variables\n            xs_dot: optional, a list of gradient input variables. The list size\n                must be equal to `len(xs)`. The shape and dtype of each element\n                must be the same as in `xs`\n\n        Returns:\n            (xs_dot, ys_dot): a tuple of two lists. `xs_dot` is the list of\n            gradient inputs of the resulting linearized program. `ys_dot` is\n            the list gradient outputs of the resulting linearized program\n\n        '
        if xs_dot is None:
            xs_dot = [fill_const(1.0, shape=x.shape, dtype=x.dtype) for x in xs]
            self.add_vars(xs_dot)
        else:
            assert len(xs) == len(xs_dot), f'len(xs) should be equal to len(xs_dot), but len(xs)={len(xs)} and len(xs_dot)={len(xs_dot)}'
        for (x, dot) in zip(xs, xs_dot):
            assert x.dtype == dot.dtype, f'x.dtype should be equal to dot.dtype, but x.dtype={x.dtype} and dot.dtype={dot.dtype}'
            assert x.shape == dot.shape, f'x.shape should be equal to dot.shape, but x.shape={x.shape} and dot.shape={dot.shape}'
            self.var2dot.add(x, dot)
        (path, unused_xs, _) = topo_path(xs, ys, self.block)
        for x in unused_xs:
            self.var2dot.delete(x)
        for op in path:
            ins = op_position_inputs(op)
            jvp_ins = self.var2dot_rec(ins)
            outs_dot = _jvp(op, *jvp_ins)
            self.add_vars_rec(outs_dot)
            outs = op_position_output(op)
            self.var2dot.add_rec(outs, outs_dot)
        ys_dot = [self.var2dot.lookup(y) for y in ys]
        return (xs_dot, ys_dot)

    def transpose(self, ys_dot, xs_dot, ys_bar=None, retain_fwd=False):
        if False:
            return 10
        'Performs the transpose transform, a.k.a, reverse mode AD\n        transform, on a linearized primitive program.\n\n        Note, `transpose` is supposed to be used in couple with `linearize`.\n\n        Args:\n            ys_dot: a list of outputs of the linearized program.\n            xs_dot: a list of inputs of the linearized program.\n            ys_bar: optional, a list of inputs of the resulting transposed\n                program. The list size must be equal to `len(ys_dot)`. The shape\n                and dtype of each element must be the same as in `ys_dot`\n\n        Returns:\n            (ys_bar, xs_bar): a tuple of two lists. `ys_bar` is the list of\n            inputs of the resulting transposed program. `xs_bar` is\n            the list outputs of the resulting transposed program\n\n        '
        assert all((v is not None for v in xs_dot)), '`xs_dot` includes None.'
        assert all((v is not None for v in ys_dot)), '`ys_dot` includes None.'
        if ys_bar is None:
            ys_bar = []
            for y in ys_dot:
                ys_bar.append(fill_const(1.0, shape=y.shape, dtype=y.dtype))
            self.add_vars(ys_bar)
        else:
            assert len(ys_dot) == len(ys_bar), f'len(ys_dot) should be equal to len(ys_bar), but len(ys_dot)={len(ys_dot)} and len(ys_bar)={len(ys_bar)}'
            for (y_dot, y_bar) in zip(ys_dot, ys_bar):
                assert y_dot.shape == y_bar.shape, f'y_dot.shape should be equal to y_bar.shape, but y_dot.shape={y_dot.shape} and y_bar.shape={y_bar.shape}'
                assert y_dot.dtype == y_bar.dtype, f'y_dot.dtype should be equal to y_bar.dtype, but y_dot.dtype={y_dot.dtype} and y_bar.dtype={y_bar.dtype}'
        for (dot, bar) in zip(ys_dot, ys_bar):
            self.dot2bar.add(dot, bar)
        (path, unused_xs_dot, _) = topo_path(xs_dot, ys_dot, self.block)
        for dot in unused_xs_dot:
            self.dot2bar.delete(dot)
        dotvars = output_vars_on_path(path)
        dotvars.update(((id(var), var) for var in xs_dot))
        is_dot = lambda v: id(v) in dotvars
        for op in reversed(path):
            out = op_position_output(op)
            out_bar_rec = self.dot2bar_rec(out)
            ins_bar_rec = _transpose(op, is_dot, out_bar_rec)
            if isinstance(ins_bar_rec, tuple):
                ins_bar_rec = list(ins_bar_rec)
            else:
                ins_bar_rec = [ins_bar_rec]
            self.add_vars_rec(ins_bar_rec)
            ins_bar = flatten(ins_bar_rec)
            ins = flatten(op_position_inputs(op))
            assert len(ins) == len(ins_bar), f'len(ins) should be equal to len(ins_bar), but len(ins)={len(ins)} and len(ins_bar)={len(ins_bar)}'
            for (dot, bar) in zip(ins, ins_bar):
                if bar is not None:
                    grad = self.dot2bar.lookup(dot)
                    if grad is None:
                        self.dot2bar.add(dot, bar)
                    else:
                        grad = add(grad, bar)
                        self.add_vars([grad])
                        self.dot2bar.add(dot, grad)
        xs_bar = [self.dot2bar.lookup(x) for x in xs_dot]
        if not retain_fwd and len(path) > 0:
            vars_to_remove = set()
            for op in path:
                vars_to_remove.update(flatten_and_remove_none(get_output_var_list(op)))
            op_indexes = []
            block = self.block
            for (i, op) in enumerate(block.ops):
                if op in path:
                    op_indexes.append(i)
                    path.pop(0)
                    if len(path) == 0:
                        break
            self.erase_ops(op_indexes)
            self.erase_dots(vars_to_remove)
        return (ys_bar, xs_bar)

def _lower(block, reverse, blacklist):
    if False:
        for i in range(10):
            print('nop')

    def bind(args, to_bind, value_table):
        if False:
            print('Hello World!')
        for i in range(len(args)):
            if isinstance(args[i], list):
                bind(args[i], to_bind, value_table)
            elif args[i] is not None and args[i].name in to_bind:
                args[i] = value_table[to_bind[args[i].name]]

    def bind_name(names, to_bind):
        if False:
            i = 10
            return i + 15
        return_list = []
        for name in names:
            if isinstance(name, list):
                return_list.append(bind_name(name, to_bind))
            else:
                return_list.append(to_bind[name] if name in to_bind else name)
        return return_list

    def expand_nested_list(xs):
        if False:
            print('Hello World!')
        return_list = []
        for x in xs:
            if isinstance(x, list):
                return_list = return_list + expand_nested_list(x)
            else:
                return_list.append(x)
        return return_list
    lower_fn = _prim2orig if reverse else _orig2prim
    lookup_fn = lookup_prim2orig if reverse else lookup_orig2prim
    value_table = {}
    to_bind = {}
    to_bind_rev = {}
    for var in block.desc.all_vars():
        value_table[var.name()] = block.var(var.name())
    ops_to_remove = []
    vars_to_remove = set()
    for op_idx in range(len(block.ops)):
        op = block.ops[op_idx]
        ops_to_remove.append(op_idx)
        if lookup_fn(op.type) is not None and op.type not in blacklist:
            input_args = get_input_var_list(op)
            bind(input_args, to_bind, value_table)
            for (orig_out, new_out) in zip(expand_nested_list(get_output_var_list(op)), expand_nested_list(as_tensors(lower_fn(op, *input_args)))):
                assert not (orig_out is None) ^ (new_out is None), 'orig_out and new_out should match.'
                vars_to_remove.add(new_out.name)
                value_table[new_out.name] = new_out
                to_bind[orig_out.name] = new_out.name
                to_bind_rev[new_out.name] = orig_out.name
        else:
            inputs = {}
            for i in range(len(op.input_names)):
                inputs[op.input_names[i]] = bind_name(op.input(op.input_names[i]), to_bind)
            outputs = {}
            for i in range(len(op.output_names)):
                outputs[op.output_names[i]] = op.output(op.output_names[i])
            attrs = {}
            for name in sorted(op.attr_names):
                attrs[name] = op.attr(name)
            from paddle.base.dygraph.base import param_guard
            new_op_desc = block.desc.append_op()
            with param_guard(inputs), param_guard(outputs):
                op = Operator(block=block, desc=new_op_desc, type=op.type, inputs=inputs, outputs=outputs, attrs=attrs)
            block.ops.append(op)
    for op_idx in reversed(ops_to_remove):
        block.desc._remove_op(op_idx, op_idx + 1)
        del block.ops[op_idx]
    block._sync_with_cpp()
    for op_idx in range(len(block.ops)):
        op = block.ops[op_idx]
        for in_name in op.input_arg_names:
            if in_name in to_bind_rev:
                op._rename_input(in_name, to_bind_rev[in_name])
        for out_name in op.output_arg_names:
            if out_name in to_bind_rev:
                op._rename_output(out_name, to_bind_rev[out_name])
    for var_name in sorted(vars_to_remove):
        assert var_name in to_bind_rev, f'var_name "{var_name}" is not in to_bind_rev.'
        if var_name != to_bind_rev[var_name]:
            block.desc._remove_var(var_name.encode())
            del block.vars[var_name]
    block._sync_with_cpp()

def _lower_composite(block, filter_: typing.Callable[[framework.Operator], bool]=lambda x: True, start_idx=-1, backward_length=-1):
    if False:
        print('Hello World!')
    'The operators in block wich satisfy the filter conditon will be decomposite into primitives.'

    def bind(args, to_bind, value_table):
        if False:
            i = 10
            return i + 15
        for i in range(len(args)):
            if isinstance(args[i], list):
                bind(args[i], to_bind, value_table)
            if not isinstance(args[i], paddle.base.framework.Variable):
                continue
            elif args[i] is not None and args[i].name in to_bind:
                args[i] = value_table[to_bind[args[i].name]]

    def bind_name(names, to_bind):
        if False:
            i = 10
            return i + 15
        return_list = []
        for name in names:
            if isinstance(name, list):
                return_list.append(bind_name(name, to_bind))
            else:
                return_list.append(to_bind[name] if name in to_bind else name)
        return return_list

    def expand_nested_list(xs):
        if False:
            for i in range(10):
                print('nop')
        return_list = []
        for x in xs:
            if isinstance(x, list):
                return_list = return_list + expand_nested_list(x)
            else:
                return_list.append(x)
        return return_list
    if isinstance(block, paddle.base.framework.Block):
        logging.info('Atomize composite op to primitive ops begin.')
        lower_fn = _composite
        lookup_fn = lookup_composite
        value_table = {}
        to_bind = {}
        to_bind_rev = {}
        for var in block.desc.all_vars():
            value_table[var.name()] = block.var(var.name())
        ops_to_remove = []
        vars_to_remove = set()
        none_vars_to_remove = set()
        change = None
        length = len(block.ops)
        idx_list = range(length)
        assert -1 <= backward_length <= length, f'expect -1 <= backward_length <= {length}, but got backward_length: {backward_length}'
        assert -1 <= start_idx <= length, f'expect -1 <= start_idx <= {length}, but got start_idx: {start_idx}'
        assert not (backward_length > -1 and start_idx > -1), f'got start_idx: {start_idx} and backward_length: {backward_length}'
        if backward_length > -1:
            idx_list = range(length - backward_length)
        if start_idx > -1:
            idx_list = range(start_idx, length)
        lower = lower_pre = False
        for op_idx in range(length):
            op = block.ops[op_idx]
            ops_to_remove.append(op_idx)
            op_name = op.type
            lower = lookup_fn(op_name) is not None and filter_(op) and (op_idx in idx_list)
            if not lower_pre and lower:
                block._sync_with_cpp()
            lower_pre = lower
            if lower:
                change = True
                prim_config['composite_ops_record'].add(op_name)
                input_args = prepare_python_api_arguments(op)
                bind(input_args, to_bind, value_table)
                orig_outs = expand_nested_list(map_output_for_composite(op))
                new_outs = expand_nested_list(as_tensors(lower_fn(op, *input_args)))
                assert len(orig_outs) == len(new_outs), f'when replace origin op {op_name} with composite rule, num of origin outs should be equal to new outs, but len(orig_outs) = {len(orig_outs)} and len(new_outs) = {len(new_outs)}'
                for (orig_out, new_out) in zip(orig_outs, new_outs):
                    if (orig_out is None or new_out is None) and op_name not in ops_contain_none:
                        raise ValueError(f'op {op_name} should not contain any None value. original outs={orig_outs} and its composite rule outs={new_outs}')
                    if orig_out is None:
                        continue
                    elif new_out is not None:
                        assert orig_out.dtype == new_out.dtype, f'when replace origin op {op_name} with composite rule, origin out dtype should be equal to new out dtype, but orig_out: {orig_out.name}.dtype={orig_out.dtype} and new_out: {new_out.name}.dtype={new_out.dtype}'
                        assert -1 not in new_out.shape, f'when replace origin op {op_name} with composite rule, composite out shape has -1.'
                        assert orig_out.shape == new_out.shape, f'when replace origin op {op_name} with composite rule, origin out shape should be equal to new out shape, but orig_out: {orig_out.name}.shape={orig_out.shape} and new_out: {new_out.name}.shape={new_out.shape}'
                        assert not (orig_out is None) ^ (new_out is None), 'orig_out and new_out should match.'
                        vars_to_remove.add(new_out.name)
                        value_table[new_out.name] = new_out
                        to_bind[orig_out.name] = new_out.name
                        to_bind_rev[new_out.name] = orig_out.name
                    else:
                        none_vars_to_remove.add(orig_out.name)
            else:
                op_desc = block.desc.append_op()
                op_desc.copy_from(op.desc)
        block._sync_with_cpp()
        for op_idx in reversed(ops_to_remove):
            block.desc._remove_op(op_idx, op_idx + 1)
            del block.ops[op_idx]
        block._sync_with_cpp()
        for op_idx in range(len(block.ops)):
            op = block.ops[op_idx]
            for in_name in op.input_arg_names:
                if in_name in to_bind_rev:
                    op._rename_input(in_name, to_bind_rev[in_name])
            for out_name in op.output_arg_names:
                if out_name in to_bind_rev:
                    op._rename_output(out_name, to_bind_rev[out_name])
        for var_name in sorted(vars_to_remove):
            assert var_name in to_bind_rev, f'var_name "{var_name}" is not in to_bind_rev.'
            if var_name != to_bind_rev[var_name]:
                block.desc._remove_var(var_name.encode())
                del block.vars[var_name]
        block._sync_with_cpp()
        for var_name in sorted(none_vars_to_remove):
            block.desc._remove_var(var_name.encode())
            del block.vars[var_name]
        block._sync_with_cpp()
        for op in block.ops:
            if op._has_kernel(op.desc.type()):
                op.desc.infer_var_type(block.desc)
                op.desc.infer_shape(block.desc)
        if change:
            _lower_composite(block, filter_, start_idx=start_idx, backward_length=backward_length)
        return
    elif isinstance(block, typing.Sequence):
        for item in block:
            _lower_composite(item, filter_, start_idx=start_idx, backward_length=backward_length)
        return
    else:
        raise TypeError

@framework.static_only
def orig2prim(block=None):
    if False:
        i = 10
        return i + 15
    '\n    Note:\n        **This API is ONLY available in the static graph mode.**\n        **Args block must be None or current block of main program.**\n\n    All operators in the target block are processed as follows.\n    If it is an original operator, it will be transformed into\n    one or a series of automatic differential basic operators with\n    equivalent function.\n\n    Args:\n        block(paddle.static.Block|None, optional): The\n            target block to process on. Default None, and will\n            process on the current block of main program.\n    '
    block = default_main_program().current_block() if block is None else block
    assert block == default_main_program().current_block(), 'block is neither None nor current block of main program'
    _lower(block, reverse=False, blacklist=[])

@framework.static_only
def prim2orig(block=None, blacklist=None):
    if False:
        i = 10
        return i + 15
    "\n    Note:\n        **ONLY available in the static graph mode.**\n        **Args block must be None or current block of main program.**\n\n    All operators in the target block are processed as follows.\n    If it is an automatic differential basic operator, it will be\n    transformed into one or a series of original operators with\n    equivalent function to support execution.\n\n    Args:\n        block(paddle.static.Block|None, optional): The\n            target block to process on. Default None, and will\n            process on the current block of main program.\n        blacklist(list[string]|None, optional): The names of automatic\n            differential basic operator that will not be transformed\n            into original operators. Default None, and the blacklist\n            is treated as empty list.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import paddle\n            >>> from paddle.incubate.autograd import enable_prim, prim_enabled, prim2orig\n\n            >>> paddle.enable_static()\n            >>> enable_prim()\n\n            >>> x = paddle.ones(shape=[2, 2], dtype='float32')\n            >>> x.stop_gradients = False\n            >>> y = x * x\n            >>> dy_dx = paddle.static.gradients(y, x)\n            >>> if prim_enabled():\n            ...     prim2orig()\n    "
    block = default_main_program().current_block() if block is None else block
    assert block == default_main_program().current_block(), 'block is neither None nor current block of main program'
    blacklist = [] if blacklist is None else blacklist
    _lower(block, reverse=True, blacklist=blacklist)