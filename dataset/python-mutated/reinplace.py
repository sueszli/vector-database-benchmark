import torch
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
from torch.utils import _pytree as pytree
from torch.multiprocessing.reductions import StorageWeakRef
import _operator
from enum import Enum
import itertools
from typing import Set, Dict
from collections import defaultdict
__all__ = ['reinplace']

class _ViewType(Enum):
    NonView = 0
    SingleOutputView = 1
    MultiOutputView = 2

def _is_view_op(tgt):
    if False:
        print('Hello World!')
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            return first_arg.alias_info is not None and (not first_arg.alias_info.is_write)

def _get_view_type(tgt) -> _ViewType:
    if False:
        i = 10
        return i + 15
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            if first_arg.alias_info is not None and (not first_arg.alias_info.is_write):
                if '*' in first_arg.alias_info.after_set:
                    return _ViewType.MultiOutputView
                else:
                    return _ViewType.SingleOutputView
    return _ViewType.NonView

@compatibility(is_backward_compatible=False)
class _FunctionalizationMetadataProp(torch.fx.Interpreter):

    def run_node(self, node: Node):
        if False:
            while True:
                i = 10
        self.node_counter += 1
        result = super().run_node(node)
        node.meta['fake_result'] = result
        node.meta['node_idx'] = self.node_counter
        node_args = node.args
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]
        if node.op == 'call_function':
            view_type = _get_view_type(node.target)
            if view_type == _ViewType.SingleOutputView:
                assert isinstance(node.args[0], Node)
                node.meta['view_of'] = node.args[0]
            elif view_type == _ViewType.MultiOutputView:
                self.multi_output_view_nodes[node] = node.args[0]
            elif node.target is _operator.getitem:
                list_arg = node.args[0]
                maybe_base_of_view = self.multi_output_view_nodes.get(list_arg, None)
                if maybe_base_of_view is not None:
                    assert isinstance(maybe_base_of_view, Node)
                    node.meta['view_of'] = maybe_base_of_view
        if 'view_of' in node.meta:
            assert isinstance(node.meta['fake_result'], FakeTensor)
            assert isinstance(node.meta['view_of'].meta['fake_result'], FakeTensor)
            view_storage = StorageWeakRef(node.meta['fake_result']._typed_storage())
            base_storage = StorageWeakRef(node.meta['view_of'].meta['fake_result']._typed_storage())
            assert view_storage == base_storage
        return result

    def propagate(self, *args):
        if False:
            i = 10
            return i + 15
        self.multi_output_view_nodes = {}
        self.node_counter = -1
        with FakeTensorMode() as mode:
            fake_args = [mode.from_tensor(a) for a in args]
            return super().run(*fake_args)

def _schemas_match(functional_schema, inplace_schema):
    if False:
        return 10
    names_match = inplace_schema.name.endswith('_') and inplace_schema.name[:-1] == functional_schema.name
    arg_types_match = len(functional_schema.arguments) == len(inplace_schema.arguments) and all((a1.type == a2.type for (a1, a2) in zip(functional_schema.arguments, inplace_schema.arguments)))
    assert inplace_schema.arguments[0].alias_info is not None and inplace_schema.arguments[0].alias_info.is_write
    assert all((a.alias_info is None for a in inplace_schema.arguments[1:]))
    return names_match and arg_types_match

def _maybe_get_inplace_op(op):
    if False:
        return 10
    if not isinstance(op, torch._ops.OpOverload):
        return None
    if _is_view_op(op):
        return None
    op_namespace = op.__module__.split('.')[-1]
    op_base_name = op.overloadpacket.__name__
    maybe_namespace_module = getattr(torch.ops, op_namespace)
    maybe_inplace_op = None if maybe_namespace_module is None else getattr(maybe_namespace_module, f'{op_base_name}_', None)
    if maybe_inplace_op is None:
        return None
    inplace_overloads = [getattr(maybe_inplace_op, overload_name) for overload_name in maybe_inplace_op.overloads()]
    inplace_overloads_with_matching_schemas = [f for f in inplace_overloads if _schemas_match(op._schema, f._schema)]
    if len(inplace_overloads_with_matching_schemas) == 0:
        return None
    assert len(inplace_overloads_with_matching_schemas) == 1
    inplace_op = inplace_overloads_with_matching_schemas[0]
    return inplace_op
_VIEW_INVERSE_MAP = {torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default, torch.ops.aten.select_scatter.default: torch.ops.aten.select.int, torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor, torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default}

def _get_all_later_node_usages(tensor_aliases: Set[Node], op_index: int):
    if False:
        i = 10
        return i + 15

    def _add_if_tensor(x, set_):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, FakeTensor):
            set_.add(StorageWeakRef(x._typed_storage()))
    nodes_used_after = set()
    for t in tensor_aliases:
        usage_nodes = t.users
        for n in usage_nodes:
            if 'node_idx' not in n.meta or n.meta['node_idx'] <= op_index:
                continue
            if n in tensor_aliases:
                if isinstance(n.target, torch._ops.OpOverload) or n.target == _operator.getitem:
                    continue
            nodes_used_after.add(n)
    return nodes_used_after

def _get_view_inverse_node_usages(later_node_usages: Set[Node], self_aliases: Set[Node]) -> Set[Node]:
    if False:
        print('Hello World!')

    def matching_view_metadata(a, b):
        if False:
            i = 10
            return i + 15
        return a.size() == b.size() and a.stride() == b.stride() and (a.storage_offset() == b.storage_offset())
    view_inverse_nodes = set()
    for n in sorted(later_node_usages, key=lambda x: x.meta['node_idx']):
        if n.target not in _VIEW_INVERSE_MAP:
            continue
        base = n.args[0]
        mutated_view = n.args[1]
        assert isinstance(base, Node)
        assert isinstance(base.meta['fake_result'], FakeTensor)
        assert isinstance(mutated_view, Node)
        assert isinstance(mutated_view.meta['fake_result'], FakeTensor)
        original_view = _VIEW_INVERSE_MAP[n.target]
        for self_alias in self_aliases:
            if 'view_of' not in self_alias.meta:
                continue
            self_alias_base = self_alias.meta['view_of']
            try:
                view_replay_metadata = original_view(self_alias_base.meta['fake_result'], *n.args[2:], **n.kwargs)
                expected_metadata = self_alias.meta['fake_result']
                if matching_view_metadata(self_alias_base.meta['fake_result'], base.meta['fake_result']) and matching_view_metadata(view_replay_metadata, expected_metadata):
                    view_inverse_nodes.add(n)
            except Exception:
                continue
    return view_inverse_nodes

@compatibility(is_backward_compatible=True)
def reinplace(gm, *sample_args):
    if False:
        i = 10
        return i + 15
    '\n    Given an fx.GraphModule, modifies it to perform "reinplacing",\n    mutating the nodes of the graph.\n    We look for out-of-place op call sites like `b = a.add(...)`,\n    and convert them to be inplace (`b = a.add_(...)`),\n    as long as the input to the current operator ("a") isn\'t re-used\n    anywhere later in the graph.\n\n    This pass currently expects to operate on a **functional, ATen** graph.\n    This can be obtained by running `make_fx(functionalize(f))`.\n\n    Sample inputs are needed to determine aliasing relationships of the inputs.\n    In general, we can\'t reinplace node `b = a.add(...)` if "a" aliases any of the\n    inputs to the program.\n\n    Given a node "b = foo(a, args...) the algorithm for re-inplacing is as follows:\n\n    (1) Perform some initial checks on the metadata of "a" and "args..."\n        that can disqualify them from being reinplaced.\n\n      (1a) Check that the self argument we\'re attempting to reinplace\n           has acceptable dtype/size metadata to reinplace with.\n\n           For example, if we have:\n             a = torch.ones(1)\n             b = torch.ones(10)\n             out = torch.add(a, b)\n           We can\'t turn that into\n             a.add_(b)\n           Because that would require resizing "a".\n\n           Similarly, we can\'t convert torch.ge(a, b) into a.ge_(b),\n           because that would require changing a\'s dtype (from e.g. float32 to bool).\n           Note that in this specific example, we could technically do better..\n\n           If we see the pattern:\n             a_1 = a.ge(b)\n             a_2 = aten._to_copy(a_1, a.dtype)\n           Then we this should be valid to completely re-inplace\n           (this is exactly what functionalization will emit when it sees a.ge_(b)).\n\n           This optimization is only really important for user programs\n           that directly use inplace comparison ops though.\n\n           We also cannot re-inplace on tensors that have overlapping memory,\n           e.g. torch.ones(1).expand(4, 4).add_(1)\n\n      (1b) Check if "a" is an alias of any of the program inputs.\n\n          If it is, skip and move to the next node.\n          Inplace\'ing an op that would cause it to mutate a program is not sound,\n          because that would be a side effect visible to the user.\n\n          NOTE: there\'s a future optimization that we should make:\n          if "a" is a (alias of a)  program input, but later in the program\n          there is a node that looks like "a.copy_(...)",\n          Then re-inplacing is ok to do - we are temporarily re-using a\'s buffer,\n          which will later be overwritten by the copy_() call.\n\n          This will be an important optimization to have for programs that mutate\n          their inputs. It currently isn\'t implemented though.\n\n      (1c) Check if "a" and "args..." alias\n\n          For example, re-inplacing to create code like the below\n          isn\'t guaranteed to be sound:\n\n            aten.mul_(a, a)\n\n    (2) Check that "a" and all of its outstanding aliases are not used anywhere\n        later in the graph. If this is the case, then it\'s safe to re-inplace\n        to "b = foo_(a)".\n\n        There are a few caveats to this, explained in more detail below:\n        (a) If "a" is used later as an argument to a view op, that is okay.\n            It\'s only a problem if "a" (or that view) is later passed\n            into a normal operator, or if it is returned as the program output.\n        (b) If "a" is a repeat argument in `foo()`, then don\'t reinplace.\n            Most ATen kernels don\'t make any guarantees that this is sound,\n            e.g. if you do aten.mul_(a, a).\n            So we\'ll just ban re-inplacing in this case.\n            It\'s only a problem if "a" (or that view) is later passed\n        (c) If "a" is used as an input into a view "inverse" / "scatter"\n            operator, it is potentially fine to re-inplace\n            (and remove that scatter operator from the graph).\n            See below for a more detailed example.\n\n        NOTE: there is an optimization in this step that is crucial\n        to fully recovering performance from functionalization.\n\n        Given this program:\n        def f(x):\n            a = torch.ops.aten.add(x, x)\n            b = torch.ops.aten.diagonal(a)\n            torch.ops.aten.fill_(b, 0)\n            return d\n\n        Functionalization will emit the following:\n        def f(x):\n            a = torch.ops.aten.add(x, x)\n            b = torch.ops.aten.diagonal(a, 0, 1)\n            b_updated = torch.ops.aten.fill(b, 0)\n            a_updated = torch.ops.aten.diagonal_scatter(a, b_updated, 0, 1)\n            return a_updated\n\n        Ordinarily, we would not be able to reinplace the fill,\n        because "b" aliases with "a" which is used by the diagonal_scatter call.\n\n        "re-inplacing" is on the hook for figuring out that it is ok to\n        completely, the expensive diagonal_scatter call, if we re-inplace the add().\n\n        So, for every `alias in alias_set(a)`, instead of checking\n        that "alias" is not used anywhere later in the graph,\n        we check that\n            EITHER:\n          (a) alias is not used anywhere later in the graph\n            OR:\n          (b) alias is used exactly once later on in the graph,\n              in the following op:\n\n                out = foo_scatter(alias, x, args...)\n\n              where the following must hold:\n                (i) "foo_scatter" is the "inverse" operator for foo.\n                    This only applies to "foo" ops that are view operators,\n                    which view into a subset of the original tensor\'s memory.\n                    In practice, there are ~4 operators where this applies:\n                      diagonal -> diagonal_scatter\n                      slice -> slice_scatter\n                      select -> select_scatter\n                      as_strided -> as_strided_scatter\n                (ii) "args..." are the same between the foo() and foo_scatter() calls.\n\n    (3) Perform the actual re-inplacing on foo!\n\n      (3b) is the common case, but special care is needed for {view}_scatter (3a)\n\n      (3a) {view}_scatter ops.\n\n        Consider this program:\n          a = torch.zeros(2, 2)\n          b = torch.ones(2)\n          a[0] = b\n\n        Post functionalization, that will look like:\n          a = torch.zeros(2)\n          b = torch.ones(1)\n          a_updated = torch.select_scatter(a, b, 0, 0)\n\n        In this case though, there is no "functional" op to re-inplace!\n        Instead, we\'d like to directly remove toe select_scatter call.\n        We already know from (3) that this is valid,\n        because "a" has no later usages in the graph.\n\n        We perform the re-inplacing on the {view}_scatter op like so\n        Before:\n          a_updated = torch.select_scatter(a, b, args...)\n        After:\n          a_slice = a.select(a, args...)\n          a_slice.copy_(b)\n\n      (3b) Otherwise, replace the functional op with its inplace variant.\n        Before:\n          b = foo(a, args...)\n        After:\n          a.foo_(args...)\n\n    (4) Finally, after converting either:\n          Before:\n            b = foo(a)\n          After:\n            foo_(a)\n        or\n          Before:\n            b = {slice}_scatter(a, mutated_slice, args...)\n          After:\n            slice = {slice}(a, args...)\n            slice.copy_(mutated_slice)\n\n        We now need to find all later nodes that use "b" as an argument\n        and update them to take in "a" instead.\n\n        Note that for the majority of inplace ops, this isn\'t actually necessary\n        (because most inplace ops return "self" as their output).\n        This isn\'t generally true for all mutable ops though, which is why\n        we need to actually replace all of the arguments.\n\n        We also need to update our metadata of Dict[StorageWeakRef, Set[Node]],\n        That maps a given tensor storage to the set of all nodes that take in that storage\n        as an input.\n        Specifically, re-inplacing `b = foo(a)` causes "a" and "b"\'s sets to get fused\n        together.\n\n    (5) Any "view_inverse/scatter" nodes that were identified as "it\'s ok to ignore them"\n        during step (3) get manually deleted from the graph.\n        Their outputs are no longer used, so technically standard DCE would be able\n        to do this, but we can no longer run FX\'s DCE pass now that we have mutable\n        ops in the graph.\n    '
    _FunctionalizationMetadataProp(gm).propagate(*sample_args)
    input_storages = {StorageWeakRef(node.meta['fake_result']._typed_storage()) for node in gm.graph.nodes if node.op == 'placeholder'}
    storage_to_nodes: Dict[StorageWeakRef, Set[Node]] = defaultdict(set)
    for n in gm.graph.nodes:
        if 'fake_result' in n.meta:

            def _add_to_map(x):
                if False:
                    return 10
                if isinstance(x, FakeTensor):
                    storage_to_nodes[StorageWeakRef(x._typed_storage())].add(n)
            pytree.tree_map_(_add_to_map, n.meta['fake_result'])
    all_later_view_inverse_nodes_to_delete = set()
    for (idx, node) in enumerate(gm.graph.nodes):
        if node.op == 'call_function':
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            if len(node.target._schema.arguments) < 1:
                continue
            if type(node.target._schema.arguments[0].type) != torch.TensorType:
                continue
            self_arg = node.args[0]
            self_flattened = pytree.tree_leaves(self_arg.meta['fake_result'])
            node_flattened = pytree.tree_leaves(node.meta['fake_result'])
            self_has_wrong_metadata = False
            if len(self_flattened) == len(node_flattened):
                for (self_meta, node_meta) in zip(self_flattened, node_flattened):
                    if self_meta.numel() != node_meta.numel():
                        self_has_wrong_metadata = True
                    if self_meta.dtype != node_meta.dtype:
                        self_has_wrong_metadata = True
                    if torch._debug_has_internal_overlap(self_meta) == 1:
                        self_has_wrong_metadata = True
            if self_has_wrong_metadata and node.target != torch.ops.aten.resize.default:
                continue
            self_arg_name = self_arg.name
            self_arg_storage = StorageWeakRef(self_arg.meta['fake_result']._typed_storage())
            if self_arg_storage in input_storages:
                continue
            if len([x for x in node.args if x is self_arg]) > 1:
                continue
            self_arg_storage = StorageWeakRef(self_arg.meta['fake_result']._typed_storage())
            self_aliases = storage_to_nodes[self_arg_storage]
            later_node_usages = _get_all_later_node_usages(self_aliases, node.meta['node_idx'])
            later_view_inverse_node_usages = _get_view_inverse_node_usages(later_node_usages, self_aliases)
            can_reinplace = len(later_node_usages - later_view_inverse_node_usages) == 0
            if not can_reinplace:
                continue
            if node.target in _VIEW_INVERSE_MAP and node not in all_later_view_inverse_nodes_to_delete:
                view_op = _VIEW_INVERSE_MAP[node.target]
                with gm.graph.inserting_before(node):
                    mutated_slice_node = node.args[1]
                    remaining_slice_args = node.args[2:]
                    slice_node = gm.graph.create_node('call_function', view_op, (self_arg,) + tuple(remaining_slice_args), node.kwargs)
                    copy_node = gm.graph.create_node('call_function', torch.ops.aten.copy_.default, (slice_node, mutated_slice_node), {})
                all_later_view_inverse_nodes_to_delete.add(node)
            else:
                maybe_inplace_op = _maybe_get_inplace_op(node.target)
                if maybe_inplace_op is None:
                    continue
                node.target = maybe_inplace_op
            curr_node_storage = StorageWeakRef(node.meta['fake_result']._typed_storage())
            storage_to_nodes[self_arg_storage].update(storage_to_nodes[curr_node_storage])
            storage_to_nodes[curr_node_storage].update(storage_to_nodes[self_arg_storage])
            all_later_view_inverse_nodes_to_delete.update(later_view_inverse_node_usages)
            for old in itertools.chain([node], later_view_inverse_node_usages):
                new = old.args[0]
                nodes_to_update = [n for n in old.users if n.meta['node_idx'] > node.meta['node_idx']]
                for node_to_update in nodes_to_update:
                    new_args = []
                    args = node_to_update.args

                    def replace_arg(a):
                        if False:
                            while True:
                                i = 10
                        if a == old:
                            return new
                        return a
                    node_to_update.args = tree_map_only(Node, replace_arg, node_to_update.args)
                    node_to_update.kwargs = tree_map_only(Node, replace_arg, node_to_update.kwargs)
                    old_flattened_res = pytree.tree_leaves(old.meta['fake_result'])
                    node_flattened_res = pytree.tree_leaves(node_to_update.meta['fake_result'])
                    old_res_storage = {StorageWeakRef(x._typed_storage()) for x in old_flattened_res if isinstance(x, FakeTensor)}
                    node_res_storage = {StorageWeakRef(x._typed_storage()) for x in node_flattened_res if isinstance(x, FakeTensor)}
                    if len(old_res_storage) == 1 and len(node_res_storage) == 1 and (old_res_storage == node_res_storage):
                        new_flattened_res = pytree.tree_leaves(new.meta['fake_result'])
                        new_res_storage = {StorageWeakRef(x._typed_storage()) for x in new_flattened_res if isinstance(x, FakeTensor)}
                        assert len(new_res_storage) == 1
                        (old_ref,) = old_res_storage
                        (new_ref,) = new_res_storage
                        (node_ref,) = node_res_storage
                        storage_to_nodes[node_ref].update(storage_to_nodes[new_ref])
                        storage_to_nodes[new_ref].update(storage_to_nodes[node_ref])
    for to_delete in all_later_view_inverse_nodes_to_delete:
        gm.graph.erase_node(to_delete)
    gm.recompile()
    return gm