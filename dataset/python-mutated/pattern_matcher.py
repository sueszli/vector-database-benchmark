from __future__ import annotations
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, NoReturn, Optional, Set, Union
from typing_extensions import TypeGuard
import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type
log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
Constant = Any
NodeOrConstant = Union[Constant, torch.fx.Node]

class Multiple:
    pass
MULTIPLE = Multiple()

class Match:
    """
    Represents a successfully matched pattern.
    """

    def __init__(self, pattern: PatternExpr, args=None, kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.pattern = pattern
        self.args = args or []
        self.kwargs = kwargs or {}
        self.nodes: List[torch.fx.Node] = []
        self.targets: Dict[_TargetExpr, torch.fx.node.Target] = {}
        self.ctx: Optional[MatchContext] = None
        self.replacement_graph: Optional[torch.fx.Graph] = None

    @property
    def graph(self) -> torch.fx.Graph:
        if False:
            i = 10
            return i + 15
        assert self.ctx
        return self.ctx.graph

    def extend(self, other: Match):
        if False:
            return 10
        if self.kwargs:
            for key in set(self.kwargs.keys()) & set(other.kwargs.keys()):
                if self.kwargs[key] != other.kwargs[key]:
                    raise FailedMatch('kwarg mismatch: {}', key)
        self.args.extend(other.args)
        self.nodes.extend(other.nodes)
        self.kwargs.update(other.kwargs)
        self.targets.update(other.targets)

    def bundle(self) -> Match:
        if False:
            print('Hello World!')
        self.args = [tuple(self.args)] if self.args else []
        return self

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'Match(..., {self.args}, {self.kwargs})'

    def erase_nodes(self, graph: torch.fx.Graph):
        if False:
            while True:
                i = 10
        for n in reversed(self.nodes):
            if not n._erased:
                graph.erase_node(n)

    def output_nodes(self) -> List[Optional[torch.fx.Node]]:
        if False:
            for i in range(10):
                print('nop')
        assert self.ctx
        return [self.ctx.pattern_to_node[p] if p is not None else None for p in self.ctx.outputs]

    def output_node(self) -> torch.fx.Node:
        if False:
            for i in range(10):
                print('nop')
        return next((p for p in self.output_nodes() if p))

    def replace_with_graph(self, replacement_graph, args):
        if False:
            while True:
                i = 10
        assert self.ctx
        ReplacementPatternEntry.replace_with_graph(self, self.ctx.graph, replacement_graph, args)

    def replace_by_example(self, replacement_fn, args, trace_fn=None):
        if False:
            return 10
        assert self.ctx
        if trace_fn is None:
            trace_fn = fwd_only
        replacement = trace_fn(replacement_fn, torch.fx.map_arg(args, lambda arg: arg.meta['val']))
        ReplacementPatternEntry.replace_with_graph(self, self.ctx.graph, replacement, args)

class FailedMatch(RuntimeError):

    def __init__(self, format_string, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.format_string = format_string
        if len(format_string) > 200:
            raise RuntimeError(f'Format string too long - use lazy construction of strings instead. Format string is\n {format_string}')
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.format_string.format(*self.args, **self.kwargs)

    def __bool__(self):
        if False:
            while True:
                i = 10
        return False

def is_match(m: Union[Match, FailedMatch]) -> TypeGuard[Match]:
    if False:
        return 10
    '\n    TypeGuards cannot act on `self`. Thus this function exists to let mypy\n    recognize FailedMatch.__bool__ as a TypeGuard.\n    '
    return bool(m)

class MatchContext:
    """
    State needed while running PatternExpr._match().
    """

    def __init__(self, outputs: List[Optional[PatternExpr]], pattern_to_node: Optional[Dict[PatternExpr, Node]]=None, *, graph: torch.fx.Graph):
        if False:
            while True:
                i = 10
        self.outputs = outputs
        self.pattern_to_node = {} if pattern_to_node is None else pattern_to_node
        self.graph = graph
        self.exclusive_node_set: List[NodeOrConstant] = []

    def match(self, pattern, node):
        if False:
            print('Hello World!')
        'wrapper to check reused nodes in patterns'
        if pattern in self.pattern_to_node:
            if self.pattern_to_node[pattern] == node:
                return Match(pattern)
            else:
                return FailedMatch('repeated pattern differs')
        m = pattern._match(node, self)
        assert pattern not in self.pattern_to_node
        self.pattern_to_node[pattern] = node if m else None
        m.ctx = self
        return m

    def filter_multi_user_patterns(self):
        if False:
            print('Hello World!')
        return {pattern: node for (pattern, node) in self.pattern_to_node.items() if pattern.has_multiple_users() and node is not None}

class PatternExpr:
    """
    Base class for types of patterns
    """

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> Union[Match, FailedMatch]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def match(self, node: torch.fx.Node) -> Union[Match, FailedMatch]:
        if False:
            print('Hello World!')
        try:
            return MatchContext([self], graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def has_multiple_users(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def __repr__(self):
        if False:
            return 10
        return self.__class__.__name__ + '()'

    def find_anchor_nodes(self, ctx: MatchContext, searched):
        if False:
            while True:
                i = 10
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]

class Arg(PatternExpr):
    """
    Capture an arg which will become an input to the handler.  Args are
    passed in depth first order.
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        if False:
            i = 10
            return i + 15
        return Match(self, args=[node])

class Ignored(PatternExpr):
    """
    Match an arg, but don't pass it to handler
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        if False:
            print('Hello World!')
        return Match(self)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '*'

    def pretty_print(self, pp: PatternPrettyPrinter):
        if False:
            i = 10
            return i + 15
        return 'Ignored()'

class KeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name: str):
        if False:
            print('Hello World!')
        super().__init__()
        self.name = name

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'KeywordArg({self.name!r})'

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        if False:
            i = 10
            return i + 15
        return Match(self, kwargs={self.name: node})

class ExclusiveKeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name):
        if False:
            while True:
                i = 10
        super().__init__()
        self.name = name

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'ExclusiveKeywordArg({self.name!r})'

    def _match(self, node: NodeOrConstant, ctx: MatchContext):
        if False:
            print('Hello World!')
        if node in ctx.exclusive_node_set:
            return FailedMatch('exclusive arg appears twice')
        ctx.exclusive_node_set.append(node)
        return Match(self, kwargs={self.name: node})

class _TargetExpr(PatternExpr):
    """
    Base class for filtering match by node.target
    """
    op: Optional[str] = None

    def __init__(self, fns, users=1):
        if False:
            return 10
        if not self.op:
            raise NotImplementedError("Shouldn't directly use _BaseNodeMatch")
        super().__init__()
        fns = [fns] if callable(fns) or isinstance(fns, str) else list(fns)
        for fn in list(fns):
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend([getattr(fn, overload) for overload in fn.overloads()])
        self.fns: List[Union[Callable[..., Any], str]] = fns
        self.fns_set: Set[Union[Callable[..., Any], str]] = set(fns)
        self.users: Union[int, Multiple] = users

    def fns_repr(self) -> str:
        if False:
            i = 10
            return i + 15
        first_repr = self.fns[0]
        if not isinstance(first_repr, str):
            first_repr = first_repr.__name__
        if len(self.fns) > 1:
            return f'[{first_repr}, ...]'
        elif self.fns[0] is getattr(torch, first_repr, None):
            return f'torch.{first_repr}'
        elif isinstance(self.fns[0], torch._ops.OpOverload):
            return str(self.fns[0])
        else:
            return first_repr

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__class__.__name__}({self.fns_repr()})'

    def has_multiple_users(self) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(self.users, Multiple) or self.users > 1

    def find_anchor_nodes(self, ctx: MatchContext, searched):
        if False:
            return 10
        raise NotImplementedError()

    def _match_fns(self, node: torch.fx.Node):
        if False:
            while True:
                i = 10
        return isinstance(node, torch.fx.Node) and node.op == self.op and (extract_target(node) in self.fns_set)

    def _match_users(self, node: torch.fx.Node, ctx: MatchContext):
        if False:
            for i in range(10):
                print('nop')
        return self in ctx.outputs or self.users is MULTIPLE or len(node.users) == self.users

class _TargetArgsExpr(_TargetExpr):
    """
    Base class for filtering match by node.{target,args,kwargs}
    """

    def __init__(self, fns, *args, _users=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(fns, _users)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        if any((isinstance(x, (dict, list, tuple)) for x in itertools.chain(args, kwargs.values()))):
            self.flatten = self.pytree_flatten
        else:
            self.flatten = self.simple_flatten
        self.flat_args_kwargs = self.flatten(self.args, self.kwargs)

    @staticmethod
    def simple_flatten(args, kwargs: Dict[Any, Any]):
        if False:
            return 10
        return ((*args, *kwargs.values()), (len(args), *kwargs.keys()))

    @staticmethod
    def pytree_flatten(args, kwargs: Dict[Any, Any]):
        if False:
            return 10

        def norm_spec(s: pytree.TreeSpec):
            if False:
                i = 10
                return i + 15
            if s.type is None:
                return s
            mapping = {immutable_list: list, tuple: list, immutable_dict: dict}
            return pytree.TreeSpec(mapping.get(s.type, s.type), s.context, list(map(norm_spec, s.children_specs)))
        (flat, spec) = pytree.tree_flatten([args, kwargs])
        spec = norm_spec(spec)
        return (flat, spec)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        args = [self.fns_repr(), *map(repr, self.args), *[f'{k}={v}' for (k, v) in self.kwargs.items()]]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def pretty_print(self, pp: PatternPrettyPrinter):
        if False:
            print('Hello World!')
        args = [self.fns_repr(), *(pp.pretty_print(x) for x in self.args), *[f'{k}={pp.pretty_print(v)}' for (k, v) in self.kwargs.items()]]
        if isinstance(self.users, Multiple):
            args.append('_users=MULTIPLE')
        elif self.users > 1:
            args.append(f'_users={self.users}')
        joiner_str = ', '
        return f'{self.__class__.__name__}({joiner_str.join(args)})'

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if False:
            print('Hello World!')
        if not self._match_fns(node) or len(node.args) != len(self.args):
            return FailedMatch('function_mismatch: node={}, pattern={}', node, self)
        if not self._match_users(node, ctx):
            return FailedMatch('multiple_users {}', self)
        _args = node.args
        _kwargs = node.kwargs
        if len(_kwargs) < len(self.kwargs):
            from torch.fx.operator_schemas import normalize_function
            normalized_args_and_kwargs = normalize_function(node.target, node.args, node.kwargs)
            if normalized_args_and_kwargs is None:
                return FailedMatch('function_mismatch: node={}, pattern={}', node, self)
            else:
                (_args, _kwargs) = normalized_args_and_kwargs
                if len(_args) == len(self.args) and len(_kwargs) >= len(self.kwargs):
                    _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
                else:
                    return FailedMatch('function_mismatch: node={}, pattern={}', node, self)
        else:
            _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
        (node_items, node_spec) = self.flatten(_args, _kwargs)
        (self_items, self_spec) = self.flat_args_kwargs
        if node_spec != self_spec:
            return FailedMatch('args_structure {} {}', node_spec, self_spec)
        assert len(node_items) == len(self_items)
        m = Match(self)
        for (i, pattern, child_node) in zip(itertools.count(), self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not child_match:
                    return child_match
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch('constant_args: {} {!r}!={pattern!r}', node, child_node)
        m.nodes.append(node)
        m.targets[self] = node.target
        return m

    def find_anchor_nodes(self, ctx: MatchContext, searched):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is used when we are matching a pattern with multiple outputs.\n        There is a partial match (stored in ctx) and we want to walk\n        this pattern to find a connection to an already-matched node.\n\n        Yields candidate nodes that `self._match` might like.\n        '
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]
            return
        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        if node not in searched:
                            if self._match_fns(node):
                                yield node
                                searched.add(node)

class CallFunction(_TargetArgsExpr):
    """
    Matches a call_function node in the FX graphs: `fns[i](*args, **kwargs)`
    """
    op = 'call_function'

class CallMethod(_TargetArgsExpr):
    """
    Matches a call_method node in the FX graphs: `fns[i].method(*args, **kwargs)`
    """
    op = 'call_method'

class CallModule(_TargetArgsExpr):
    """
    Matches a call_module node in the FX graphs: `module(*args, **kwargs)`
    """
    op = 'call_module'

class _TargetExprVarArgs(_TargetExpr):
    """
    Matches a call_function node with any arguments which are passed into the pattern
    """

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if False:
            print('Hello World!')
        if not self._match_fns(node):
            return FailedMatch('function_mismatch')
        if not self._match_users(node, ctx):
            return FailedMatch('multiple_users')
        m = Match(self)
        m.nodes.append(node)
        m.targets[self] = node.target
        m.args.extend(node.args)
        m.kwargs.update(node.kwargs)
        return m

class CallFunctionVarArgs(_TargetExprVarArgs):
    op = 'call_function'

class CallMethodVarArgs(_TargetExprVarArgs):
    op = 'call_method'

class CallModuleVarArgs(_TargetExprVarArgs):
    op = 'call_module'

class ListOf(PatternExpr):
    """
    Matches a repeated pattern
    """

    def __init__(self, pattern: PatternExpr, partial=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert isinstance(pattern, PatternExpr)
        self.pattern = pattern
        self.partial = partial

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}({self.pattern})'

    def _match(self, node: List[torch.fx.Node], ctx: MatchContext):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(node, (list, tuple)) or len(node) == 0:
            return FailedMatch('non_list')
        m = Match(self)
        pattern_to_node = ctx.filter_multi_user_patterns()
        matched = False
        for (i, child_node) in enumerate(node):
            child_ctx = MatchContext(ctx.outputs, pattern_to_node, graph=child_node.graph)
            child_match = child_ctx.match(self.pattern, child_node)
            pattern_to_node = child_ctx.filter_multi_user_patterns()
            if not child_match:
                if not self.partial:
                    return FailedMatch('list[{}]: {}', i, child_match)
                continue
            matched = True
            m.extend(child_match.bundle())
        if not matched:
            return FailedMatch('list: no_match')
        return m.bundle()

class MultiOutputPattern(PatternExpr):

    def __init__(self, outputs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert all((isinstance(x, (PatternExpr, type(None))) for x in outputs)), outputs
        self.outputs: List[Optional[PatternExpr]] = outputs

    @property
    def fns(self):
        if False:
            i = 10
            return i + 15
        assert self.outputs[0] and hasattr(self.outputs[0], 'fns')
        return self.outputs[0].fns

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}({self.outputs})'

    def pretty_print(self, pp: PatternPrettyPrinter):
        if False:
            i = 10
            return i + 15
        args = [pp.pretty_print(x) for x in self.outputs]
        joiner_str = f",\n{'  '}"
        str_out = f'{self.__class__.__name__}([{joiner_str.join(args)}'
        str_out = f'{str_out}\n])'
        return str_out

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if False:
            return 10
        m = ctx.match(self.outputs[0], node)
        if not m:
            return m
        for pattern in self.outputs[1:]:
            if pattern is None:
                continue
            child_match = self._match_from_anchors(pattern, ctx)
            if not child_match:
                return child_match
            m.extend(child_match)
        return m

    def _match_from_anchors(self, pattern, ctx):
        if False:
            return 10
        prior = dict(ctx.pattern_to_node)
        m = FailedMatch('no anchor found')
        for node in pattern.find_anchor_nodes(ctx, set()):
            m = ctx.match(pattern, node)
            if m:
                return m
            ctx.pattern_to_node = dict(prior)
        return m

    def match(self, node: torch.fx.Node) -> Union[Match, FailedMatch]:
        if False:
            print('Hello World!')
        try:
            return MatchContext(self.outputs, graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

class RepeatedExpr(PatternExpr):
    """
    Checks for a repeated pattern. Useful for repeated operations after a node such as `split` or `unbind`
    """

    def __init__(self, inner_pattern: PatternExpr):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert hasattr(inner_pattern, 'fns')
        self.inner_pattern = inner_pattern

    @property
    def fns(self):
        if False:
            for i in range(10):
                print('nop')
        return self.inner_pattern.fns

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if False:
            while True:
                i = 10
        m = ctx.match(self.inner_pattern, node)
        if not m:
            return m
        ctx.pattern_to_node.pop(self.inner_pattern)
        for anchor_node in self.inner_pattern.find_anchor_nodes(ctx, set()):
            anchor_m = MatchContext([self], graph=node.graph).match(self.inner_pattern, anchor_node)
            if not anchor_m:
                return anchor_m
            m.extend(anchor_m)
        return m

class PatternPrettyPrinter:
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.namespace = torch.fx.graph._Namespace()
        self.memoized_objs_names: Dict[PatternExpr, str] = {}
        self.memoized_objs_pp: Dict[PatternExpr, str] = {}

    @staticmethod
    def run(obj: PatternExpr, output_name='output'):
        if False:
            print('Hello World!')
        '\n        Serializes obj to python code with obj written out to `output_name`\n        '
        pp = PatternPrettyPrinter()
        assert hasattr(obj, 'pretty_print')
        out_str = obj.pretty_print(pp=pp)
        output = []
        for key in pp.memoized_objs_names:
            output.append(f'{pp.memoized_objs_names[key]} = {pp.memoized_objs_pp[key]}')
        output.append(f'{output_name} = {out_str}')
        return '\n'.join(output)

    def pretty_print(self, obj):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, _TargetArgsExpr):
            if (memoized_name := self.memoized_objs_names.get(obj)):
                return memoized_name
            else:
                return self.memoize(obj)
        if hasattr(obj, 'pretty_print'):
            return obj.pretty_print(self)
        return repr(obj)

    def memoize(self, obj):
        if False:
            print('Hello World!')
        obj_str = obj.pretty_print(self)
        obj_name = obj.fns_repr()
        for prefix in ('aten.', 'torch.', 'prims.'):
            obj_name = obj_name.replace(prefix, '')
        tmp_name = self.namespace.create_name(obj_name, None)
        self.memoized_objs_names[obj] = tmp_name
        self.memoized_objs_pp[obj] = obj_str
        return tmp_name

@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def register(self, pass_dicts, target=None, prepend=False):
        if False:
            print('Hello World!')
        if target is None:
            assert hasattr(self.pattern, 'fns')
            for fn in self.pattern.fns:
                self.register(pass_dicts, fn, prepend=prepend)
        elif isinstance(pass_dicts, (dict, PatternMatcherPass)):
            if prepend:
                pass_dicts[target].insert(0, self)
            else:
                pass_dicts[target].append(self)
        else:
            for x in pass_dicts:
                self.register(x, target, prepend=prepend)

@dataclasses.dataclass
class LoweringPatternEntry(PatternEntry):
    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        if False:
            for i in range(10):
                print('nop')
        handler = functools.wraps(self.handler)(functools.partial(self.handler, match))
        with graph.inserting_before(node):
            replacement = graph.call_function(handler, tuple(match.args), match.kwargs)
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
        assert match.nodes[-1] is node
        match.erase_nodes(graph)

@dataclasses.dataclass
class GraphPatternEntry(PatternEntry):
    """
    A pattern that runs a function on the FX graph
    """
    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        if False:
            print('Hello World!')
        with graph.inserting_before(node):
            self.handler(match, *match.args, **match.kwargs)

@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    normalize_args: Callable[..., List[Any]]

    @staticmethod
    def replace_with_graph(match: Match, graph: torch.fx.Graph, replacement_graph: torch.fx.Graph, args: List[Any]):
        if False:
            while True:
                i = 10
        output_nodes = match.output_nodes()
        first_node = output_nodes[0]

        class Replacer(torch.fx.Interpreter):
            call_method = None
            call_module = None
            get_attr = None

            def run_node(self, node) -> Any:
                if False:
                    return 10
                if node.op in ('placeholder', 'output'):
                    return super().run_node(node)
                if node.op == 'call_function':
                    target = node.target
                    (args, kwargs) = self.fetch_args_kwargs_from_env(node)
                    result = graph.call_function(target, args, kwargs)
                    if 'val' in node.meta and 'val' not in result.meta:
                        result.meta['val'] = node.meta['val']
                        if isinstance(node.meta['val'], torch.Tensor):
                            assert 'tensor_meta' in node.meta
                            result.meta['tensor_meta'] = node.meta['tensor_meta']
                    return result
                raise NotImplementedError(f'unhandled {node}')
        output_nodes = match.output_nodes()
        if len(output_nodes) == 1:
            last_node = output_nodes[0]
        else:
            assert output_nodes[0]
            nodes = list(output_nodes[0].graph.nodes)
            indices = [(nodes.index(n), n) for n in output_nodes if isinstance(n, torch.fx.Node)]
            last_node = min(indices, key=lambda tup: tup[0])[1]

        def percolate_tags(node, recompute_tag):
            if False:
                return 10
            for arg in node.all_input_nodes:
                if hasattr(arg, 'meta'):
                    arg.meta['recompute'] = recompute_tag
                    percolate_tags(arg, recompute_tag)
        with graph.inserting_before(last_node):
            replacement = Replacer(replacement_graph).run(*args)
            if isinstance(replacement, torch.fx.Node):
                replacement = [replacement]
            assert len(replacement) == len(output_nodes)
            for (old, new) in zip(output_nodes, replacement):
                if old is None:
                    assert new is None
                elif new is None:
                    old.replace_all_uses_with(None)
                else:
                    if 'val' not in new.meta:
                        new.meta.update(old.meta)
                    if 'recompute' in old.meta:
                        percolate_tags(new, old.meta['recompute'])
                    old.replace_all_uses_with(new)
        match.erase_nodes(graph)

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        if False:
            print('Hello World!')
        self.replace_with_graph(match, graph, match.replacement_graph, self.normalize_args(*match.args, **match.kwargs))

def _return_true(match):
    if False:
        for i in range(10):
            print('nop')
    return True

def register_replacement(search_fn, replace_fn, example_inputs: Iterable[Any], trace_fn: Callable[[Callable[..., Any], Iterable[Any]], torch.fx.GraphModule], pass_dicts, extra_check=_return_true, scalar_workaround=(), exclusive_arg_names=(), search_fn_pattern=None):
    if False:
        print('Hello World!')
    '\n    Create a replacement rule based on example functions that get traced\n    to create patterns.  This supports both training and inference when\n    run on a joint forward+backward graph.\n\n    Args:\n        search_fn: traced to give original pattern\n        replace_fn: traced to give replacement graph\n        example_inputs: example inputs for initial trace\n        trace_fn: fwd_only or joint_fwd_bwd\n        pass_dict: dict of passes to register to\n        extra_check: additional check to run on match(using real shapes)\n    '
    argnames = [*inspect.signature(search_fn).parameters.keys()]

    def check_fn(match: Match):
        if False:
            while True:
                i = 10
        '\n        Often shapes get burned into the pattern, so our initial match ran with\n        `ignore_types=(int, ...)`.\n\n        Recheck the match with the correct shapes.\n        '
        for name in argnames:
            if name not in match.kwargs:
                raise RuntimeError(f'Not all inputs to pattern found in match.kwargs. Perhaps one of the inputs is unused? argnames={argnames}, match.kwargs={match.kwargs}')
        args = list(torch.fx.map_arg([match.kwargs[name] for name in argnames], lambda n: n.meta['val']))
        with torch._dynamo.utils.detect_fake_mode(args):
            for (i, grad) in enumerate(requires_grad):
                if isinstance(args[i], torch.Tensor):
                    if grad and is_integer_dtype(args[i].dtype):
                        return False
                    args[i] = torch.empty_strided(args[i].size(), args[i].stride(), dtype=args[i].dtype, device=args[i].device, requires_grad=grad)
            specific_graph = trace_fn(search_fn, args)
            specific_pattern = fx_to_pattern(specific_graph, argnames=argnames, exclusive_arg_names=exclusive_arg_names, scalar_workaround=scalar_workaround)
            specific_pattern_match = specific_pattern.match(match.output_nodes()[0])
            if specific_pattern_match and extra_check(specific_pattern_match):
                match.replacement_graph = trace_fn(replace_fn, args)
                return True
            return False

    def normalize_args(**kwargs):
        if False:
            print('Hello World!')
        args = []
        for name in argnames:
            args.append(kwargs.pop(name))
        for i in range(1, len(kwargs) + 1):
            if f'tangents_{i}' not in kwargs:
                break
            args.append(kwargs.pop(f'tangents_{i}'))
        assert not kwargs, f'leftover kwargs: {kwargs!r}'
        return args
    if trace_fn is joint_fwd_bwd:
        if torch.is_inference_mode_enabled():
            return False
    with functorch_config.patch(functionalize_rng_ops=False):
        requires_grad: List[bool] = [isinstance(x, torch.Tensor) and x.requires_grad for x in example_inputs]
        if search_fn_pattern is None:
            pattern = gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround, exclusive_arg_names)
        else:
            pattern = search_fn_pattern
        pattern_repr = PatternPrettyPrinter.run(pattern)
        assert pattern_repr not in _seen_patterns
        _seen_patterns.add(pattern_repr)
        pattern = ReplacementPatternEntry(pattern=pattern, extra_check=check_fn, normalize_args=normalize_args)
        pattern.register(pass_dicts)
        return pattern.pattern

@functorch_config.patch(functionalize_rng_ops=False)
def gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround=(), exclusive_arg_names=()) -> PatternExpr:
    if False:
        for i in range(10):
            print('nop')
    argnames = [*inspect.signature(search_fn).parameters.keys()]
    if scalar_workaround == ():
        scalar_workaround = {}
    flat_inputs = []
    input_idx = 0
    for argname in argnames:
        if argname in scalar_workaround:
            flat_inputs.append(scalar_workaround[argname])
        else:
            flat_inputs.append(example_inputs[input_idx])
            input_idx += 1
    search_gm = trace_fn(search_fn, flat_inputs)
    return fx_to_pattern(search_gm, ignore_types=(int, float, list, torch.device, torch.dtype), argnames=argnames, scalar_workaround=scalar_workaround, exclusive_arg_names=exclusive_arg_names)

def register_lowering_pattern(pattern: PatternExpr, extra_check=_return_true, *, pass_dict, prepend=False):
    if False:
        print('Hello World!')
    '\n    Register an aten to inductor IR replacement pattern.  The decorated\n    function is saved and then called a lowering time allowing direct\n    pattern to inductor IR conversion.\n    '

    def decorator(handler):
        if False:
            print('Hello World!')
        assert callable(handler)
        LoweringPatternEntry(pattern=pattern, extra_check=extra_check, handler=handler).register(pass_dict, prepend=prepend)
        handler._inductor_lowering_function = True
        return handler
    return decorator

def register_graph_pattern(pattern: PatternExpr, extra_check=_return_true, *, pass_dict, prepend=False):
    if False:
        return 10
    '\n    Register a pattern that runs a function on the FX graph, allowing\n    custom transformation code.\n    '

    def decorator(handler):
        if False:
            i = 10
            return i + 15
        assert callable(handler)
        GraphPatternEntry(pattern=pattern, extra_check=extra_check, handler=handler).register(pass_dict, prepend=prepend)
        return handler
    return decorator

def is_start_of_fx_graph(graph: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if False:
        while True:
            i = 10
    return node is next(iter(graph.nodes))
_mutation_op_re = re.compile('_$|(\\b|_)(set|enter|exit|seed)(\\b|_)')

def is_mutation_op(node: torch.fx.Node) -> bool:
    if False:
        while True:
            i = 10
    if node.op == 'call_function':
        if _mutation_op_re.search(node.target.__name__):
            return True
    elif node.op == 'call_method':
        if _mutation_op_re.search(node.target):
            return True
    return node.kwargs.get('out') is not None

def get_mutation_region_id(graph: torch.fx.GraphModule, node: torch.fx.Node) -> int:
    if False:
        return 10
    n = node
    while 'mutation_region_id' not in n.meta and (not is_start_of_fx_graph(graph, n)):
        n = n.prev
    mutation_region_id = n.meta.get('mutation_region_id', 0)
    while n is not node:
        n = n.next
        if is_mutation_op(n):
            mutation_region_id += 1
        n.meta['mutation_region_id'] = mutation_region_id
    return mutation_region_id

def should_compute_mutation_region_ids(graph: torch.fx.GraphModule) -> bool:
    if False:
        return 10
    return 'mutation_region_id' not in next(iter(graph.nodes)).meta

def compute_mutation_region_ids(graph: torch.fx.GraphModule):
    if False:
        for i in range(10):
            print('nop')
    mutation_region_id = 0
    for nd in graph.nodes:
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta['mutation_region_id'] = mutation_region_id

class PatternMatcherPass:

    def __init__(self, prevent_match_across_mutations=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.patterns: DefaultDict[torch.fx.node.Target, List[PatternEntry]] = defaultdict(list)
        self.prevent_match_across_mutations = prevent_match_across_mutations

    def __getitem__(self, item: torch.fx.node.Target) -> List[PatternEntry]:
        if False:
            return 10
        return self.patterns[item]

    def apply(self, graph: torch.fx.GraphModule) -> int:
        if False:
            print('Hello World!')
        if not self.patterns:
            return 0
        if isinstance(graph, torch.fx.GraphModule):
            graph = graph.graph
        if self.prevent_match_across_mutations:
            if should_compute_mutation_region_ids(graph):
                compute_mutation_region_ids(graph)
            get_mutation_region_id_partial = functools.partial(get_mutation_region_id, graph)
        count = 0
        for node in reversed(graph.nodes):
            target = extract_target(node)
            if node.op in ['call_function', 'call_method', 'call_module'] and target in self.patterns:
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue
                for entry in self.patterns[target]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    if self.prevent_match_across_mutations and is_match(m) and (len(set(map(get_mutation_region_id_partial, m.nodes))) != 1):
                        continue
                    if os.environ.get('TORCHINDUCTOR_PATTERN_MATCH_DEBUG') == node.name:
                        log.warning('%s%s %s %s', node, node.args, m, entry.pattern)
                    if is_match(m) and entry.extra_check(m):
                        count += 1
                        entry.apply(m, graph, node)
                        counters['inductor']['pattern_matcher_count'] += 1
                        counters['inductor']['pattern_matcher_nodes'] += len(m.nodes)
        return count

    def clear(self):
        if False:
            while True:
                i = 10
        self.patterns.clear()

def _not_implemented(*args, **kwargs) -> NoReturn:
    if False:
        return 10
    raise NotImplementedError()

def fx_to_pattern(gm, ignore_types=(), argnames=(), scalar_workaround=(), exclusive_arg_names=()) -> PatternExpr:
    if False:
        print('Hello World!')
    '\n    Convert an FX graph into a PatternExpr.  This is useful for simple\n    patterns that can only match single functions and fixed-length lists.\n    '
    scalar_workaround = scalar_workaround or {}
    inv_scalar_workaround = {v: k for (k, v) in scalar_workaround.items()}
    assert len(inv_scalar_workaround) == len(scalar_workaround)

    def process_arg(x):
        if False:
            return 10
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in ignore_types:
            return Ignored()
        if isinstance(x, list) and all((isinstance(y, Ignored) for y in x)) and x:
            return Ignored()
        return x
    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        call_method = _not_implemented
        call_module = _not_implemented
        get_attr = _not_implemented

        def placeholder(self, target, args, kwargs):
            if False:
                while True:
                    i = 10
            n = next(argnum)
            if n < len(argnames):
                name = argnames[n]
            elif argnames:
                assert target.startswith('tangent')
                name = target
            else:
                target = re.sub('_\\d+$', '', target)
                name = target
            if name in exclusive_arg_names:
                return ExclusiveKeywordArg(name)
            else:
                return KeywordArg(name)

        def call_function(self, target, args, kwargs):
            if False:
                i = 10
                return i + 15
            (args, kwargs) = pytree.tree_map(process_arg, (args, kwargs))
            if list in ignore_types:
                args = [process_arg(a) for a in args]
                kwargs = {k: process_arg(a) for (k, a) in kwargs.items()}
            return CallFunction(target, *args, **kwargs)

        def run_node(self, n):
            if False:
                return 10
            rv = super().run_node(n)
            if n.op == 'output' and isinstance(rv, tuple):
                assert len(rv) == len(n.args[0])
                for (r, arg) in zip(rv, n.args[0]):
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv
    pattern = Converter(gm).run()
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern

@torch.no_grad()
def fwd_only(fn, args) -> torch.fx.GraphModule:
    if False:
        print('Hello World!')
    'Build a normalized inference graph, for use with fx_to_pattern'
    with enable_python_dispatcher():
        gm = make_fx(fn, select_decomp_table())(*args)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

@torch.enable_grad()
def joint_fwd_bwd(fn, args) -> torch.fx.GraphModule:
    if False:
        while True:
            i = 10
    'Build a normalized training graph, for use with fx_to_pattern'
    gm: Optional[torch.fx.GraphModule] = None

    def record_joint_graph(joint_graph, inputs, **kwargs):
        if False:
            print('Hello World!')
        nonlocal gm
        assert not gm
        gm = clone_graph(joint_graph)
        return default_partition(joint_graph, inputs, **kwargs)
    with torch._guards.tracing(None):
        aot_function(fn, lambda g, i: make_boxed_func(g), partition_fn=record_joint_graph, decompositions=select_decomp_table(), keep_inference_input_mutations=True, enable_log=False)(*args)
    assert gm
    from .fx_passes.joint_graph import pointless_view
    matcher_pass = PatternMatcherPass()
    pattern = CallFunction(torch.ops.aten.view.default, KeywordArg('arg'), KeywordArg('size'))
    GraphPatternEntry(pattern=pattern, handler=pointless_view, extra_check=_return_true).register(matcher_pass.patterns)
    matcher_pass.apply(gm.graph)
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def _args(n: torch.fx.Node) -> List[torch.fx.node.Argument]:
    if False:
        for i in range(10):
            print('nop')
    args: List[torch.fx.node.Argument] = list()
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    return args

def stable_topological_sort(graph: torch.fx.Graph):
    if False:
        i = 10
        return i + 15
    waiting = defaultdict(list)
    ready = set()
    cursor = None

    def check(node):
        if False:
            i = 10
            return i + 15
        waiting_for = [x for x in _args(node) if x not in ready]
        if waiting_for:
            waiting[waiting_for[0]].append(node)
        else:
            nonlocal cursor
            cursor = node
            ready.add(node)
            for other in waiting.pop(node, ()):
                cursor.append(other)
                check(other)
    for n in list(graph.nodes):
        check(n)
    assert not waiting and len(ready) == len(graph.nodes)

def init_once_fakemode(fn: Callable[..., Any]):
    if False:
        while True:
            i = 10
    'Wrapper around lazy init functions in fx_passes/'

    @functools.lru_cache(None)
    @functools.wraps(fn)
    def lazy_init():
        if False:
            print('Hello World!')
        counters_ref = counters['inductor'].copy()
        with torch._guards.tracing(None), maybe_disable_fake_tensor_mode(), FakeTensorMode():
            result = fn()
        counters['inductor'] = counters_ref
        return result
    return lazy_init

def config_flag(name):
    if False:
        while True:
            i = 10
    'Function for extra_check to put pass behind a flag'

    def flag_check(match):
        if False:
            i = 10
            return i + 15
        return getattr(config, name)
    return flag_check

def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    if False:
        i = 10
        return i + 15

    class CopyGraph(Transformer):

        def run_node(self, old_node):
            if False:
                print('Hello World!')
            new_node = super().run_node(old_node)
            if isinstance(new_node, torch.fx.Proxy):
                new_node.node.meta.update(old_node.meta)
                new_node.node.name = self.new_graph._graph_namespace.create_name(old_node.name, None)
            return new_node
    return CopyGraph(input_graph).transform()
_seen_patterns: Set[str] = set()

def get_arg_value(node: torch.fx.Node, arg_number: int, kwarg_name: Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    return node.args[arg_number] if len(node.args) > arg_number else node.kwargs.get(kwarg_name)

def filter_nodes(nodes: Iterable[torch.fx.Node], fn) -> List[torch.fx.Node]:
    if False:
        print('Hello World!')
    fns = [fn]
    if isinstance(fn, torch._ops.OpOverloadPacket):
        fns.extend([getattr(fn, overload) for overload in fn.overloads()])
    return [node for node in nodes if node.target in fns]

def extract_target(node: Node):
    if False:
        i = 10
        return i + 15
    'For call_function and call_method, we directly use the target function;\n    For call_module, the target is string, and we treat the module class\n     as a function.\n    '
    if node.op == 'call_module':
        return getattr(node.graph.owning_module, node.target).__class__
    return node.target