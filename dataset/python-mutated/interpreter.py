from .graph_module import GraphModule
from .graph import Graph
from .node import Argument, Node, Target, map_arg, map_aggregate
from .proxy import Proxy
from ._symbolic_trace import Tracer
from ._compatibility import compatibility
from . import config
import torch.fx.traceback as fx_traceback
import torch
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import inspect
from contextlib import contextmanager
from torch.hub import tqdm
__all__ = ['Interpreter', 'Transformer']

@compatibility(is_backward_compatible=True)
class Interpreter:
    """
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overrideable methods
    in terms of call hierarchy::

        run()
            +-- run_node
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(self, target : Target,
                                  args : Tuple, kwargs : Dict) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(n)

                def call_method(self, target : Target,
                                args : Tuple, kwargs : Dict) -> Any:
                    if target == 'neg':
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(n)

            def fn(x):
                return torch.sigmoid(x).neg()

            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            torch.testing.assert_close(result, torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The module to be executed
        garbage_collect_values (bool): Whether to delete values after their last
            use within the Module's execution. This ensures optimal memory usage during
            execution. This can be disabled to, for example, examine all of the intermediate
            values in the execution by looking at the ``Interpreter.env`` attribute.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, module: GraphModule, garbage_collect_values: bool=True):
        if False:
            print('Hello World!')
        assert isinstance(module, GraphModule)
        self.module = module
        self.submodules = dict(self.module.named_modules())
        self.env: Dict[Node, Any] = {}
        self.name = 'Interpreter'
        self.garbage_collect_values = garbage_collect_values
        self.extra_traceback = True
        if self.garbage_collect_values:
            node_to_last_use: Dict[Node, Node] = {}
            self.user_to_last_uses: Dict[Node, List[Node]] = {}

            def register_last_uses(n: Node, user: Node):
                if False:
                    print('Hello World!')
                if n not in node_to_last_use:
                    node_to_last_use[n] = user
                    self.user_to_last_uses.setdefault(user, []).append(n)
            for node in reversed(self.module.graph.nodes):
                map_arg(node.args, lambda n: register_last_uses(n, node))
                map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    @compatibility(is_backward_compatible=True)
    def run(self, *args, initial_env: Optional[Dict[Node, Any]]=None, enable_io_processing: bool=True) -> Any:
        if False:
            print('Hello World!')
        "\n        Run `module` via interpretation and return the result.\n\n        Args:\n            *args: The arguments to the Module to run, in positional order\n            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.\n                This is a dict mapping `Node` to any value. This can be used, for example, to\n                pre-populate results for certain `Nodes` so as to do only partial evaluation within\n                the interpreter.\n            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and\n                process_outputs function first before using them.\n\n        Returns:\n            Any: The value returned from executing the Module\n        "
        self.env = initial_env if initial_env is not None else {}
        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter: Iterator[Any] = iter(args)
        pbar = tqdm(total=len(self.module.graph.nodes), desc=f"{self.name}: {(str(list(self.module.graph.nodes)) if config.verbose_progress else '')}", initial=0, position=0, leave=True, disable=config.disable_progress, delay=0)
        for node in self.module.graph.nodes:
            pbar.update(1)
            if node in self.env:
                continue
            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                if self.extra_traceback:
                    msg = f'While executing {node.format_node()}'
                    msg = f'{e.args[0]}\n\n{msg}' if e.args else str(msg)
                    msg += f'\nOriginal traceback:\n{node.stack_trace}'
                    e.args = (msg,) + e.args[1:]
                    if isinstance(e, KeyError):
                        raise RuntimeError(*e.args) from e
                raise
            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]
            if node.op == 'output':
                output_val = self.env[node]
                return self.module.graph.process_outputs(output_val) if enable_io_processing else output_val

    @compatibility(is_backward_compatible=True)
    def boxed_run(self, args_list):
        if False:
            i = 10
            return i + 15
        '\n        Run `module` via interpretation and return the result.  This uses the "boxed"\n        calling convention, where you pass a list of arguments, which will be cleared\n        by the interpreter.  This ensures that input tensors are promptly deallocated.\n        '
        args_iter = iter(args_list)
        env = {}
        for n in self.module.graph.nodes:
            if n.op == 'placeholder':
                env[n] = next(args_iter)
        args_list.clear()
        return self.run(initial_env=env)

    @contextmanager
    def _set_current_node(self, node):
        if False:
            i = 10
            return i + 15
        with fx_traceback.set_current_meta(node):
            yield

    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Run a specific node ``n`` and return the result.\n        Calls into placeholder, get_attr, call_function,\n        call_method, call_module, or output depending\n        on ``node.op``\n\n        Args:\n            n (Node): The Node to execute\n\n        Returns:\n            Any: The result of executing ``n``\n        '
        with self._set_current_node(n):
            (args, kwargs) = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            return getattr(self, n.op)(n.target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            print('Hello World!')
        '\n        Execute a ``placeholder`` node. Note that this is stateful:\n        ``Interpreter`` maintains an internal iterator over\n        arguments passed to ``run`` and this method returns\n        next() on that iterator.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n\n        Returns:\n            Any: The argument value that was retrieved.\n        '
        assert isinstance(target, str)
        if target.startswith('*'):
            return list(self.args_iter)
        else:
            try:
                return next(self.args_iter)
            except StopIteration as si:
                if len(args) > 0:
                    return args[0]
                else:
                    raise RuntimeError(f'Expected positional argument for parameter {target}, but one was not passed in!') from si

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            while True:
                i = 10
        '\n        Execute a ``get_attr`` node. Will retrieve an attribute\n        value from the ``Module`` hierarchy of ``self.module``.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n\n        Return:\n            Any: The value of the attribute that was retrieved\n        '
        assert isinstance(target, str)
        return self.fetch_attr(target)

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Execute a ``call_function`` node and return the result.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n\n        Return\n            Any: The value returned by the function invocation\n        '
        assert not isinstance(target, str)
        return target(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Execute a ``call_method`` node and return the result.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n\n        Return\n            Any: The value returned by the method invocation\n        '
        (self_obj, *args_tail) = args
        assert isinstance(target, str)
        return getattr(self_obj, target)(*args_tail, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Execute a ``call_module`` node and return the result.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n\n        Return\n            Any: The value returned by the module invocation\n        '
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return submod(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute an ``output`` node. This really just retrieves\n        the value referenced by the ``output`` node and returns it.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n\n        Return:\n            Any: The return value referenced by the output node\n        '
        return args[0]

    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.\n\n        Args:\n            target (str): The fully-qualified name of the attribute to fetch\n\n        Return:\n            Any: The value of the attribute.\n        '
        target_atoms = target.split('.')
        attr_itr = self.module
        for (i, atom) in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    @compatibility(is_backward_compatible=True)
    def fetch_args_kwargs_from_env(self, n: Node) -> Tuple[Tuple, Dict]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``\n        from the current execution environment.\n\n        Args:\n            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.\n\n        Return:\n            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.\n        '
        args = self.map_nodes_to_values(n.args, n)
        assert isinstance(args, tuple)
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        assert isinstance(kwargs, dict)
        return (args, kwargs)

    @compatibility(is_backward_compatible=True)
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument:
        if False:
            print('Hello World!')
        '\n        Recursively descend through ``args`` and look up the concrete value\n        for each ``Node`` in the current execution environment.\n\n        Args:\n            args (Argument): Data structure within which to look up concrete values\n\n            n (Node): Node to which ``args`` belongs. This is only used for error reporting.\n        '

        def load_arg(n_arg: Node) -> Any:
            if False:
                return 10
            if n_arg not in self.env:
                raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() to diagnose such issues')
            return self.env[n_arg]
        return map_arg(args, load_arg)

@compatibility(is_backward_compatible=True)
class Transformer(Interpreter):
    """
    ``Transformer`` is a special type of interpreter that produces a
    new ``Module``. It exposes a ``transform()`` method that returns
    the transformed ``Module``. ``Transformer`` does not require
    arguments to run, as ``Interpreter`` does. ``Transformer`` works
    entirely symbolically.

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass ``Transformer`` like so::

            class NegSigmSwapXformer(Transformer):
                def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(n)

                def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
                    if target == 'neg':
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(n)

            def fn(x):
                return torch.sigmoid(x).neg()

            gm = torch.fx.symbolic_trace(fn)

            transformed : torch.nn.Module = NegSigmSwapXformer(gm).transform()
            input = torch.randn(3, 4)
            torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The ``Module`` to be transformed.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, module):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(module)
        self.new_graph = Graph()
        self.new_graph.set_codegen(module.graph._codegen)

        class TransformerTracer(Tracer):

            def __init__(self, graph: Graph):
                if False:
                    return 10
                super().__init__()
                self.graph = graph
                self.tensor_attrs: Dict[torch.Tensor, str] = {}

            def is_leaf_module(self, _, __) -> bool:
                if False:
                    i = 10
                    return i + 15
                return True
        self.tracer = TransformerTracer(self.new_graph)
        self.tracer.root = module

    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Proxy:
        if False:
            print('Hello World!')
        '\n        Execute a ``placeholder`` node. In ``Transformer``, this is\n        overridden to insert a new ``placeholder`` into the output\n        graph.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n        '
        assert isinstance(target, str)
        default_value = next(iter(args)) if args else inspect.Signature.empty
        return Proxy(self.new_graph.placeholder(target, default_value=default_value), self.tracer)

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Proxy:
        if False:
            return 10
        '\n        Execute a ``get_attr`` node. In ``Transformer``, this is\n        overridden to insert a new ``get_attr`` node into the output\n        graph.\n\n        Args:\n            target (Target): The call target for this node. See\n                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for\n                details on semantics\n            args (Tuple): Tuple of positional args for this invocation\n            kwargs (Dict): Dict of keyword arguments for this invocation\n        '
        assert isinstance(target, str)
        return self.tracer.create_proxy('get_attr', target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            print('Hello World!')
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return self.tracer.call_module(submod, submod.forward, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if False:
            print('Hello World!')
        return self.tracer.create_proxy('call_function', target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def transform(self) -> GraphModule:
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform ``self.module`` and return the transformed\n        ``GraphModule``.\n        '
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                if False:
                    i = 10
                    return i + 15
                return a.node if isinstance(a, Proxy) else a
            self.new_graph.output(map_aggregate(result, strip_proxy))
        return GraphModule(self.module, self.new_graph)