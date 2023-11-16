"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.
"""
import functools
from typing import Any, Callable, Iterable, List, Optional, overload, Tuple, Type, TypeVar, Union
import optree
from optree import PyTreeSpec
__all__ = ['PyTree', 'Context', 'FlattenFunc', 'UnflattenFunc', 'TreeSpec', 'LeafSpec', 'register_pytree_node', 'tree_flatten', 'tree_unflatten', 'tree_leaves', 'tree_structure', 'tree_map', 'tree_map_', 'tree_map_only', 'tree_map_only_', 'tree_all', 'tree_any', 'tree_all_only', 'tree_any_only', 'treespec_dumps', 'treespec_loads', 'treespec_pprint']
T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
R = TypeVar('R')
Context = Optional[Any]
PyTree = Any
TreeSpec = PyTreeSpec
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[Iterable, Context], PyTree]
OpTreeUnflattenFunc = Callable[[Context, Iterable], PyTree]

def _reverse_args(func: UnflattenFunc) -> OpTreeUnflattenFunc:
    if False:
        return 10

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return func(*reversed(args), **kwargs)
    return wrapped

def register_pytree_node(cls: Type[Any], flatten_fn: FlattenFunc, unflatten_fn: UnflattenFunc, *, serialized_type_name: Optional[str]=None, namespace: str='torch') -> None:
    if False:
        return 10
    'Extend the set of types that are considered internal nodes in pytrees.\n\n    The ``namespace`` argument is used to avoid collisions that occur when different libraries\n    register the same Python type with different behaviors. It is recommended to add a unique prefix\n    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify\n    the same class in different namespaces for different use cases.\n\n    .. warning::\n        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is\n        used to isolate the behavior of flattening and unflattening a pytree node type. This is to\n        prevent accidental collisions between different libraries that may register the same type.\n\n    Args:\n        cls (type): A Python type to treat as an internal pytree node.\n        flatten_fn (callable): A function to be used during flattening, taking an instance of\n            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened\n            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be\n            passed to the ``unflatten_fn``.\n        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was\n            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.\n            The function should return an instance of ``cls``.\n        serialized_type_name (str, optional): A keyword argument used to specify the fully\n            qualified name used when serializing the tree spec.\n        namespace (str, optional): A non-empty string that uniquely identifies the namespace of the\n            type registry. This is used to isolate the registry from other modules that might\n            register a different custom behavior for the same type. (default: :const:`"torch"`)\n\n    Example::\n\n        >>> # xdoctest: +SKIP\n        >>> # Registry a Python type with lambda functions\n        >>> register_pytree_node(\n        ...     set,\n        ...     lambda s: (sorted(s), None, None),\n        ...     lambda children, _: set(children),\n        ...     namespace=\'set\',\n        ... )\n\n        >>> # xdoctest: +SKIP\n        >>> # Register a Python type into a namespace\n        >>> import torch\n        >>> register_pytree_node(\n        ...     torch.Tensor,\n        ...     flatten_func=lambda tensor: (\n        ...         (tensor.cpu().detach().numpy(),),\n        ...         {\'dtype\': tensor.dtype, \'device\': tensor.device, \'requires_grad\': tensor.requires_grad},\n        ...     ),\n        ...     unflatten_func=lambda children, metadata: torch.tensor(children[0], **metadata),\n        ...     namespace=\'torch2numpy\',\n        ... )\n\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)\n        >>> tree = {\'weight\': torch.ones(size=(1, 2)).cuda(), \'bias\': torch.zeros(size=(2,))}\n        >>> tree\n        {\'weight\': tensor([[1., 1.]], device=\'cuda:0\'), \'bias\': tensor([0., 0.])}\n\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)\n        >>> # Flatten without specifying the namespace\n        >>> tree_flatten(tree)  # `torch.Tensor`s are leaf nodes  # xdoctest: +SKIP\n        ([tensor([0., 0.]), tensor([[1., 1.]], device=\'cuda:0\')], PyTreeSpec({\'bias\': *, \'weight\': *}))\n\n        >>> # xdoctest: +SKIP\n        >>> # Flatten with the namespace\n        >>> tree_flatten(tree, namespace=\'torch2numpy\')  # xdoctest: +SKIP\n        (\n            [array([0., 0.], dtype=float32), array([[1., 1.]], dtype=float32)],\n            PyTreeSpec(\n                {\n                    \'bias\': CustomTreeNode(Tensor[{\'dtype\': torch.float32, ...}], [*]),\n                    \'weight\': CustomTreeNode(Tensor[{\'dtype\': torch.float32, ...}], [*])\n                },\n                namespace=\'torch2numpy\'\n            )\n        )\n\n        >>> # xdoctest: +SKIP\n        >>> # Register the same type with a different namespace for different behaviors\n        >>> def tensor2flatparam(tensor):\n        ...     return [torch.nn.Parameter(tensor.reshape(-1))], tensor.shape, None\n        ...\n        >>> def flatparam2tensor(children, metadata):\n        ...     return children[0].reshape(metadata)\n        ...\n        >>> register_pytree_node(\n        ...     torch.Tensor,\n        ...     flatten_func=tensor2flatparam,\n        ...     unflatten_func=flatparam2tensor,\n        ...     namespace=\'tensor2flatparam\',\n        ... )\n\n        >>> # xdoctest: +SKIP\n        >>> # Flatten with the new namespace\n        >>> tree_flatten(tree, namespace=\'tensor2flatparam\')  # xdoctest: +SKIP\n        (\n            [\n                Parameter containing: tensor([0., 0.], requires_grad=True),\n                Parameter containing: tensor([1., 1.], device=\'cuda:0\', requires_grad=True)\n            ],\n            PyTreeSpec(\n                {\n                    \'bias\': CustomTreeNode(Tensor[torch.Size([2])], [*]),\n                    \'weight\': CustomTreeNode(Tensor[torch.Size([1, 2])], [*])\n                },\n                namespace=\'tensor2flatparam\'\n            )\n        )\n    '
    from ._pytree import _register_pytree_node
    _register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=serialized_type_name)
    optree.register_pytree_node(cls, flatten_fn, _reverse_args(unflatten_fn), namespace=namespace)
_register_pytree_node = register_pytree_node

def tree_flatten(tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> Tuple[List[Any], TreeSpec]:
    if False:
        return 10
    'Flatten a pytree.\n\n    See also :func:`tree_unflatten`.\n\n    The flattening order (i.e., the order of elements in the output list) is deterministic,\n    corresponding to a left-to-right depth-first tree traversal.\n\n    >>> tree = {\'b\': (2, [3, 4]), \'a\': 1, \'c\': None, \'d\': 5}\n    >>> tree_flatten(tree)\n    ([1, 2, 3, 4, None, 5], PyTreeSpec({\'a\': *, \'b\': (*, [*, *]), \'c\': *, \'d\': *}, NoneIsLeaf))\n    >>> tree_flatten(tree, none_is_leaf=False)\n    ([1, 2, 3, 4, 5], PyTreeSpec({\'a\': *, \'b\': (*, [*, *]), \'c\': None, \'d\': *}))\n    >>> tree_flatten(1)\n    ([1], PyTreeSpec(*, NoneIsLeaf))\n    >>> tree_flatten(None)\n    ([None], PyTreeSpec(*, NoneIsLeaf))\n    >>> tree_flatten(None, none_is_leaf=False)\n    ([], PyTreeSpec(None))\n\n    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is\n    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`\n    if you want to keep the keys in the insertion order.\n\n    >>> from collections import OrderedDict\n    >>> tree = OrderedDict([(\'b\', (2, [3, 4])), (\'a\', 1), (\'c\', None), (\'d\', 5)])\n    >>> tree_flatten(tree)\n    ([2, 3, 4, 1, None, 5], PyTreeSpec(OrderedDict([(\'b\', (*, [*, *])), (\'a\', *), (\'c\', *), (\'d\', *)]), NoneIsLeaf))\n    >>> tree_flatten(tree, none_is_leaf=False)\n    ([2, 3, 4, 1, 5], PyTreeSpec(OrderedDict([(\'b\', (*, [*, *])), (\'a\', *), (\'c\', None), (\'d\', *)])))\n\n    Args:\n        tree (pytree): A pytree to flatten.\n        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,\n            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the\n            treespec rather than in the leaves list. (default: :data:`True`)\n        namespace (str, optional): The registry namespace used for custom pytree node types.\n            (default: :const:`"torch"`)\n\n    Returns:\n        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the\n        second element is a treespec representing the structure of the pytree.\n    '
    return optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)

def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    if False:
        i = 10
        return i + 15
    "Reconstruct a pytree from the treespec and the leaves.\n\n    The inverse of :func:`tree_flatten`.\n\n    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}\n    >>> leaves, treespec = tree_flatten(tree)\n    >>> tree == tree_unflatten(leaves, treespec)\n    True\n\n    Args:\n        leaves (iterable): The list of leaves to use for reconstruction. The list must match the\n            number of leaves of the treespec.\n        treespec (TreeSpec): The treespec to reconstruct.\n\n    Returns:\n        The reconstructed pytree, containing the ``leaves`` placed in the structure described by\n        ``treespec``.\n    "
    if not isinstance(treespec, TreeSpec):
        raise TypeError(f'tree_unflatten(values, spec): Expected `spec` to be instance of TreeSpec but got item of type {type(treespec)}.')
    return optree.tree_unflatten(treespec, leaves)

def tree_leaves(tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> List[Any]:
    if False:
        print('Hello World!')
    'Get the leaves of a pytree.\n\n    See also :func:`tree_flatten`.\n\n    >>> tree = {\'b\': (2, [3, 4]), \'a\': 1, \'c\': None, \'d\': 5}\n    >>> tree_leaves(tree)\n    [1, 2, 3, 4, None, 5]\n    >>> tree_leaves(tree, none_is_leaf=False)\n    [1, 2, 3, 4, 5]\n    >>> tree_leaves(1)\n    [1]\n    >>> tree_leaves(None)\n    [None]\n    >>> tree_leaves(None, none_is_leaf=False)\n    []\n\n    Args:\n        tree (pytree): A pytree to flatten.\n        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,\n            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the\n            treespec rather than in the leaves list. (default: :data:`True`)\n        namespace (str, optional): The registry namespace used for custom pytree node types.\n            (default: :const:`"torch"`)\n\n    Returns:\n        A list of leaf values.\n    '
    return optree.tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)

def tree_structure(tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> TreeSpec:
    if False:
        return 10
    'Get the treespec for a pytree.\n\n    See also :func:`tree_flatten`.\n\n    >>> tree = {\'b\': (2, [3, 4]), \'a\': 1, \'c\': None, \'d\': 5}\n    >>> tree_structure(tree)\n    PyTreeSpec({\'a\': *, \'b\': (*, [*, *]), \'c\': *, \'d\': *}, NoneIsLeaf)\n    >>> tree_structure(tree, none_is_leaf=False)\n    PyTreeSpec({\'a\': *, \'b\': (*, [*, *]), \'c\': None, \'d\': *})\n    >>> tree_structure(1)\n    PyTreeSpec(*, NoneIsLeaf)\n    >>> tree_structure(None)\n    PyTreeSpec(*, NoneIsLeaf)\n    >>> tree_structure(None, none_is_leaf=False)\n    PyTreeSpec(None)\n\n    Args:\n        tree (pytree): A pytree to flatten.\n        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,\n            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the\n            treespec rather than in the leaves list. (default: :data:`True`)\n        namespace (str, optional): The registry namespace used for custom pytree node types.\n            (default: :const:`"torch"`)\n\n    Returns:\n        A treespec object representing the structure of the pytree.\n    '
    return optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)

def tree_map(func: Callable[..., Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        print('Hello World!')
    'Map a multi-input function over pytree args to produce a new pytree.\n\n    See also :func:`tree_map_`.\n\n    >>> tree_map(lambda x: x + 1, {\'x\': 7, \'y\': (42, 64)})\n    {\'x\': 8, \'y\': (43, 65)}\n    >>> tree_map(lambda x: x is None, {\'x\': 7, \'y\': (42, 64), \'z\': None})\n    {\'x\': False, \'y\': (False, False), \'z\': True}\n    >>> tree_map(lambda x: x + 1, {\'x\': 7, \'y\': (42, 64), \'z\': None}, none_is_leaf=False)\n    {\'x\': 8, \'y\': (43, 65), \'z\': None}\n    >>> tree_map(lambda x: x is None, {\'x\': 7, \'y\': (42, 64), \'z\': None}, none_is_leaf=False)\n    {\'x\': False, \'y\': (False, False), \'z\': None}\n\n    If multiple inputs are given, the structure of the tree is taken from the first input;\n    subsequent inputs need only have ``tree`` as a prefix:\n\n    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])\n    [[5, 7, 9], [6, 1, 2]]\n\n    Args:\n        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the\n            corresponding leaves of the pytrees.\n        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional\n            argument to function ``func``.\n        rests (tuple of pytrees): A tuple of pytrees, each of which has the same structure as\n            ``tree`` or has ``tree`` as a prefix.\n        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,\n            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the\n            treespec rather than in the leaves list. (default: :data:`True`)\n        namespace (str, optional): The registry namespace used for custom pytree node types.\n            (default: :const:`"torch"`)\n\n    Returns:\n        A new pytree with the same structure as ``tree`` but with the value at each leaf given by\n        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``\n        is the tuple of values at corresponding nodes in ``rests``.\n    '
    return optree.tree_map(func, tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace)

def tree_map_(func: Callable[..., Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        while True:
            i = 10
    'Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.\n\n    See also :func:`tree_map`.\n\n    Args:\n        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the\n            corresponding leaves of the pytrees.\n        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional\n            argument to function ``func``.\n        rests (tuple of pytrees): A tuple of pytrees, each of which has the same structure as\n            ``tree`` or has ``tree`` as a prefix.\n        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,\n            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the\n            treespec rather than in the leaves list. (default: :data:`True`)\n        namespace (str, optional): The registry namespace used for custom pytree node types.\n            (default: :const:`"torch"`)\n\n    Returns:\n        The original ``tree`` with the value at each leaf is given by the side-effect of function\n        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf\n        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.\n    '
    return optree.tree_map_(func, tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace)
Type2 = Tuple[Type[T], Type[S]]
Type3 = Tuple[Type[T], Type[S], Type[U]]
TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]
Fn2 = Callable[[Union[T, S]], R]
Fn3 = Callable[[Union[T, S, U]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]
MapOnlyFn = Callable[[T], Callable[[Any], Any]]

@overload
def map_only(__type_or_types: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    if False:
        while True:
            i = 10
    ...

@overload
def map_only(__type_or_types: Type3[T, S, U]) -> MapOnlyFn[Fn3[T, S, U, Any]]:
    if False:
        while True:
            i = 10
    ...

@overload
def map_only(__type_or_types: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    if False:
        print('Hello World!')
    ...

@overload
def map_only(__type_or_types: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    if False:
        for i in range(10):
            print('nop')
    ...

def map_only(__type_or_types: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    if False:
        return 10
    "\n    Suppose you are writing a tree_map over tensors, leaving everything\n    else unchanged.  Ordinarily you would have to write:\n\n        def go(t):\n            if isinstance(t, Tensor):\n                return ...\n            else:\n                return t\n\n    With this function, you only need to write:\n\n        @map_only(Tensor)\n        def go(t):\n            return ...\n\n    You can also directly use 'tree_map_only'\n    "

    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        if False:
            while True:
                i = 10

        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            if False:
                print('Hello World!')
            if isinstance(x, __type_or_types):
                return func(x)
            return x
        return wrapped
    return wrapper

@overload
def tree_map_only(__type_or_types: Type[T], func: Fn[T, Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        return 10
    ...

@overload
def tree_map_only(__type_or_types: Type2[T, S], func: Fn2[T, S, Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        i = 10
        return i + 15
    ...

@overload
def tree_map_only(__type_or_types: Type3[T, S, U], func: Fn3[T, S, U, Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        i = 10
        return i + 15
    ...

def tree_map_only(__type_or_types: TypeAny, func: FnAny[Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        for i in range(10):
            print('nop')
    return tree_map(map_only(__type_or_types)(func), tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace)

@overload
def tree_map_only_(__type_or_types: Type[T], func: Fn[T, Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def tree_map_only_(__type_or_types: Type2[T, S], func: Fn2[T, S, Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        while True:
            i = 10
    ...

@overload
def tree_map_only_(__type_or_types: Type3[T, S, U], func: Fn3[T, S, U, Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        for i in range(10):
            print('nop')
    ...

def tree_map_only_(__type_or_types: TypeAny, func: FnAny[Any], tree: PyTree, *rests: PyTree, none_is_leaf: bool=True, namespace: str='torch') -> PyTree:
    if False:
        for i in range(10):
            print('nop')
    return tree_map_(map_only(__type_or_types)(func), tree, *rests, none_is_leaf=none_is_leaf, namespace=namespace)

def tree_all(pred: Callable[[Any], bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        for i in range(10):
            print('nop')
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return all(map(pred, flat_args))

def tree_any(pred: Callable[[Any], bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        i = 10
        return i + 15
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return any(map(pred, flat_args))

@overload
def tree_all_only(__type_or_types: Type[T], pred: Fn[T, bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        i = 10
        return i + 15
    ...

@overload
def tree_all_only(__type_or_types: Type2[T, S], pred: Fn2[T, S, bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        return 10
    ...

@overload
def tree_all_only(__type_or_types: Type3[T, S, U], pred: Fn3[T, S, U, bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        for i in range(10):
            print('nop')
    ...

def tree_all_only(__type_or_types: TypeAny, pred: FnAny[bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        while True:
            i = 10
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return all((pred(x) for x in flat_args if isinstance(x, __type_or_types)))

@overload
def tree_any_only(__type_or_types: Type[T], pred: Fn[T, bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        print('Hello World!')
    ...

@overload
def tree_any_only(__type_or_types: Type2[T, S], pred: Fn2[T, S, bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        while True:
            i = 10
    ...

@overload
def tree_any_only(__type_or_types: Type3[T, S, U], pred: Fn3[T, S, U, bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        while True:
            i = 10
    ...

def tree_any_only(__type_or_types: TypeAny, pred: FnAny[bool], tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> bool:
    if False:
        i = 10
        return i + 15
    flat_args = tree_leaves(tree, none_is_leaf=none_is_leaf, namespace=namespace)
    return any((pred(x) for x in flat_args if isinstance(x, __type_or_types)))

def broadcast_prefix(prefix_tree: PyTree, full_tree: PyTree, *, none_is_leaf: bool=True, namespace: str='torch') -> List[Any]:
    if False:
        i = 10
        return i + 15
    'Return a list of broadcasted leaves in ``prefix_tree`` to match the number of leaves in ``full_tree``.\n\n    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be\n    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.\n\n    This function returns a list of leaves with the same size as ``full_tree``. The leaves are\n    replicated from ``prefix_tree``. The number of replicas is determined by the corresponding\n    subtree in ``full_tree``.\n\n    >>> broadcast_prefix(1, [1, 2, 3])\n    [1, 1, 1]\n    >>> broadcast_prefix([1, 2, 3], [1, 2, 3])\n    [1, 2, 3]\n    >>> broadcast_prefix([1, 2, 3], [1, 2, 3, 4])\n    Traceback (most recent call last):\n        ...\n    ValueError: list arity mismatch; expected: 3, got: 4; list: [1, 2, 3, 4].\n    >>> broadcast_prefix([1, 2, 3], [1, 2, (3, 4)])\n    [1, 2, 3, 3]\n    >>> broadcast_prefix([1, 2, 3], [1, 2, {\'a\': 3, \'b\': 4, \'c\': (None, 5)}])\n    [1, 2, 3, 3, 3, 3]\n    >>> broadcast_prefix([1, 2, 3], [1, 2, {\'a\': 3, \'b\': 4, \'c\': (None, 5)}], none_is_leaf=False)\n    [1, 2, 3, 3, 3]\n\n    Args:\n        prefix_tree (pytree): A pytree with the same structure as a prefix of ``full_tree``.\n        full_tree (pytree): A pytree with the same structure as a suffix of ``prefix_tree``.\n        is_leaf (callable, optional): An optionally specified function that will be called at each\n            flattening step. It should return a boolean, with :data:`True` stopping the traversal\n            and the whole subtree being treated as a leaf, and :data:`False` indicating the\n            flattening should traverse the current object.\n        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,\n            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the\n            treespec rather than in the leaves list. (default: :data:`True`)\n        namespace (str, optional): The registry namespace used for custom pytree node types.\n            (default: :const:`"torch"`)\n\n    Returns:\n        A list of leaves in ``prefix_tree`` broadcasted to match the number of leaves in ``full_tree``.\n    '
    return optree.broadcast_prefix(prefix_tree, full_tree, none_is_leaf=none_is_leaf, namespace=namespace)

def _broadcast_to_and_flatten(tree: PyTree, treespec: TreeSpec, *, none_is_leaf: bool=True, namespace: str='torch') -> Optional[List[Any]]:
    if False:
        return 10
    assert isinstance(treespec, TreeSpec)
    full_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    try:
        return broadcast_prefix(tree, full_tree, none_is_leaf=none_is_leaf, namespace=namespace)
    except ValueError:
        return None

def treespec_dumps(treespec: TreeSpec) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Serialize a treespec to a JSON string.'
    if not isinstance(treespec, TreeSpec):
        raise TypeError(f'treespec_dumps(spec): Expected `spec` to be instance of TreeSpec but got item of type {type(treespec)}.')
    from ._pytree import tree_structure as _tree_structure, treespec_dumps as _treespec_dumps
    orig_treespec = _tree_structure(tree_unflatten([0] * treespec.num_leaves, treespec))
    return _treespec_dumps(orig_treespec)

def treespec_loads(serialized: str) -> TreeSpec:
    if False:
        return 10
    'Deserialize a treespec from a JSON string.'
    from ._pytree import tree_unflatten as _tree_unflatten, treespec_loads as _treespec_loads
    orig_treespec = _treespec_loads(serialized)
    dummy_tree = _tree_unflatten([0] * orig_treespec.num_leaves, orig_treespec)
    treespec = tree_structure(dummy_tree)
    return treespec

class _DummyLeaf:

    def __repr__(self) -> str:
        if False:
            return 10
        return '*'

def treespec_pprint(treespec: TreeSpec) -> str:
    if False:
        for i in range(10):
            print('nop')
    dummy_tree = tree_unflatten([_DummyLeaf() for _ in range(treespec.num_leaves)], treespec)
    return repr(dummy_tree)

class LeafSpecMeta(type(TreeSpec)):

    def __instancecheck__(self, instance: object) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(instance, TreeSpec) and instance.is_leaf()

class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):

    def __new__(cls, none_is_leaf: bool=True) -> 'LeafSpec':
        if False:
            i = 10
            return i + 15
        return optree.treespec_leaf(none_is_leaf=none_is_leaf)