"""
Functions for *walking* the DOM.

!!! note

    For most purposes you would be better off using [query][textual.dom.DOMNode.query], which uses these functions internally.
"""
from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING, Iterable, Iterator, TypeVar, overload
if TYPE_CHECKING:
    from textual.dom import DOMNode
    WalkType = TypeVar('WalkType', bound=DOMNode)

@overload
def walk_depth_first(root: DOMNode, *, with_root: bool=True) -> Iterable[DOMNode]:
    if False:
        while True:
            i = 10
    ...

@overload
def walk_depth_first(root: WalkType, filter_type: type[WalkType], *, with_root: bool=True) -> Iterable[WalkType]:
    if False:
        while True:
            i = 10
    ...

def walk_depth_first(root: DOMNode, filter_type: type[WalkType] | None=None, *, with_root: bool=True) -> Iterable[DOMNode] | Iterable[WalkType]:
    if False:
        while True:
            i = 10
    "Walk the tree depth first (parents first).\n\n    !!! note\n\n        Avoid changing the DOM (mounting, removing etc.) while iterating with this function.\n        Consider [walk_children][textual.dom.DOMNode.walk_children] which doesn't have this limitation.\n\n    Args:\n        root: The root note (starting point).\n        filter_type: Optional DOMNode subclass to filter by, or ``None`` for no filter.\n        with_root: Include the root in the walk.\n\n    Returns:\n        An iterable of DOMNodes, or the type specified in ``filter_type``.\n    "
    from textual.dom import DOMNode
    stack: list[Iterator[DOMNode]] = [iter(root.children)]
    pop = stack.pop
    push = stack.append
    check_type = filter_type or DOMNode
    if with_root and isinstance(root, check_type):
        yield root
    while stack:
        node = next(stack[-1], None)
        if node is None:
            pop()
        else:
            if isinstance(node, check_type):
                yield node
            if node.children:
                push(iter(node.children))

@overload
def walk_breadth_first(root: DOMNode, *, with_root: bool=True) -> Iterable[DOMNode]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def walk_breadth_first(root: WalkType, filter_type: type[WalkType], *, with_root: bool=True) -> Iterable[WalkType]:
    if False:
        i = 10
        return i + 15
    ...

def walk_breadth_first(root: DOMNode, filter_type: type[WalkType] | None=None, *, with_root: bool=True) -> Iterable[DOMNode] | Iterable[WalkType]:
    if False:
        print('Hello World!')
    "Walk the tree breadth first (children first).\n\n    !!! note\n\n        Avoid changing the DOM (mounting, removing etc.) while iterating with this function.\n        Consider [walk_children][textual.dom.DOMNode.walk_children] which doesn't have this limitation.\n\n    Args:\n        root: The root note (starting point).\n        filter_type: Optional DOMNode subclass to filter by, or ``None`` for no filter.\n        with_root: Include the root in the walk.\n\n    Returns:\n        An iterable of DOMNodes, or the type specified in ``filter_type``.\n    "
    from textual.dom import DOMNode
    queue: deque[DOMNode] = deque()
    popleft = queue.popleft
    extend = queue.extend
    check_type = filter_type or DOMNode
    if with_root and isinstance(root, check_type):
        yield root
    extend(root.children)
    while queue:
        node = popleft()
        if isinstance(node, check_type):
            yield node
        extend(node.children)