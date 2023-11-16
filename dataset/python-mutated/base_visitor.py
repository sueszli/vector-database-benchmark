"""Base class for visitors."""
import re
from typing import Any
from pytype.pytd import pytd
from pytype.typegraph import cfg_utils
ALL_NODE_NAMES = type('contains_everything', (), {'__contains__': lambda *args: True})()

class _NodeClassInfo:
    """Representation of a node class in the graph."""

    def __init__(self, cls):
        if False:
            print('Hello World!')
        self.cls = cls
        self.name = cls.__name__
        self.outgoing = set()

def _FindNodeClasses():
    if False:
        while True:
            i = 10
    'Yields _NodeClassInfo objects for each node found in pytd.'
    for name in dir(pytd):
        value = getattr(pytd, name)
        if isinstance(value, type) and issubclass(value, pytd.Node) and (value is not pytd.Node):
            yield _NodeClassInfo(value)
_IGNORED_TYPES = frozenset([str, bool, int, type(None), Any])
_ancestor_map = None

def _GetChildTypes(node_classes, cls: Any):
    if False:
        while True:
            i = 10
    "Get all the types that can be in a node's subtree."
    types = set()

    def AddType(t: Any):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(t, '__args__'):
            for x in t.__args__:
                if x is not Ellipsis:
                    AddType(x)
            return
        if hasattr(t, '__forward_arg__'):
            t = t.__forward_arg__
        if isinstance(t, str) and t in node_classes:
            types.add(node_classes[t].cls)
        else:
            types.add(t)
    for field in cls.__attrs_attrs__:
        AddType(field.type)
    for x in types:
        assert isinstance(x, type) or x == Any
    return types

def _GetAncestorMap():
    if False:
        i = 10
        return i + 15
    'Return a map of node class names to a set of ancestor class names.'
    global _ancestor_map
    if _ancestor_map is None:
        node_classes = {i.name: i for i in _FindNodeClasses()}
        for info in node_classes.values():
            for allowed in _GetChildTypes(node_classes, info.cls):
                if allowed in _IGNORED_TYPES:
                    pass
                elif allowed.__module__ == 'pytype.pytd.pytd':
                    info.outgoing.update([i for i in node_classes.values() if issubclass(i.cls, allowed)])
                else:
                    raise AssertionError(f'Unknown child type: {allowed}')
        predecessors = cfg_utils.compute_predecessors(node_classes.values())
        get_names = lambda v: {n.name for n in v}
        _ancestor_map = {k.name: get_names(v) for (k, v) in predecessors.items()}
    return _ancestor_map

class Visitor:
    """Base class for visitors.

  Each class inheriting from visitor SHOULD have a fixed set of methods,
  otherwise it might break the caching in this class.

  Attributes:
    visits_all_node_types: Whether the visitor can visit every node type.
    unchecked_node_names: Contains the names of node classes that are unchecked
      when constructing a new node from visited children.  This is useful
      if a visitor returns data in part or all of its walk that would violate
      node preconditions.
    enter_functions: A dictionary mapping node class names to the
      corresponding Enter functions.
    visit_functions: A dictionary mapping node class names to the
      corresponding Visit functions.
    leave_functions: A dictionary mapping node class names to the
      corresponding Leave functions.
    visit_class_names: A set of node class names that must be visited.  This is
      constructed based on the enter/visit/leave functions and precondition
      data about legal ASTs.  As an optimization, the visitor will only visit
      nodes under which some actionable node can appear.
  """
    old_node: Any
    visits_all_node_types = False
    unchecked_node_names = set()
    _visitor_functions_cache = {}

    def __init__(self):
        if False:
            return 10
        cls = self.__class__
        if cls in Visitor._visitor_functions_cache:
            (enter_fns, visit_fns, leave_fns, visit_class_names) = Visitor._visitor_functions_cache[cls]
        else:
            enter_fns = {}
            enter_prefix = 'Enter'
            enter_len = len(enter_prefix)
            visit_fns = {}
            visit_prefix = 'Visit'
            visit_len = len(visit_prefix)
            leave_fns = {}
            leave_prefix = 'Leave'
            leave_len = len(leave_prefix)
            for attrib in dir(cls):
                if attrib.startswith(enter_prefix):
                    enter_fns[attrib[enter_len:]] = getattr(cls, attrib)
                elif attrib.startswith(visit_prefix):
                    visit_fns[attrib[visit_len:]] = getattr(cls, attrib)
                elif attrib.startswith(leave_prefix):
                    leave_fns[attrib[leave_len:]] = getattr(cls, attrib)
            ancestors = _GetAncestorMap()
            visit_class_names = set()
            visit_all = cls.Enter != Visitor.Enter or cls.Visit != Visitor.Visit or cls.Leave != Visitor.Leave
            for node in set(enter_fns) | set(visit_fns) | set(leave_fns):
                if node in ancestors:
                    visit_class_names.update(ancestors[node])
                elif node:
                    if node == 'StrictType':
                        visit_all = True
                    elif cls.__module__ == '__main__' or re.fullmatch('.*(_test|test_[^\\.]+)', cls.__module__):
                        visit_all = True
                    else:
                        raise AssertionError(f'Unknown node type: {node} {cls!r}')
            if visit_all:
                visit_class_names = ALL_NODE_NAMES
            Visitor._visitor_functions_cache[cls] = (enter_fns, visit_fns, leave_fns, visit_class_names)
        self.enter_functions = enter_fns
        self.visit_functions = visit_fns
        self.leave_functions = leave_fns
        self.visit_class_names = visit_class_names

    def Enter(self, node, *args, **kwargs):
        if False:
            return 10
        return self.enter_functions[node.__class__.__name__](self, node, *args, **kwargs)

    def Visit(self, node, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.visit_functions[node.__class__.__name__](self, node, *args, **kwargs)

    def Leave(self, node, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.leave_functions[node.__class__.__name__](self, node, *args, **kwargs)