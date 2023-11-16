"""Utility methods for docstring checking."""
from __future__ import annotations
import re
import astroid
from pylint.checkers import utils
from pylint.extensions import _check_docs_utils
from typing import Optional, Set

def space_indentation(s: str) -> int:
    if False:
        i = 10
        return i + 15
    'The number of leading spaces in a string\n\n    Args:\n        s: str. The input string.\n\n    Returns:\n        int. The number of leading spaces.\n    '
    return len(s) - len(s.lstrip(' '))

def get_setters_property_name(node: astroid.FunctionDef) -> Optional[str]:
    if False:
        print('Hello World!')
    'Get the name of the property that the given node is a setter for.\n\n    Args:\n        node: astroid.FunctionDef. The node to get the property name for.\n\n    Returns:\n        str|None. The name of the property that the node is a setter for,\n        or None if one could not be found.\n    '
    decorator_nodes = node.decorators.nodes if node.decorators else []
    for decorator_node in decorator_nodes:
        if isinstance(decorator_node, astroid.Attribute) and decorator_node.attrname == 'setter' and isinstance(decorator_node.expr, astroid.Name):
            decorator_name: Optional[str] = decorator_node.expr.name
            return decorator_name
    return None

def get_setters_property(node: astroid.FunctionDef) -> Optional[astroid.FunctionDef]:
    if False:
        print('Hello World!')
    'Get the property node for the given setter node.\n\n    Args:\n        node: astroid.FunctionDef. The node to get the property for.\n\n    Returns:\n        astroid.FunctionDef|None. The node relating to the property of\n        the given setter node, or None if one could not be found.\n    '
    setters_property = None
    property_name = get_setters_property_name(node)
    class_node = utils.node_frame_class(node)
    if property_name and class_node:
        class_attrs = class_node.getattr(node.name)
        for attr in class_attrs:
            if utils.decorated_with_property(attr):
                setters_property = attr
                break
    return setters_property

def returns_something(return_node: astroid.Return) -> bool:
    if False:
        return 10
    'Check if a return node returns a value other than None.\n\n    Args:\n        return_node: astroid.Return. The return node to check.\n\n    Returns:\n        bool. True if the return node returns a value other than None, False\n        otherwise.\n    '
    returns = return_node.value
    if returns is None:
        return False
    return not (isinstance(returns, astroid.Const) and returns.value is None)

def possible_exc_types(node: astroid.NodeNG) -> Set[str]:
    if False:
        while True:
            i = 10
    'Gets all of the possible raised exception types for the given raise node.\n    Caught exception types are ignored.\n\n    Args:\n        node: astroid.node_classes.NodeNG. The raise\n            to find exception types for.\n\n    Returns:\n        set(str). A list of exception types.\n    '
    excs = []
    if isinstance(node.exc, astroid.Name):
        inferred = utils.safe_infer(node.exc)
        if inferred:
            excs = [inferred.name]
    elif isinstance(node.exc, astroid.Call) and isinstance(node.exc.func, astroid.Name):
        target = utils.safe_infer(node.exc.func)
        if isinstance(target, astroid.ClassDef):
            excs = [target.name]
        elif isinstance(target, astroid.FunctionDef):
            for ret in target.nodes_of_class(astroid.Return):
                if ret.frame() != target:
                    continue
                val = utils.safe_infer(ret.value)
                if val and isinstance(val, (astroid.Instance, astroid.ClassDef)) and utils.inherit_from_std_ex(val):
                    excs.append(val.name)
    elif node.exc is None:
        handler = node.parent
        while handler and (not isinstance(handler, astroid.ExceptHandler)):
            handler = handler.parent
        if handler and handler.type:
            inferred_excs = astroid.unpack_infer(handler.type)
            excs = [exc.name for exc in inferred_excs if exc is not astroid.Uninferable]
    try:
        return set((exc for exc in excs if not utils.node_ignores_exception(node, exc)))
    except astroid.InferenceError:
        return set()

def docstringify(docstring: astroid.nodes.Const) -> _check_docs_utils.Docstring:
    if False:
        print('Hello World!')
    "Converts a docstring node to its Docstring object\n    as defined in the pylint library.\n\n    Args:\n        docstring: astroid.nodes.Const. Docstring for a particular class or\n            function.\n\n    Returns:\n        Docstring. Pylint Docstring class instance representing\n        a node's docstring.\n    "
    for docstring_type in [GoogleDocstring]:
        instance = docstring_type(docstring)
        if instance.matching_sections() > 0:
            return instance
    return _check_docs_utils.Docstring(docstring)

class GoogleDocstring(_check_docs_utils.GoogleDocstring):
    """Class for checking whether docstrings follow the Google Python Style
    Guide.
    """
    re_multiple_type = _check_docs_utils.GoogleDocstring.re_multiple_type
    re_param_line = re.compile('\n        \\s*  \\*{{0,2}}(\\w+)             # identifier potentially with asterisks\n        \\s*  ( [:]\n            \\s*\n            ({type}|\\S*)\n            (?:,\\s+optional)?\n            [.] )? \\s*                  # optional type declaration\n        \\s*  (.*)                       # beginning of optional description\n    '.format(type=re_multiple_type), flags=re.X | re.S | re.M)
    re_returns_line = re.compile('\n        \\s* (({type}|\\S*).)?              # identifier\n        \\s* (.*)                          # beginning of description\n    '.format(type=re_multiple_type), flags=re.X | re.S | re.M)
    re_yields_line = re_returns_line
    re_raise_line = re.compile('\n        \\s* ({type}|\\S*)?[.:]                    # identifier\n        \\s* (.*)                         # beginning of description\n    '.format(type=re_multiple_type), flags=re.X | re.S | re.M)