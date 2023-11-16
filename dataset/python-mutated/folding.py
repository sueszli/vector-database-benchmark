from typing import Optional, Union
from vyper.ast import nodes as vy_ast
from vyper.builtins.functions import DISPATCH_TABLE
from vyper.exceptions import UnfoldableNode, UnknownType
from vyper.semantics.types.base import VyperType
from vyper.semantics.types.utils import type_from_annotation

def fold(vyper_module: vy_ast.Module) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform literal folding operations on a Vyper AST.\n\n    Arguments\n    ---------\n    vyper_module : Module\n        Top-level Vyper AST node.\n    '
    changed_nodes = 1
    while changed_nodes:
        changed_nodes = 0
        changed_nodes += replace_user_defined_constants(vyper_module)
        changed_nodes += replace_literal_ops(vyper_module)
        changed_nodes += replace_subscripts(vyper_module)
        changed_nodes += replace_builtin_functions(vyper_module)

def replace_literal_ops(vyper_module: vy_ast.Module) -> int:
    if False:
        print('Hello World!')
    '\n    Find and evaluate operation and comparison nodes within the Vyper AST,\n    replacing them with Constant nodes where possible.\n\n    Arguments\n    ---------\n    vyper_module : Module\n        Top-level Vyper AST node.\n\n    Returns\n    -------\n    int\n        Number of nodes that were replaced.\n    '
    changed_nodes = 0
    node_types = (vy_ast.BoolOp, vy_ast.BinOp, vy_ast.UnaryOp, vy_ast.Compare)
    for node in vyper_module.get_descendants(node_types, reverse=True):
        try:
            new_node = node.evaluate()
        except UnfoldableNode:
            continue
        changed_nodes += 1
        vyper_module.replace_in_tree(node, new_node)
    return changed_nodes

def replace_subscripts(vyper_module: vy_ast.Module) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Find and evaluate Subscript nodes within the Vyper AST, replacing them with\n    Constant nodes where possible.\n\n    Arguments\n    ---------\n    vyper_module : Module\n        Top-level Vyper AST node.\n\n    Returns\n    -------\n    int\n        Number of nodes that were replaced.\n    '
    changed_nodes = 0
    for node in vyper_module.get_descendants(vy_ast.Subscript, reverse=True):
        try:
            new_node = node.evaluate()
        except UnfoldableNode:
            continue
        changed_nodes += 1
        vyper_module.replace_in_tree(node, new_node)
    return changed_nodes

def replace_builtin_functions(vyper_module: vy_ast.Module) -> int:
    if False:
        print('Hello World!')
    '\n    Find and evaluate builtin function calls within the Vyper AST, replacing\n    them with Constant nodes where possible.\n\n    Arguments\n    ---------\n    vyper_module : Module\n        Top-level Vyper AST node.\n\n    Returns\n    -------\n    int\n        Number of nodes that were replaced.\n    '
    changed_nodes = 0
    for node in vyper_module.get_descendants(vy_ast.Call, reverse=True):
        if not isinstance(node.func, vy_ast.Name):
            continue
        name = node.func.id
        func = DISPATCH_TABLE.get(name)
        if func is None or not hasattr(func, 'evaluate'):
            continue
        try:
            new_node = func.evaluate(node)
        except UnfoldableNode:
            continue
        changed_nodes += 1
        vyper_module.replace_in_tree(node, new_node)
    return changed_nodes

def replace_user_defined_constants(vyper_module: vy_ast.Module) -> int:
    if False:
        return 10
    '\n    Find user-defined constant assignments, and replace references\n    to the constants with their literal values.\n\n    Arguments\n    ---------\n    vyper_module : Module\n        Top-level Vyper AST node.\n\n    Returns\n    -------\n    int\n        Number of nodes that were replaced.\n    '
    changed_nodes = 0
    for node in vyper_module.get_children(vy_ast.VariableDecl):
        if not isinstance(node.target, vy_ast.Name):
            continue
        if not node.is_constant:
            continue
        type_ = None
        try:
            type_ = type_from_annotation(node.annotation)
        except UnknownType:
            pass
        changed_nodes += replace_constant(vyper_module, node.target.id, node.value, False, type_=type_)
    return changed_nodes

def _replace(old_node, new_node, type_=None):
    if False:
        print('Hello World!')
    if isinstance(new_node, vy_ast.Constant):
        new_node = new_node.from_node(old_node, value=new_node.value)
        if type_:
            new_node._metadata['type'] = type_
        return new_node
    elif isinstance(new_node, vy_ast.List):
        base_type = type_.value_type if type_ else None
        list_values = [_replace(old_node, i, type_=base_type) for i in new_node.elements]
        new_node = new_node.from_node(old_node, elements=list_values)
        if type_:
            new_node._metadata['type'] = type_
        return new_node
    elif isinstance(new_node, vy_ast.Call):
        keyword = keywords = None
        if hasattr(new_node, 'keyword'):
            keyword = new_node.keyword
        if hasattr(new_node, 'keywords'):
            keywords = new_node.keywords
        new_node = new_node.from_node(old_node, func=new_node.func, args=new_node.args, keyword=keyword, keywords=keywords)
        return new_node
    else:
        raise UnfoldableNode

def replace_constant(vyper_module: vy_ast.Module, id_: str, replacement_node: Union[vy_ast.Constant, vy_ast.List, vy_ast.Call], raise_on_error: bool, type_: Optional[VyperType]=None) -> int:
    if False:
        while True:
            i = 10
    '\n    Replace references to a variable name with a literal value.\n\n    Arguments\n    ---------\n    vyper_module : Module\n        Module-level ast node to perform replacement in.\n    id_ : str\n        String representing the `.id` attribute of the node(s) to be replaced.\n    replacement_node : Constant | List | Call\n        Vyper ast node representing the literal value to be substituted in.\n        `Call` nodes are for struct constants.\n    raise_on_error: bool\n        Boolean indicating if `UnfoldableNode` exception should be raised or ignored.\n    type_ : VyperType, optional\n        Type definition to be propagated to type checker.\n\n    Returns\n    -------\n    int\n        Number of nodes that were replaced.\n    '
    changed_nodes = 0
    for node in vyper_module.get_descendants(vy_ast.Name, {'id': id_}, reverse=True):
        parent = node.get_ancestor()
        if isinstance(parent, vy_ast.Call) and node == parent.func:
            continue
        if isinstance(parent, vy_ast.Dict) and node in parent.keys:
            continue
        if not node.get_ancestor(vy_ast.Index):
            assign = node.get_ancestor((vy_ast.Assign, vy_ast.AnnAssign, vy_ast.AugAssign, vy_ast.VariableDecl))
            if assign and node in assign.target.get_descendants(include_self=True):
                continue
        if node.get_ancestor(vy_ast.EnumDef):
            continue
        try:
            new_node = _replace(node, replacement_node, type_=type_)
        except UnfoldableNode:
            if raise_on_error:
                raise
            continue
        changed_nodes += 1
        vyper_module.replace_in_tree(node, new_node)
    return changed_nodes