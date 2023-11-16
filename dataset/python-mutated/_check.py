import ast
import inspect
import textwrap
import warnings
import torch

class AttributeTypeIsSupportedChecker(ast.NodeVisitor):
    """Check the ``__init__`` method of a given ``nn.Module``.

    It ensures that all instance-level attributes can be properly initialized.

    Specifically, we do type inference based on attribute values...even
    if the attribute in question has already been typed using
    Python3-style annotations or ``torch.jit.annotate``. This means that
    setting an instance-level attribute to ``[]`` (for ``List``),
    ``{}`` for ``Dict``), or ``None`` (for ``Optional``) isn't enough
    information for us to properly initialize that attribute.

    An object of this class can walk a given ``nn.Module``'s AST and
    determine if it meets our requirements or not.

    Known limitations
    1. We can only check the AST nodes for certain constructs; we can't
    ``eval`` arbitrary expressions. This means that function calls,
    class instantiations, and complex expressions that resolve to one of
    the "empty" values specified above will NOT be flagged as
    problematic.
    2. We match on string literals, so if the user decides to use a
    non-standard import (e.g. `from typing import List as foo`), we
    won't catch it.

    Example:
        .. code-block:: python

            class M(torch.nn.Module):
                def fn(self):
                    return []

                def __init__(self):
                    super().__init__()
                    self.x: List[int] = []

                def forward(self, x: List[int]):
                    self.x = x
                    return 1

        The above code will pass the ``AttributeTypeIsSupportedChecker``
        check since we have a function call in ``__init__``. However,
        it will still fail later with the ``RuntimeError`` "Tried to set
        nonexistent attribute: x. Did you forget to initialize it in
        __init__()?".

    Args:
        nn_module - The instance of ``torch.nn.Module`` whose
            ``__init__`` method we wish to check
    """

    def check(self, nn_module: torch.nn.Module) -> None:
        if False:
            for i in range(10):
                print('nop')
        source_lines = inspect.getsource(nn_module.__class__.__init__)

        def is_useless_comment(line):
            if False:
                while True:
                    i = 10
            line = line.strip()
            return line.startswith('#') and (not line.startswith('# type:'))
        source_lines = '\n'.join([l for l in source_lines.split('\n') if not is_useless_comment(l)])
        init_ast = ast.parse(textwrap.dedent(source_lines))
        self.class_level_annotations = list(nn_module.__annotations__.keys())
        self.visiting_class_level_ann = False
        self.visit(init_ast)

    def _is_empty_container(self, node: ast.AST, ann_type: str) -> bool:
        if False:
            i = 10
            return i + 15
        if ann_type == 'List':
            if not isinstance(node, ast.List):
                return False
            if node.elts:
                return False
        elif ann_type == 'Dict':
            if not isinstance(node, ast.Dict):
                return False
            if node.keys:
                return False
        elif ann_type == 'Optional':
            if not isinstance(node, ast.Constant):
                return False
            if node.value:
                return False
        return True

    def visit_Assign(self, node):
        if False:
            return 10
        "Store assignment state when assigning to a Call Node.\n\n        If we're visiting a Call Node (the right-hand side of an\n        assignment statement), we won't be able to check the variable\n        that we're assigning to (the left-hand side of an assignment).\n        Because of this, we need to store this state in visitAssign.\n        (Luckily, we only have to do this if we're assigning to a Call\n        Node, i.e. ``torch.jit.annotate``. If we're using normal Python\n        annotations, we'll be visiting an AnnAssign Node, which has its\n        target built in.)\n        "
        try:
            if isinstance(node.value, ast.Call) and node.targets[0].attr in self.class_level_annotations:
                self.visiting_class_level_ann = True
        except AttributeError:
            return
        self.generic_visit(node)
        self.visiting_class_level_ann = False

    def visit_AnnAssign(self, node):
        if False:
            print('Hello World!')
        "Visit an AnnAssign node in an ``nn.Module``'s ``__init__`` method.\n\n        It checks if it conforms to our attribute annotation rules."
        try:
            if node.target.value.id != 'self':
                return
        except AttributeError:
            return
        if node.target.attr in self.class_level_annotations:
            return
        containers = {'List', 'Dict', 'Optional'}
        try:
            if node.annotation.value.id not in containers:
                return
        except AttributeError:
            return
        ann_type = node.annotation.value.id
        if not self._is_empty_container(node.value, ann_type):
            return
        warnings.warn("The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.")

    def visit_Call(self, node):
        if False:
            return 10
        "Determine if a Call node is 'torch.jit.annotate' in __init__.\n\n        Visit a Call node in an ``nn.Module``'s ``__init__``\n        method and determine if it's ``torch.jit.annotate``. If so,\n        see if it conforms to our attribute annotation rules.\n        "
        if self.visiting_class_level_ann:
            return
        try:
            if node.func.value.value.id != 'torch' or node.func.value.attr != 'jit' or node.func.attr != 'annotate':
                self.generic_visit(node)
            elif node.func.value.value.id != 'jit' or node.func.value.attr != 'annotate':
                self.generic_visit(node)
        except AttributeError:
            self.generic_visit(node)
        if len(node.args) != 2:
            return
        if not isinstance(node.args[0], ast.Subscript):
            return
        containers = {'List', 'Dict', 'Optional'}
        try:
            ann_type = node.args[0].value.id
        except AttributeError:
            return
        if ann_type not in containers:
            return
        if not self._is_empty_container(node.args[1], ann_type):
            return
        warnings.warn("The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.")