import ast
import spack.directives
import spack.error
import spack.package_base
import spack.repo
import spack.spec
import spack.util.hash
import spack.util.naming
from spack.util.unparse import unparse

class RemoveDocstrings(ast.NodeTransformer):
    """Transformer that removes docstrings from a Python AST.

    This removes *all* strings that aren't on the RHS of an assignment statement from
    the body of functions, classes, and modules -- even if they're not directly after
    the declaration.

    """

    def remove_docstring(self, node):
        if False:
            while True:
                i = 10

        def unused_string(node):
            if False:
                while True:
                    i = 10
            'Criteria for unassigned body strings.'
            return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)
        if node.body:
            node.body = [child for child in node.body if not unused_string(child)]
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.remove_docstring(node)

    def visit_ClassDef(self, node):
        if False:
            while True:
                i = 10
        return self.remove_docstring(node)

    def visit_Module(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self.remove_docstring(node)

class RemoveDirectives(ast.NodeTransformer):
    """Remove Spack directives from a package AST.

    This removes Spack directives (e.g., ``depends_on``, ``conflicts``, etc.) and
    metadata attributes (e.g., ``tags``, ``homepage``, ``url``) in a top-level class
    definition within a ``package.py``, but it does not modify nested classes or
    functions.

    If removing directives causes a ``for``, ``with``, or ``while`` statement to have an
    empty body, we remove the entire statement. Similarly, If removing directives causes
    an ``if`` statement to have an empty body or ``else`` block, we'll remove the block
    (or replace the body with ``pass`` if there is an ``else`` block but no body).

    """

    def __init__(self, spec):
        if False:
            i = 10
            return i + 15
        self.metadata_attrs = [s.url_attr for s in spack.fetch_strategy.all_strategies]
        self.metadata_attrs += spack.package_base.PackageBase.metadata_attrs
        self.spec = spec
        self.in_classdef = False

    def visit_Expr(self, node):
        if False:
            for i in range(10):
                print('nop')
        return None if node.value and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id in spack.directives.directive_names) else node

    def visit_Assign(self, node):
        if False:
            return 10
        return None if node.targets and isinstance(node.targets[0], ast.Name) and (node.targets[0].id in self.metadata_attrs) else node

    def visit_With(self, node):
        if False:
            i = 10
            return i + 15
        self.generic_visit(node)
        return node if node.body else None

    def visit_For(self, node):
        if False:
            i = 10
            return i + 15
        self.generic_visit(node)
        return node if node.body else None

    def visit_While(self, node):
        if False:
            i = 10
            return i + 15
        self.generic_visit(node)
        return node if node.body else None

    def visit_If(self, node):
        if False:
            while True:
                i = 10
        self.generic_visit(node)
        if not node.body:
            if node.orelse:
                node.body = [ast.Pass()]
            else:
                return None
        return node

    def visit_FunctionDef(self, node):
        if False:
            i = 10
            return i + 15
        return node

    def visit_ClassDef(self, node):
        if False:
            return 10
        if self.in_classdef:
            return node
        self.in_classdef = True
        self.generic_visit(node)
        self.in_classdef = False
        if not node.body:
            node.body = [ast.Pass()]
        return node

class TagMultiMethods(ast.NodeVisitor):
    """Tag @when-decorated methods in a package AST."""

    def __init__(self, spec):
        if False:
            while True:
                i = 10
        self.spec = spec
        self.methods = {}

    def visit_FunctionDef(self, func):
        if False:
            return 10
        conditions = []
        for dec in func.decorator_list:
            if isinstance(dec, ast.Call) and dec.func.id == 'when':
                try:
                    cond = dec.args[0].s
                    if isinstance(cond, bool):
                        conditions.append(cond)
                        continue
                    try:
                        cond_spec = spack.spec.Spec(cond)
                    except Exception:
                        conditions.append(None)
                    else:
                        conditions.append(self.spec.satisfies(cond_spec))
                except AttributeError:
                    conditions.append(None)
        if not conditions:
            self.methods[func.name] = []
        impl_conditions = self.methods.setdefault(func.name, [])
        impl_conditions.append((func, conditions))
        return func

class ResolveMultiMethods(ast.NodeTransformer):
    """Remove multi-methods when we know statically that they won't be used.

    Say we have multi-methods like this::

        class SomePackage:
            def foo(self): print("implementation 1")

            @when("@1.0")
            def foo(self): print("implementation 2")

            @when("@2.0")
            @when(sys.platform == "darwin")
            def foo(self): print("implementation 3")

            @when("@3.0")
            def foo(self): print("implementation 4")

    The multimethod that will be chosen at runtime depends on the package spec and on
    whether we're on the darwin platform *at build time* (the darwin condition for
    implementation 3 is dynamic). We know the package spec statically; we don't know
    statically what the runtime environment will be. We need to include things that can
    possibly affect package behavior in the package hash, and we want to exclude things
    when we know that they will not affect package behavior.

    If we're at version 4.0, we know that implementation 1 will win, because some @when
    for 2, 3, and 4 will be `False`. We should only include implementation 1.

    If we're at version 1.0, we know that implementation 2 will win, because it
    overrides implementation 1.  We should only include implementation 2.

    If we're at version 3.0, we know that implementation 4 will win, because it
    overrides implementation 1 (the default), and some @when on all others will be
    False.

    If we're at version 2.0, it's a bit more complicated. We know we can remove
    implementations 2 and 4, because their @when's will never be satisfied. But, the
    choice between implementations 1 and 3 will happen at runtime (this is a bad example
    because the spec itself has platform information, and we should prefer to use that,
    but we allow arbitrary boolean expressions in @when's, so this example suffices).
    For this case, we end up needing to include *both* implementation 1 and 3 in the
    package hash, because either could be chosen.

    """

    def __init__(self, methods):
        if False:
            return 10
        self.methods = methods

    def resolve(self, impl_conditions):
        if False:
            i = 10
            return i + 15
        'Given list of nodes and conditions, figure out which node will be chosen.'
        result = []
        default = None
        for (impl, conditions) in impl_conditions:
            if not conditions:
                default = impl
                result.append(default)
                continue
            if any((c is False for c in conditions)):
                continue
            if all((c is True for c in conditions)):
                if result and result[0] is default:
                    return [impl]
            result.append(impl)
        return result

    def visit_FunctionDef(self, func):
        if False:
            print('Hello World!')
        assert func.name in self.methods, 'Inconsistent package traversal!'
        impl_conditions = self.methods[func.name]
        resolutions = self.resolve(impl_conditions)
        if not any((r is func for r in resolutions)):
            return None
        func.decorator_list = [dec for dec in func.decorator_list if not (isinstance(dec, ast.Call) and dec.func.id == 'when')]
        return func

def canonical_source(spec, filter_multimethods=True, source=None):
    if False:
        print('Hello World!')
    "Get canonical source for a spec's package.py by unparsing its AST.\n\n    Arguments:\n        filter_multimethods (bool): By default, filter multimethods out of the\n            AST if they are known statically to be unused. Supply False to disable.\n        source (str): Optionally provide a string to read python code from.\n    "
    return unparse(package_ast(spec, filter_multimethods, source=source), py_ver_consistent=True)

def package_hash(spec, source=None):
    if False:
        while True:
            i = 10
    "Get a hash of a package's canonical source code.\n\n    This function is used to determine whether a spec needs a rebuild when a\n    package's source code changes.\n\n    Arguments:\n        source (str): Optionally provide a string to read python code from.\n\n    "
    source = canonical_source(spec, filter_multimethods=True, source=source)
    return spack.util.hash.b32_hash(source)

def package_ast(spec, filter_multimethods=True, source=None):
    if False:
        while True:
            i = 10
    'Get the AST for the ``package.py`` file corresponding to ``spec``.\n\n    Arguments:\n        filter_multimethods (bool): By default, filter multimethods out of the\n            AST if they are known statically to be unused. Supply False to disable.\n        source (str): Optionally provide a string to read python code from.\n    '
    spec = spack.spec.Spec(spec)
    if source is None:
        filename = spack.repo.PATH.filename_for_package_name(spec.name)
        with open(filename) as f:
            source = f.read()
    root = ast.parse(source)
    root = RemoveDocstrings().visit(root)
    root = RemoveDirectives(spec).visit(root)
    if filter_multimethods:
        tagger = TagMultiMethods(spec)
        tagger.visit(root)
        root = ResolveMultiMethods(tagger.methods).visit(root)
    return root

class PackageHashError(spack.error.SpackError):
    """Raised for all errors encountered during package hashing."""