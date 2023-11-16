"""Converts pyi files to pickled asts and saves them to disk.

Used to speed up module importing. This is done by loading the ast and
serializing it to disk. Further users only need to read the serialized data from
disk, which is faster to digest than a pyi file.
"""
from typing import List, NamedTuple, Optional, Set, Tuple
from pytype import utils
from pytype.pyi import parser
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import visitors

class UnrestorableDependencyError(Exception):
    """If a dependency can't be restored in the current state."""

class FindClassTypesVisitor(visitors.Visitor):
    """Visitor to find class and function types."""

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.class_type_nodes = []

    def EnterClassType(self, n):
        if False:
            return 10
        self.class_type_nodes.append(n)

class UndoModuleAliasesVisitor(visitors.Visitor):
    """Visitor to undo module aliases in late types.

  Since late types are loaded out of context, they need to contain the original
  names of modules, not whatever they've been aliased to in the current module.
  """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._module_aliases = {}

    def EnterTypeDeclUnit(self, node):
        if False:
            return 10
        for alias in node.aliases:
            if isinstance(alias.type, pytd.Module):
                name = utils.strip_prefix(alias.name, f'{node.name}.')
                self._module_aliases[name] = alias.type.module_name

    def VisitLateType(self, node):
        if False:
            print('Hello World!')
        if '.' not in node.name:
            return node
        (prefix, suffix) = node.name.rsplit('.', 1)
        while prefix:
            if prefix in self._module_aliases:
                return node.Replace(name=self._module_aliases[prefix] + '.' + suffix)
            (prefix, _, remainder) = prefix.rpartition('.')
            suffix = f'{remainder}.{suffix}'
        return node

class SerializableTupleClass(NamedTuple):
    ast: pytd.TypeDeclUnit
    dependencies: List[Tuple[str, Set[str]]]
    late_dependencies: List[Tuple[str, Set[str]]]
    class_type_nodes: Optional[List[pytd.ClassType]]
    src_path: Optional[str]
    metadata: List[str]

class SerializableAst(SerializableTupleClass):
    """The data pickled to disk to save an ast.

  Attributes:
    ast: The TypeDeclUnit representing the serialized module.
    dependencies: A list of modules this AST depends on. The modules are
      represented as Fully Qualified names. E.g. foo.bar.module. This set will
      also contain the module being imported, if the module is not empty.
      Therefore it might be different from the set found by
      visitors.CollectDependencies in
      load_pytd._load_and_resolve_ast_dependencies.
    late_dependencies: This AST's late dependencies.
    class_type_nodes: A list of all the ClassType instances in ast or None. If
      this list is provided only the ClassType instances in the list will be
      visited and have their .cls set. If this attribute is None the whole AST
      will be visited and all found ClassType instances will have their .cls
      set.
    src_path: Optionally, the filepath of the original source file.
    metadata: A list of arbitrary string-encoded metadata.
  """
    Replace = SerializableTupleClass._replace

def SerializeAst(ast, src_path=None, metadata=None):
    if False:
        print('Hello World!')
    'Loads and stores an ast to disk.\n\n  Args:\n    ast: The pytd.TypeDeclUnit to save to disk.\n    src_path: Optionally, the filepath of the original source file.\n    metadata: A list of arbitrary string-encoded metadata.\n\n  Returns:\n    The SerializableAst derived from `ast`.\n  '
    if ast.name.endswith('.__init__'):
        ast = ast.Visit(visitors.RenameModuleVisitor(ast.name, ast.name.rsplit('.__init__', 1)[0]))
    ast = ast.Visit(UndoModuleAliasesVisitor())
    deps = visitors.CollectDependencies()
    ast.Visit(deps)
    dependencies = deps.dependencies
    late_dependencies = deps.late_dependencies
    ast.Visit(visitors.ClearClassPointers())
    indexer = FindClassTypesVisitor()
    ast.Visit(indexer)
    ast = ast.Visit(visitors.CanonicalOrderingVisitor())
    metadata = metadata or []
    return SerializableAst(ast, sorted(dependencies.items()), sorted(late_dependencies.items()), sorted(indexer.class_type_nodes), src_path=src_path, metadata=metadata)

def EnsureAstName(ast, module_name, fix=False):
    if False:
        while True:
            i = 10
    'Verify that serializable_ast has the name module_name, or repair it.\n\n  Args:\n    ast: An instance of SerializableAst.\n    module_name: The name under which ast.ast should be loaded.\n    fix: If this function should repair the wrong name.\n\n  Returns:\n    The updated SerializableAst.\n  '
    raw_ast = ast.ast
    if fix and module_name != raw_ast.name:
        ast = ast.Replace(class_type_nodes=None)
        ast = ast.Replace(ast=raw_ast.Visit(visitors.RenameModuleVisitor(raw_ast.name, module_name)))
    else:
        assert module_name == raw_ast.name
    return ast

def ProcessAst(serializable_ast, module_map):
    if False:
        while True:
            i = 10
    'Postprocess a pickled ast.\n\n  Postprocessing will either just fill the ClassType references from module_map\n  or if module_name changed between pickling and loading rename the module\n  internal references to the new module_name.\n  Renaming is more expensive than filling references, as the whole AST needs to\n  be rebuild.\n\n  Args:\n    serializable_ast: A SerializableAst instance.\n    module_map: Used to resolve ClassType.cls links to already loaded modules.\n      The loaded module will be added to the dict.\n\n  Returns:\n    A pytd.TypeDeclUnit, this is either the input raw_ast with the references\n    set or a newly created AST with the new module_name and the references set.\n\n  Raises:\n    AssertionError: If module_name is already in module_map, which means that\n      module_name is already loaded.\n    UnrestorableDependencyError: If no concrete module exists in module_map for\n      one of the references from the pickled ast.\n  '
    serializable_ast = _LookupClassReferences(serializable_ast, module_map, serializable_ast.ast.name)
    serializable_ast = FillLocalReferences(serializable_ast, {'': serializable_ast.ast, serializable_ast.ast.name: serializable_ast.ast})
    return serializable_ast.ast

def _LookupClassReferences(serializable_ast, module_map, self_name):
    if False:
        return 10
    'Fills .cls references in serializable_ast.ast with ones from module_map.\n\n  Already filled references are not changed. References to the module self._name\n  are not filled. Setting self_name=None will fill all references.\n\n  Args:\n    serializable_ast: A SerializableAst instance.\n    module_map: Used to resolve ClassType.cls links to already loaded modules.\n      The loaded module will be added to the dict.\n    self_name: A string representation of a module which should not be resolved,\n      for example: "foo.bar.module1" or None to resolve all modules.\n\n  Returns:\n    A SerializableAst with an updated .ast. .class_type_nodes is set to None\n    if any of the Nodes needed to be regenerated.\n  '
    class_lookup = visitors.LookupExternalTypes(module_map, self_name=self_name)
    raw_ast = serializable_ast.ast
    decorators = {d.type.name for c in raw_ast.classes + raw_ast.functions for d in c.decorators}
    for node in serializable_ast.class_type_nodes or ():
        try:
            class_lookup.allow_functions = node.name in decorators
            if node is not class_lookup.VisitClassType(node):
                serializable_ast = serializable_ast.Replace(class_type_nodes=None)
                break
        except KeyError as e:
            raise UnrestorableDependencyError(f'Unresolved class: {str(e)!r}.') from e
    if serializable_ast.class_type_nodes is None:
        try:
            raw_ast = raw_ast.Visit(class_lookup)
        except KeyError as e:
            raise UnrestorableDependencyError(f'Unresolved class: {str(e)!r}.') from e
    serializable_ast = serializable_ast.Replace(ast=raw_ast)
    return serializable_ast

def FillLocalReferences(serializable_ast, module_map):
    if False:
        i = 10
        return i + 15
    'Fill in local references.'
    local_filler = visitors.FillInLocalPointers(module_map)
    if serializable_ast.class_type_nodes is None:
        serializable_ast.ast.Visit(local_filler)
        return serializable_ast.Replace(class_type_nodes=None)
    else:
        for node in serializable_ast.class_type_nodes:
            local_filler.EnterClassType(node)
            if node.cls is None:
                raise AssertionError(f'This should not happen: {str(node)}')
        return serializable_ast

def PrepareForExport(module_name, ast, loader):
    if False:
        return 10
    'Prepare an ast as if it was parsed and loaded.\n\n  External dependencies will not be resolved, as the ast generated by this\n  method is supposed to be exported.\n\n  Args:\n    module_name: The module_name as a string for the returned ast.\n    ast: pytd.TypeDeclUnit, is only used if src is None.\n    loader: A load_pytd.Loader instance.\n\n  Returns:\n    A pytd.TypeDeclUnit representing the supplied AST as it would look after\n    being written to a file and parsed.\n  '
    src = pytd_utils.Print(ast)
    return SourceToExportableAst(module_name, src, loader)

def SourceToExportableAst(module_name, src, loader):
    if False:
        for i in range(10):
            print('nop')
    'Parse the source code into a pickle-able ast.'
    ast = parser.parse_string(src=src, name=module_name, filename=loader.options.input, options=parser.PyiOptions.from_toplevel_options(loader.options))
    ast = ast.Visit(visitors.LookupBuiltins(loader.builtins, full_names=False))
    ast = ast.Visit(visitors.LookupLocalTypes())
    ast = ast.Visit(visitors.AdjustTypeParameters())
    ast = ast.Visit(visitors.NamedTypeToClassType())
    ast = ast.Visit(visitors.FillInLocalPointers({'': ast, module_name: ast}))
    ast = ast.Visit(visitors.ClassTypeToLateType(ignore=[module_name + '.', 'builtins.', 'typing.']))
    return ast