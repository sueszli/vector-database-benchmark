import libcst as cst
import libcst.matchers as m
from libcst import codemod
from libcst.codemod.visitors import AddImportsVisitor, RemoveImportsVisitor
from libqtile.scripts.migrations._base import Change, Check, MigrationTransformer, _QtileMigrator, add_migration
MODULE_MAP = {'command_graph': cst.parse_expression('libqtile.command.graph'), 'command_client': cst.parse_expression('libqtile.command.client'), 'command_interface': cst.parse_expression('libqtile.command.interface'), 'command_object': cst.parse_expression('libqtile.command.base'), 'window': cst.parse_expression('libqtile.backend.x11.window')}
IMPORT_MAP = {'command_graph': ('libqtile.command', 'graph'), 'command_client': ('libqtile.command', 'client'), 'command_interface': ('libqtile.command', 'interface'), 'command_object': ('libqtile.command', 'base'), 'window': ('libqtile.backend.x11', 'window')}

class ModuleRenamesTransformer(MigrationTransformer):
    """
    This transfore does a number of things:
    - Where possible, it replaces module names directly. It is able to do so where
      the module is shown in its dotted form e.g. libqtile.command_graph.
    - In addition, it identifies where modules are imported from libqtile and
      stores these values in a list where they can be accessed later.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.from_imports = []
        MigrationTransformer.__init__(self, *args, **kwargs)

    def do_lint(self, original_node, module):
        if False:
            for i in range(10):
                print('nop')
        if module == 'window':
            self.lint(original_node, "The 'libqtile.window' has been moved to 'libqtile.backend.x11.window'.")
        else:
            self.lint(original_node, "The 'libqtile.command_*' modules have been moved to 'libqtile.command.*'.")

    @m.leave(m.ImportAlias(name=m.Attribute(value=m.Name('libqtile'), attr=m.Name(m.MatchIfTrue(lambda x: x in MODULE_MAP)))))
    def update_import_module_names(self, original_node, updated_node) -> cst.ImportAlias:
        if False:
            for i in range(10):
                print('nop')
        "Renames modules in 'import ...' statements."
        module = original_node.name.attr.value
        self.do_lint(original_node, module)
        new_module = MODULE_MAP[module]
        return updated_node.with_changes(name=new_module)

    @m.leave(m.ImportFrom(module=m.Attribute(value=m.Name('libqtile'), attr=m.Name(m.MatchIfTrue(lambda x: x in MODULE_MAP)))))
    def update_import_from_module_names(self, original_node, updated_node) -> cst.ImportFrom:
        if False:
            print('Hello World!')
        "Renames modules in 'from ... import ...' statements."
        module = original_node.module.attr.value
        self.do_lint(original_node, module)
        new_module = MODULE_MAP[module]
        return updated_node.with_changes(module=new_module)

    @m.leave(m.ImportFrom(module=m.Name('libqtile'), names=[m.ZeroOrMore(), m.ImportAlias(name=m.Name(m.MatchIfTrue(lambda x: x in IMPORT_MAP))), m.ZeroOrMore()]))
    def tag_from_imports(self, original_node, _) -> cst.ImportFrom:
        if False:
            for i in range(10):
                print('nop')
        'Marks which modules are'
        for name in original_node.names:
            if name.name.value in IMPORT_MAP:
                self.lint(original_node, f'From libqtile import {name.name.value} is deprecated.')
                self.from_imports.append(name.name.value)
        return original_node

class ModuleRenames(_QtileMigrator):
    ID = 'ModuleRenames'
    SUMMARY = 'Updates certain deprecated ``libqtile.`` module names.'
    HELP = '\n    To tidy up the qtile codebase, the ``libqtile.command_*`` modules were moved to\n    ``libqtile.command.*`` with one exception, ``libqtile.command_object`` was renamed\n    ``libqtile.command.base``.\n\n    In addition, the ``libqtile.window`` module was moved to ``libqtile.backend.x11.window``.\n\n    NB. this migration will update imports in the following forms:\n\n    .. code:: python\n\n        import libqtile.command_client\n        import libqtile.command_interface as interface\n        from libqtile import command_client\n        from libqtile.command_client import CommandClient\n\n    Results in:\n\n    .. code:: python\n\n        import libqtile.command.client\n        import libqtile.command.interface as interface\n        from libqtile.command.client import CommandClient\n        from libqtile.command import client\n\n    '
    AFTER_VERSION = '0.18.1'
    TESTS = []
    for (mod, (new_mod, new_file)) in IMPORT_MAP.items():
        TESTS.append(Change(f'import libqtile.{mod}', f'import {new_mod}.{new_file}'))
        TESTS.append(Change(f'from libqtile.{mod} import foo', f'from {new_mod}.{new_file} import foo'))
        TESTS.append(Change(f'from libqtile import {mod}', f'from {new_mod} import {new_file}'))
    TESTS.append(Check('\n            from libqtile.command_client import CommandClient\n            from libqtile.command_graph import CommandGraphNode\n            from libqtile.command_interface import QtileCommandInterface\n            from libqtile.command_object import CommandObject\n            ', '\n            from libqtile.command.client import CommandClient\n            from libqtile.command.graph import CommandGraphNode\n            from libqtile.command.interface import QtileCommandInterface\n            from libqtile.command.base import CommandObject\n            '))

    def run(self, original_node):
        if False:
            while True:
                i = 10
        transformer = ModuleRenamesTransformer()
        updated = original_node.visit(transformer)
        self.update_lint(transformer)
        if transformer.from_imports:
            context = codemod.CodemodContext()
            for name in transformer.from_imports:
                (base, module) = IMPORT_MAP[name]
                RemoveImportsVisitor.remove_unused_import(context, 'libqtile', name)
                AddImportsVisitor.add_needed_import(context, base, module)
            remove_visitor = RemoveImportsVisitor(context)
            add_visitor = AddImportsVisitor(context)
            updated = remove_visitor.transform_module(updated)
            updated = add_visitor.transform_module(updated)
        return (original_node, updated)
add_migration(ModuleRenames)