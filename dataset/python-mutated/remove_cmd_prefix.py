import libcst as cst
import libcst.matchers as m
from libcst import codemod
from libcst.codemod.visitors import AddImportsVisitor
from libqtile.scripts.migrations._base import Change, Check, MigrationTransformer, NoChange, _QtileMigrator, add_migration
MIGRATION_MAP = {'cmd_hints': 'get_hints', 'cmd_groups': 'get_groups', 'cmd_screens': 'get_screens', 'cmd_opacity': 'set_opacity'}

def is_cmd(value):
    if False:
        while True:
            i = 10
    return value.startswith('cmd_')

class CmdPrefixTransformer(MigrationTransformer):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        MigrationTransformer.__init__(self, *args, **kwargs)
        self.needs_import = False

    @m.call_if_inside(m.Call())
    @m.leave(m.Name(m.MatchIfTrue(is_cmd)))
    def change_func_call_name(self, original_node, updated_node) -> cst.Name:
        if False:
            while True:
                i = 10
        'Removes cmd_prefix from a call where the command begins with the cmd_ prefix.'
        name = original_node.value
        if name in MIGRATION_MAP:
            replacement = MIGRATION_MAP[name]
        else:
            replacement = name[4:]
        self.lint(original_node, f"Use of 'cmd_' prefix is deprecated. '{name}' should be replaced with '{replacement}'")
        return updated_node.with_changes(value=replacement)

    @m.call_if_inside(m.ClassDef())
    @m.leave(m.FunctionDef(name=m.Name(m.MatchIfTrue(is_cmd))))
    def change_func_def(self, original_node, updated_node) -> cst.FunctionDef:
        if False:
            i = 10
            return i + 15
        "\n        Renames method definitions using the cmd_ prefix and adds the\n        @expose_command decorator.\n\n        Also sets a flag to show that the script should add the relevant\n        import if it's missing.\n        "
        decorator = cst.Decorator(cst.Name('expose_command'))
        name = original_node.name
        updated = name.with_changes(value=name.value[4:])
        self.lint(original_node, "Use of 'cmd_' prefix is deprecated. Use '@expose_command' when defining methods.")
        self.needs_import = True
        return updated_node.with_changes(name=updated, decorators=[decorator])

class RemoveCmdPrefix(_QtileMigrator):
    ID = 'RemoveCmdPrefix'
    SUMMARY = 'Removes ``cmd_`` prefix from method calls and definitions.'
    HELP = '\n    The ``cmd_`` prefix was used to identify methods that should be exposed to\n    qtile\'s command API. This has been deprecated and so calls no longer require\n    the prefix.\n\n    For example:\n\n    .. code:: python\n\n      qtile.cmd_spawn("vlc")\n\n    would be replaced with:\n\n    .. code:: python\n\n      qtile.spawn("vlc")\n\n    Where users have created their own widgets with methods using this prefix,\n    the syntax has also changed:\n\n    For example:\n\n    .. code:: python\n\n        class MyWidget(libqtile.widget.base._Widget):\n            def cmd_my_command(self):\n                pass\n\n    Should be updated as follows:\n\n    .. code:: python\n\n        from libqtile.command.base import expose_command\n\n        class MyWidget(libqtile.widget.base._Widget):\n            @expose_command\n            def my_command(self):\n                pass\n    '
    AFTER_VERSION = '0.22.1'
    TESTS = [Change('qtile.cmd_spawn("alacritty")', 'qtile.spawn("alacritty")'), Change('qtile.cmd_groups()', 'qtile.get_groups()'), Change('qtile.cmd_screens()', 'qtile.get_screens()'), Change('qtile.current_window.cmd_hints()', 'qtile.current_window.get_hints()'), Change('qtile.current_window.cmd_opacity(0.5)', 'qtile.current_window.set_opacity(0.5)'), Change('\n            class MyWidget(widget.Clock):\n                def cmd_my_command(self):\n                    pass\n            ', '\n            from libqtile.command.base import expose_command\n\n            class MyWidget(widget.Clock):\n                @expose_command\n                def my_command(self):\n                    pass\n            '), NoChange('\n            def cmd_some_other_func():\n                pass\n            '), Check('\n            from libqtile import qtile, widget\n\n            class MyClock(widget.Clock):\n                def cmd_my_exposed_command(self):\n                    pass\n\n            def my_func(qtile):\n                qtile.cmd_spawn("rickroll")\n                hints = qtile.current_window.cmd_hints()\n                groups = qtile.cmd_groups()\n                screens = qtile.cmd_screens()\n                qtile.current_window.cmd_opacity(0.5)\n\n            def cmd_some_other_func():\n                pass\n            ', '\n            from libqtile import qtile, widget\n            from libqtile.command.base import expose_command\n\n            class MyClock(widget.Clock):\n                @expose_command\n                def my_exposed_command(self):\n                    pass\n\n            def my_func(qtile):\n                qtile.spawn("rickroll")\n                hints = qtile.current_window.get_hints()\n                groups = qtile.get_groups()\n                screens = qtile.get_screens()\n                qtile.current_window.set_opacity(0.5)\n\n            def cmd_some_other_func():\n                pass\n            ')]

    def run(self, original):
        if False:
            while True:
                i = 10
        transformer = CmdPrefixTransformer()
        updated = original.visit(transformer)
        self.update_lint(transformer)
        if transformer.needs_import:
            context = codemod.CodemodContext()
            AddImportsVisitor.add_needed_import(context, 'libqtile.command.base', 'expose_command')
            visitor = AddImportsVisitor(context)
            updated = updated.visit(visitor)
        return (original, updated)
add_migration(RemoveCmdPrefix)