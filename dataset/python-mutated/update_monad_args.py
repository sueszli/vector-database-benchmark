import libcst as cst
import libcst.matchers as m
from libqtile.scripts.migrations._base import Check, MigrationTransformer, _QtileMigrator, add_migration

class UpdateMonadLayoutTransformer(MigrationTransformer):

    @m.call_if_inside(m.Call(func=m.Name(m.MatchIfTrue(lambda n: n.startswith('Monad')))) | m.Call(func=m.Attribute(attr=m.Name(m.MatchIfTrue(lambda n: n.startswith('Monad'))))))
    @m.leave(m.Arg(keyword=m.Name('new_at_current')))
    def update_monad_args(self, original_node, updated_node) -> cst.Arg:
        if False:
            return 10
        "\n        Changes 'new_at_current' kwarg to 'new_client_position' and sets correct\n        value ('before|after_current').\n        "
        self.lint(original_node, "The 'new_at_current' keyword argument in 'Monad' layouts is invalid.")
        new_value = cst.SimpleString('"before_current"' if original_node.value.value == 'True' else '"after_current"')
        return updated_node.with_changes(keyword=cst.Name('new_client_position'), value=new_value)

class UpdateMonadArgs(_QtileMigrator):
    ID = 'UpdateMonadArgs'
    SUMMARY = 'Updates ``new_at_current`` keyword argument in Monad layouts.'
    HELP = '\n    Replaces the ``new_at_current=True|False`` argument in ``Monad*`` layouts with\n    ``new_client_position`` to be consistent with other layouts.\n\n    ``new_at_current=True`` is replaced with ``new_client_position="before_current`` and\n    ``new_at_current=False`` is replaced with ``new_client_position="after_current"``.\n\n    '
    AFTER_VERSION = '0.17.0'
    TESTS = [Check('\n            from libqtile import layout\n\n            layouts = [\n                layout.MonadTall(border_focus="#ff0000", new_at_current=False),\n                layout.MonadWide(new_at_current=True, border_focus="#ff0000"),\n            ]\n            ', '\n            from libqtile import layout\n\n            layouts = [\n                layout.MonadTall(border_focus="#ff0000", new_client_position="after_current"),\n                layout.MonadWide(new_client_position="before_current", border_focus="#ff0000"),\n            ]\n            ')]
    visitor = UpdateMonadLayoutTransformer
add_migration(UpdateMonadArgs)