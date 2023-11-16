import libcst as cst
import libcst.matchers as m
from libqtile.scripts.migrations._base import Check, MigrationTransformer, _QtileMigrator, add_migration

class UpdateTogroupTransformer(MigrationTransformer):

    @m.call_if_inside(m.Call(func=m.Name(m.MatchIfTrue(lambda n: 'togroup' in n))) | m.Call(func=m.Attribute(attr=m.Name(m.MatchIfTrue(lambda n: 'togroup' in n)))))
    @m.leave(m.Arg(keyword=m.Name('groupName')))
    def update_togroup_args(self, original_node, updated_node) -> cst.Arg:
        if False:
            while True:
                i = 10
        "Changes 'groupName' kwarg to 'group_name'."
        self.lint(original_node, "The 'groupName' keyword argument should be replaced with 'group_name.")
        return updated_node.with_changes(keyword=cst.Name('group_name'))

class UpdateTogroupArgs(_QtileMigrator):
    ID = 'UpdateTogroupArgs'
    SUMMARY = 'Updates ``groupName`` keyword argument to ``group_name``.'
    HELP = '\n    To be consistent with codestyle, the ``groupName`` argument in the ``togroup`` command needs to be\n    changed to ``group_name``.\n\n\n    The following code:\n\n    .. code:: python\n\n        lazy.window.togroup(groupName="1")\n\n    will result in a warning in your logfile: ``Window.togroup\'s groupName is deprecated; use group_name``.\n\n    The code should be updated to:\n\n    .. code:: python\n\n        lazy.window.togroup(group_name="1")\n\n    '
    AFTER_VERSION = '0.18.1'
    TESTS = [Check('\n            from libqtile.config import Key\n            from libqtile.lazy import lazy\n\n            k = Key([], \'s\', lazy.window.togroup(groupName="g"))\n            c = lambda win: win.togroup(groupName="g")\n            ', '\n            from libqtile.config import Key\n            from libqtile.lazy import lazy\n\n            k = Key([], \'s\', lazy.window.togroup(group_name="g"))\n            c = lambda win: win.togroup(group_name="g")\n            ')]
    visitor = UpdateTogroupTransformer
add_migration(UpdateTogroupArgs)