import libcst as cst
import libcst.matchers as m
from libqtile.scripts.migrations._base import Check, MigrationTransformer, _QtileMigrator, add_migration

class KeychordTransformer(MigrationTransformer):

    @m.leave(m.Call(func=m.Name('KeyChord'), args=[m.ZeroOrMore(), m.Arg(keyword=m.Name('mode')), m.ZeroOrMore()]))
    def update_keychord_args(self, original_node, updated_node) -> cst.Call:
        if False:
            while True:
                i = 10
        "Changes 'mode' kwarg to 'mode' and 'value' kwargs."
        args = original_node.args
        if not args:
            return original_node
        pos = 0
        for (i, arg) in enumerate(args):
            if (kwarg := arg.keyword):
                if kwarg.value == 'mode':
                    if m.matches(arg.value, m.Name('True') | m.Name('False')):
                        return original_node
                    pos = i
                    break
        else:
            return original_node
        self.lint(arg, "The use of mode='mode name' for KeyChord is deprecated. Use mode=True and value='mode name'.")
        name_arg = arg.with_changes(keyword=cst.Name('name'))
        mode_arg = arg.with_changes(value=cst.Name('True'))
        new_args = [a for (i, a) in enumerate(args) if i != pos]
        new_args += [name_arg, mode_arg]
        return updated_node.with_changes(args=new_args)

class KeychordArgs(_QtileMigrator):
    ID = 'UpdateKeychordArgs'
    SUMMARY = 'Updates ``KeyChord`` argument signature.'
    HELP = '\n    Previously, users could make a key chord persist by setting the `mode` to a string representing\n    the name of the mode. For example:\n\n    .. code:: python\n\n        keys = [\n            KeyChord(\n                [mod],\n                "x",\n                [\n                    Key([], "Up", lazy.layout.grow()),\n                    Key([], "Down", lazy.layout.shrink())\n                ],\n                mode="Resize layout",\n            )\n        ]\n\n    This will now result in the following warning message in the log file:\n\n    .. code::\n\n        The use of `mode` to set the KeyChord name is deprecated. Please use `name=\'Resize Layout\'` instead.\n        \'mode\' should be a boolean value to set whether the chord is persistent (True) or not."\n\n    To remove the error, the config should be amended as follows:\n\n    .. code:: python\n\n        keys = [\n            KeyChord(\n                [mod],\n                "x",\n                [\n                    Key([], "Up", lazy.layout.grow()),\n                    Key([], "Down", lazy.layout.shrink())\n                ],\n                name="Resize layout",\n            mode=True,\n            )\n        ]\n\n    .. note::\n\n       The formatting of the inserted argument may not correctly match your own formatting. You may this\n       to run a tool like ``black`` after applying this migration to tidy up your code.\n\n    '
    AFTER_VERSION = '0.21.0'
    TESTS = [Check('\n            from libqtile.config import Key, KeyChord\n            from libqtile.lazy import lazy\n\n            mod = "mod4"\n\n            keys = [\n                KeyChord(\n                    [mod],\n                    "x",\n                    [\n                        Key([], "Up", lazy.layout.grow()),\n                        Key([], "Down", lazy.layout.shrink())\n                    ],\n                    mode="Resize layout",\n                )\n            ]\n            ', '\n            from libqtile.config import Key, KeyChord\n            from libqtile.lazy import lazy\n\n            mod = "mod4"\n\n            keys = [\n                KeyChord(\n                    [mod],\n                    "x",\n                    [\n                        Key([], "Up", lazy.layout.grow()),\n                        Key([], "Down", lazy.layout.shrink())\n                    ],\n                    name="Resize layout",\n                mode=True,\n                )\n            ]\n            ')]
    visitor = KeychordTransformer
add_migration(KeychordArgs)