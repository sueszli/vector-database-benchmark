import libcst as cst
import libcst.matchers as m
from libqtile.scripts.migrations._base import Check, MigrationTransformer, _QtileMigrator, add_migration

class BluetoothArgsTransformer(MigrationTransformer):

    @m.call_if_inside(m.Call(func=m.Name('Bluetooth')) | m.Call(func=m.Attribute(attr=m.Name('Bluetooth'))))
    @m.leave(m.Arg(keyword=m.Name('hci')))
    def update_bluetooth_args(self, original_node, updated_node) -> cst.Arg:
        if False:
            for i in range(10):
                print('nop')
        "Changes positional  argumentto 'widgets' kwargs."
        self.lint(original_node, "The 'hci' argument is deprecated and should be replaced with 'device'.")
        return updated_node.with_changes(keyword=cst.Name('device'))

class BluetoothArgs(_QtileMigrator):
    ID = 'UpdateBluetoothArgs'
    SUMMARY = 'Updates ``Bluetooth`` argument signature.'
    HELP = '\n    The ``Bluetooth`` widget previously accepted a ``hci`` keyword argument. This has\n    been deprecated following a major overhaul of the widget and should be replaced with\n    a keyword argument named ``device``.\n\n    For example:\n\n    .. code:: python\n\n        widget.Bluetooth(hci="/dev_XX_XX_XX_XX_XX_XX")\n\n    should be changed to:\n\n    .. code::\n\n        widget.Bluetooth(device="/dev_XX_XX_XX_XX_XX_XX")\n\n    '
    AFTER_VERSION = '0.23.0'
    TESTS = [Check('\n            from libqtile import bar, widget\n            from libqtile.widget import Bluetooth\n\n            bar.Bar(\n                [\n                    Bluetooth(hci="/dev_XX_XX_XX_XX_XX_XX"),\n                    widget.Bluetooth(hci="/dev_XX_XX_XX_XX_XX_XX"),\n                ],\n                20,\n            )\n            ', '\n            from libqtile import bar, widget\n            from libqtile.widget import Bluetooth\n\n            bar.Bar(\n                [\n                    Bluetooth(device="/dev_XX_XX_XX_XX_XX_XX"),\n                    widget.Bluetooth(device="/dev_XX_XX_XX_XX_XX_XX"),\n                ],\n                20,\n            )\n            ')]
    visitor = BluetoothArgsTransformer
add_migration(BluetoothArgs)