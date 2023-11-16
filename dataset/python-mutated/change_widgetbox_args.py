import libcst as cst
import libcst.matchers as m
from libqtile.scripts.migrations._base import EQUALS_NO_SPACE, Change, Check, MigrationTransformer, NoChange, _QtileMigrator, add_migration

class WidgetboxArgsTransformer(MigrationTransformer):

    @m.call_if_inside(m.Call(func=m.Name('WidgetBox')) | m.Call(func=m.Attribute(attr=m.Name('WidgetBox'))))
    @m.leave(m.Arg(keyword=None))
    def update_widgetbox_args(self, original_node, updated_node) -> cst.Arg:
        if False:
            for i in range(10):
                print('nop')
        "Changes positional  argumentto 'widgets' kwargs."
        self.lint(original_node, "The positional argument should be replaced with a keyword argument named 'widgets'.")
        return updated_node.with_changes(keyword=cst.Name('widgets'), equal=EQUALS_NO_SPACE)

class WidgetboxArgs(_QtileMigrator):
    ID = 'UpdateWidgetboxArgs'
    SUMMARY = 'Updates ``WidgetBox`` argument signature.'
    HELP = '\n    The ``WidgetBox`` widget allowed a position argument to set the contents of the widget.\n    This behaviour is deprecated and, instead, the contents should be specified with a\n    keyword argument called ``widgets``.\n\n    For example:\n\n    .. code:: python\n\n        widget.WidgetBox(\n            [\n                widget.Systray(),\n                widget.Volume(),\n            ]\n        )\n\n    should be changed to:\n\n    .. code::\n\n        widget.WidgetBox(\n            widgets=[\n                widget.Systray(),\n                widget.Volume(),\n            ]\n        )\n\n    '
    AFTER_VERSION = '0.20.0'
    TESTS = [Change('\n            widget.WidgetBox(\n                [\n                    widget.Systray(),\n                    widget.Volume(),\n                ]\n            )\n            ', '\n            widget.WidgetBox(\n                widgets=[\n                    widget.Systray(),\n                    widget.Volume(),\n                ]\n            )\n            '), Change('\n            WidgetBox(\n                [\n                    widget.Systray(),\n                    widget.Volume(),\n                ]\n            )\n            ', '\n            WidgetBox(\n                widgets=[\n                    widget.Systray(),\n                    widget.Volume(),\n                ]\n            )\n            '), NoChange('\n            widget.WidgetBox(\n                widgets=[\n                    widget.Systray(),\n                    widget.Volume(),\n                ]\n            )\n            '), Check('\n            from libqtile import bar, widget\n            from libqtile.widget import WidgetBox\n\n            bar.Bar(\n                [\n                    WidgetBox(\n                        [\n                            widget.Systray(),\n                            widget.Volume(),\n                        ]\n                    ),\n                    widget.WidgetBox(\n                        [\n                            widget.Systray(),\n                            widget.Volume(),\n                        ]\n                    )\n                ],\n                20,\n            )\n            ', '\n            from libqtile import bar, widget\n            from libqtile.widget import WidgetBox\n\n            bar.Bar(\n                [\n                    WidgetBox(\n                        widgets=[\n                            widget.Systray(),\n                            widget.Volume(),\n                        ]\n                    ),\n                    widget.WidgetBox(\n                        widgets=[\n                            widget.Systray(),\n                            widget.Volume(),\n                        ]\n                    )\n                ],\n                20,\n            )\n            ')]
    visitor = WidgetboxArgsTransformer
add_migration(WidgetboxArgs)