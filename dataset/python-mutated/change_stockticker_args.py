import libcst as cst
import libcst.matchers as m
from libqtile.scripts.migrations._base import Change, Check, MigrationTransformer, NoChange, _QtileMigrator, add_migration

class StocktickerArgsTransformer(MigrationTransformer):

    @m.call_if_inside(m.Call(func=m.Name('StockTicker')) | m.Call(func=m.Attribute(attr=m.Name('StockTicker'))))
    @m.leave(m.Arg(keyword=m.Name('function')))
    def update_stockticker_args(self, original_node, updated_node) -> cst.Arg:
        if False:
            return 10
        "Changes 'function' kwarg to 'mode' and 'func' kwargs."
        self.lint(original_node, "The 'function' keyword argument should be renamed 'func'.")
        return updated_node.with_changes(keyword=cst.Name('func'))

class StocktickerArgs(_QtileMigrator):
    ID = 'UpdateStocktickerArgs'
    SUMMARY = 'Updates ``StockTicker`` argument signature.'
    HELP = '\n    The ``StockTicker`` widget had a keyword argument called ``function``. This needs to be\n    renamed to ``func`` to prevent clashes with the ``function()`` method of ``CommandObject``.\n\n    For example:\n\n    .. code:: python\n\n        widget.StockTicker(function="TIME_SERIES_INTRADAY")\n\n    should be changed to:\n\n    .. code::\n\n        widget.StockTicker(func="TIME_SERIES_INTRADAY")\n\n    '
    AFTER_VERSION = '0.22.1'
    TESTS = [Change('StockTicker(function="TIME_SERIES_INTRADAY")', 'StockTicker(func="TIME_SERIES_INTRADAY")'), Change('widget.StockTicker(function="TIME_SERIES_INTRADAY")', 'widget.StockTicker(func="TIME_SERIES_INTRADAY")'), Change('libqtile.widget.StockTicker(function="TIME_SERIES_INTRADAY")', 'libqtile.widget.StockTicker(func="TIME_SERIES_INTRADAY")'), NoChange('StockTicker(func="TIME_SERIES_INTRADAY")'), NoChange('widget.StockTicker(func="TIME_SERIES_INTRADAY")'), NoChange('libqtile.widget.StockTicker(func="TIME_SERIES_INTRADAY")'), Check('\n            import libqtile\n            from libqtile import bar, widget\n            from libqtile.widget import StockTicker\n\n            bar.Bar(\n                [\n                    StockTicker(function="TIME_SERIES_INTRADAY"),\n                    widget.StockTicker(function="TIME_SERIES_INTRADAY"),\n                    libqtile.widget.StockTicker(function="TIME_SERIES_INTRADAY")\n                ],\n                20\n            )\n            ', '\n            import libqtile\n            from libqtile import bar, widget\n            from libqtile.widget import StockTicker\n\n            bar.Bar(\n                [\n                    StockTicker(func="TIME_SERIES_INTRADAY"),\n                    widget.StockTicker(func="TIME_SERIES_INTRADAY"),\n                    libqtile.widget.StockTicker(func="TIME_SERIES_INTRADAY")\n                ],\n                20\n            )\n            ')]
    visitor = StocktickerArgsTransformer
add_migration(StocktickerArgs)