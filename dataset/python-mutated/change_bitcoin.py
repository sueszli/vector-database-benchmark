import libcst as cst
import libcst.matchers as m
from libqtile.scripts.migrations._base import Check, RenamerTransformer, _QtileMigrator, add_migration

def is_bitcoin(func):
    if False:
        while True:
            i = 10
    name = func.value
    if hasattr(func, 'attr'):
        attr = func.attr.value
    else:
        attr = None
    return name == 'BitcoinTicker' or attr == 'BitcoinTicker'

class BitcoinTransformer(RenamerTransformer):
    from_to = ('BitcoinTicker', 'CryptoTicker')

    @m.leave(m.Call(func=m.MatchIfTrue(is_bitcoin), args=[m.ZeroOrMore(), m.Arg(keyword=m.Name('format')), m.ZeroOrMore()]))
    def remove_format_kwarg(self, original_node, updated_node) -> cst.Call:
        if False:
            print('Hello World!')
        "Removes the 'format' keyword argument from 'BitcoinTracker'."
        new_args = [a for a in original_node.args if a.keyword.value != 'format']
        new_args[-1] = new_args[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        return updated_node.with_changes(args=new_args)

class BitcoinToCrypto(_QtileMigrator):
    ID = 'UpdateBitcoin'
    SUMMARY = 'Updates ``BitcoinTicker`` to ``CryptoTicker``.'
    HELP = '\n    The ``BitcoinTicker`` widget has been renamed ``CryptoTicker``. In addition, the ``format``\n    keyword argument is removed during this migration as the available fields for the format\n    have changed.\n\n    The removal only happens on instances of ``BitcoinTracker``. i.e. running ``qtile migrate``\n    on the following code:\n\n    .. code:: python\n\n        BitcoinTicker(format="...")\n        CryptoTicker(format="...")\n\n    will return:\n\n    .. code:: python\n\n        CryptoTicker()\n        CryptoTicker(format="...")\n\n    '
    AFTER_VERSION = '0.18.0'
    TESTS = [Check("\n            from libqtile import bar\n            from libqtile.widget import BitcoinTicker\n\n            bar.Bar(\n                [\n                    BitcoinTicker(crypto='BTC', format='BTC: {avg}'),\n                    BitcoinTicker(format='{crypto}: {avg}', font='sans'),\n                    BitcoinTicker(),\n                    BitcoinTicker(currency='EUR', format='{avg}', foreground='ffffff'),\n                ],\n                30\n            )\n            ", "\n            from libqtile import bar\n            from libqtile.widget import CryptoTicker\n\n            bar.Bar(\n                [\n                    CryptoTicker(crypto='BTC'),\n                    CryptoTicker(font='sans'),\n                    CryptoTicker(),\n                    CryptoTicker(currency='EUR', foreground='ffffff'),\n                ],\n                30\n            )\n            ")]
    visitor = BitcoinTransformer
add_migration(BitcoinToCrypto)