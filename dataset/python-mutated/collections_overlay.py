"""Implementation of types from Python 2's collections library."""
from pytype.overlays import named_tuple
from pytype.overlays import overlay
from pytype.overlays import typing_overlay

class CollectionsOverlay(overlay.Overlay):
    """A custom overlay for the 'collections' module."""

    def __init__(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        "Initializes the CollectionsOverlay.\n\n    This function loads the AST for the collections module, which is used to\n    access type information for any members that are not explicitly provided by\n    the overlay. See get_attribute in attribute.py for how it's used.\n\n    Args:\n      ctx: An instance of context.Context.\n    "
        member_map = collections_overlay.copy()
        ast = ctx.loader.import_name('collections')
        super().__init__(ctx, 'collections', member_map, ast)
collections_overlay = {'namedtuple': named_tuple.CollectionsNamedTupleBuilder.make}

class ABCOverlay(typing_overlay.Redirect):
    """A custom overlay for the 'collections.abc' module."""

    def __init__(self, ctx):
        if False:
            print('Hello World!')
        super().__init__('collections.abc', {'Set': 'typing.AbstractSet'}, ctx)