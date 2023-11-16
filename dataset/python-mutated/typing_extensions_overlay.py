"""Implementation of special members of typing_extensions."""
from pytype.overlays import typing_overlay

class TypingExtensionsOverlay(typing_overlay.Redirect):
    """A custom overlay for the 'typing_extensions' module."""

    def __init__(self, ctx):
        if False:
            return 10
        aliases = {'runtime': 'typing.runtime_checkable'}
        super().__init__('typing_extensions', aliases, ctx)

    def _convert_member(self, name, member, subst=None):
        if False:
            while True:
                i = 10
        var = super()._convert_member(name, member, subst)
        for val in var.data:
            val.module = 'typing'
        return var