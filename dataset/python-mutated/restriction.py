from pyrogram import raw
from ..object import Object

class Restriction(Object):
    """A restriction applied to bots or chats.

    Parameters:
        platform (``str``):
            The platform the restriction is applied to, e.g. "ios", "android"

        reason (``str``):
            The restriction reason, e.g. "porn", "copyright".

        text (``str``):
            The restriction text.
    """

    def __init__(self, *, platform: str, reason: str, text: str):
        if False:
            i = 10
            return i + 15
        super().__init__(None)
        self.platform = platform
        self.reason = reason
        self.text = text

    @staticmethod
    def _parse(restriction: 'raw.types.RestrictionReason') -> 'Restriction':
        if False:
            while True:
                i = 10
        return Restriction(platform=restriction.platform, reason=restriction.reason, text=restriction.text)