from typing import List
from pyrogram import raw
from pyrogram import types
from ..object import Object

class TermsOfService(Object):
    """Telegram's Terms of Service returned by :meth:`~pyrogram.Client.sign_in`.

    Parameters:
        id (``str``):
            Terms of Service identifier.

        text (``str``):
            Terms of Service text.

        entities (List of :obj:`~pyrogram.types.MessageEntity`):
            Special entities like URLs that appear in the text.
    """

    def __init__(self, *, id: str, text: str, entities: List['types.MessageEntity']):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.id = id
        self.text = text
        self.entities = entities

    @staticmethod
    def _parse(terms_of_service: 'raw.types.help.TermsOfService') -> 'TermsOfService':
        if False:
            i = 10
            return i + 15
        return TermsOfService(id=terms_of_service.id.data, text=terms_of_service.text, entities=[types.MessageEntity._parse(None, entity, {}) for entity in terms_of_service.entities] if terms_of_service.entities else None)