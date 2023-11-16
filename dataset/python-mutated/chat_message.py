import html
from typing import List, Optional, Union
from ..element import Element

class ChatMessage(Element, component='chat_message.js'):

    def __init__(self, text: Union[str, List[str]], *, name: Optional[str]=None, label: Optional[str]=None, stamp: Optional[str]=None, avatar: Optional[str]=None, sent: bool=False, text_html: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Chat Message\n\n        Based on Quasar's `Chat Message <https://quasar.dev/vue-components/chat/>`_ component.\n\n        :param text: the message body (can be a list of strings for multiple message parts)\n        :param name: the name of the message author\n        :param label: renders a label header/section only\n        :param stamp: timestamp of the message\n        :param avatar: URL to an avatar\n        :param sent: render as a sent message (so from current user) (default: False)\n        :param text_html: render text as HTML (default: False)\n        "
        super().__init__()
        if isinstance(text, str):
            text = [text]
        if not text_html:
            text = [html.escape(part) for part in text]
            text = [part.replace('\n', '<br />') for part in text]
        self._props['text'] = text
        self._props['text-html'] = True
        if name is not None:
            self._props['name'] = name
        if label is not None:
            self._props['label'] = label
        if stamp is not None:
            self._props['stamp'] = stamp
        if avatar is not None:
            self._props['avatar'] = avatar
        self._props['sent'] = sent