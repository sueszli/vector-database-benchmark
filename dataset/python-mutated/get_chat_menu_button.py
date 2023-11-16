from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union
from ..types import MenuButtonCommands, MenuButtonDefault, MenuButtonWebApp
from .base import TelegramMethod

class GetChatMenuButton(TelegramMethod[Union[MenuButtonDefault, MenuButtonWebApp, MenuButtonCommands]]):
    """
    Use this method to get the current value of the bot's menu button in a private chat, or the default menu button. Returns :class:`aiogram.types.menu_button.MenuButton` on success.

    Source: https://core.telegram.org/bots/api#getchatmenubutton
    """
    __returning__ = Union[MenuButtonDefault, MenuButtonWebApp, MenuButtonCommands]
    __api_method__ = 'getChatMenuButton'
    chat_id: Optional[int] = None
    "Unique identifier for the target private chat. If not specified, default bot's menu button will be returned"
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, chat_id: Optional[int]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(chat_id=chat_id, **__pydantic_kwargs)