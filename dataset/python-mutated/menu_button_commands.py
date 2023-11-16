from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..enums import MenuButtonType
from .menu_button import MenuButton

class MenuButtonCommands(MenuButton):
    """
    Represents a menu button, which opens the bot's list of commands.

    Source: https://core.telegram.org/bots/api#menubuttoncommands
    """
    type: Literal[MenuButtonType.COMMANDS] = MenuButtonType.COMMANDS
    'Type of the button, must be *commands*'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, type: Literal[MenuButtonType.COMMANDS]=MenuButtonType.COMMANDS, **__pydantic_kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(type=type, **__pydantic_kwargs)