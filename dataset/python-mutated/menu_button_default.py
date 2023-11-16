from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..enums import MenuButtonType
from .menu_button import MenuButton

class MenuButtonDefault(MenuButton):
    """
    Describes that no specific value for the menu button was set.

    Source: https://core.telegram.org/bots/api#menubuttondefault
    """
    type: Literal[MenuButtonType.DEFAULT] = MenuButtonType.DEFAULT
    'Type of the button, must be *default*'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, type: Literal[MenuButtonType.DEFAULT]=MenuButtonType.DEFAULT, **__pydantic_kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(type=type, **__pydantic_kwargs)