from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramObject

class PollOption(TelegramObject):
    """
    This object contains information about one answer option in a poll.

    Source: https://core.telegram.org/bots/api#polloption
    """
    text: str
    'Option text, 1-100 characters'
    voter_count: int
    'Number of users that voted for this option'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, text: str, voter_count: int, **__pydantic_kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(text=text, voter_count=voter_count, **__pydantic_kwargs)