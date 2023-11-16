from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from .input_message_content import InputMessageContent

class InputContactMessageContent(InputMessageContent):
    """
    Represents the `content <https://core.telegram.org/bots/api#inputmessagecontent>`_ of a contact message to be sent as the result of an inline query.

    Source: https://core.telegram.org/bots/api#inputcontactmessagecontent
    """
    phone_number: str
    "Contact's phone number"
    first_name: str
    "Contact's first name"
    last_name: Optional[str] = None
    "*Optional*. Contact's last name"
    vcard: Optional[str] = None
    '*Optional*. Additional data about the contact in the form of a `vCard <https://en.wikipedia.org/wiki/VCard>`_, 0-2048 bytes'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, phone_number: str, first_name: str, last_name: Optional[str]=None, vcard: Optional[str]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(phone_number=phone_number, first_name=first_name, last_name=last_name, vcard=vcard, **__pydantic_kwargs)