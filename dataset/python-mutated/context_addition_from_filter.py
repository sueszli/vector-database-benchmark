from typing import Any, Dict, Optional, Union
from aiogram import Router
from aiogram.filters import Filter
from aiogram.types import Message, User
router = Router(name=__name__)

class HelloFilter(Filter):

    def __init__(self, name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        self.name = name

    async def __call__(self, message: Message, event_from_user: User) -> Union[bool, Dict[str, Any]]:
        if message.text.casefold() == 'hello':
            return {'name': event_from_user.mention_html(name=self.name)}
        return False

@router.message(HelloFilter())
async def my_handler(message: Message, name: str) -> Any:
    return message.answer('Hello, {name}!'.format(name=name))