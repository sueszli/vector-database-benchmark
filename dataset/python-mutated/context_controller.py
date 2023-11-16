from typing import TYPE_CHECKING, Any, Optional
from pydantic import BaseModel, PrivateAttr
from typing_extensions import Self
if TYPE_CHECKING:
    from aiogram.client.bot import Bot

class BotContextController(BaseModel):
    _bot: Optional['Bot'] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        if False:
            print('Hello World!')
        self._bot = __context.get('bot') if __context else None

    def as_(self, bot: Optional['Bot']) -> Self:
        if False:
            for i in range(10):
                print('nop')
        '\n        Bind object to a bot instance.\n\n        :param bot: Bot instance\n        :return: self\n        '
        self._bot = bot
        return self

    @property
    def bot(self) -> Optional['Bot']:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get bot instance.\n\n        :return: Bot instance\n        '
        return self._bot