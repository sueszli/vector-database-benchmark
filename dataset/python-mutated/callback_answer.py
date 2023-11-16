from typing import Any, Awaitable, Callable, Dict, Optional, Union
from aiogram import BaseMiddleware, loggers
from aiogram.dispatcher.flags import get_flag
from aiogram.exceptions import CallbackAnswerException
from aiogram.methods import AnswerCallbackQuery
from aiogram.types import CallbackQuery, TelegramObject

class CallbackAnswer:

    def __init__(self, answered: bool, disabled: bool=False, text: Optional[str]=None, show_alert: Optional[bool]=None, url: Optional[str]=None, cache_time: Optional[int]=None) -> None:
        if False:
            return 10
        '\n        Callback answer configuration\n\n        :param answered: this request is already answered by middleware\n        :param disabled: answer will not be performed\n        :param text: answer with text\n        :param show_alert: show alert\n        :param url: game url\n        :param cache_time: cache answer for some time\n        '
        self._answered = answered
        self._disabled = disabled
        self._text = text
        self._show_alert = show_alert
        self._url = url
        self._cache_time = cache_time

    def disable(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Deactivate answering for this handler\n        '
        self.disabled = True

    @property
    def disabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Indicates that automatic answer is disabled in this handler'
        return self._disabled

    @disabled.setter
    def disabled(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._answered:
            raise CallbackAnswerException("Can't change disabled state after answer")
        self._disabled = value

    @property
    def answered(self) -> bool:
        if False:
            return 10
        '\n        Indicates that request is already answered by middleware\n        '
        return self._answered

    @property
    def text(self) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Response text\n        :return:\n        '
        return self._text

    @text.setter
    def text(self, value: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._answered:
            raise CallbackAnswerException("Can't change text after answer")
        self._text = value

    @property
    def show_alert(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        '\n        Whether to display an alert\n        '
        return self._show_alert

    @show_alert.setter
    def show_alert(self, value: Optional[bool]) -> None:
        if False:
            i = 10
            return i + 15
        if self._answered:
            raise CallbackAnswerException("Can't change show_alert after answer")
        self._show_alert = value

    @property
    def url(self) -> Optional[str]:
        if False:
            return 10
        '\n        Game url\n        '
        return self._url

    @url.setter
    def url(self, value: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._answered:
            raise CallbackAnswerException("Can't change url after answer")
        self._url = value

    @property
    def cache_time(self) -> Optional[int]:
        if False:
            return 10
        '\n        Response cache time\n        '
        return self._cache_time

    @cache_time.setter
    def cache_time(self, value: Optional[int]) -> None:
        if False:
            while True:
                i = 10
        if self._answered:
            raise CallbackAnswerException("Can't change cache_time after answer")
        self._cache_time = value

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        args = ', '.join((f'{k}={v!r}' for (k, v) in {'answered': self.answered, 'disabled': self.disabled, 'text': self.text, 'show_alert': self.show_alert, 'url': self.url, 'cache_time': self.cache_time}.items() if v is not None))
        return f'{type(self).__name__}({args})'

class CallbackAnswerMiddleware(BaseMiddleware):

    def __init__(self, pre: bool=False, text: Optional[str]=None, show_alert: Optional[bool]=None, url: Optional[str]=None, cache_time: Optional[int]=None) -> None:
        if False:
            return 10
        '\n        Inner middleware for callback query handlers, can be useful in bots with a lot of callback\n        handlers to automatically take answer to all requests\n\n        :param pre: send answer before execute handler\n        :param text: answer with text\n        :param show_alert: show alert\n        :param url: game url\n        :param cache_time: cache answer for some time\n        '
        self.pre = pre
        self.text = text
        self.show_alert = show_alert
        self.url = url
        self.cache_time = cache_time

    async def __call__(self, handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]], event: TelegramObject, data: Dict[str, Any]) -> Any:
        if not isinstance(event, CallbackQuery):
            return await handler(event, data)
        callback_answer = data['callback_answer'] = self.construct_callback_answer(properties=get_flag(data, 'callback_answer'))
        if not callback_answer.disabled and callback_answer.answered:
            await self.answer(event, callback_answer)
        try:
            return await handler(event, data)
        finally:
            if not callback_answer.disabled and (not callback_answer.answered):
                await self.answer(event, callback_answer)

    def construct_callback_answer(self, properties: Optional[Union[Dict[str, Any], bool]]) -> CallbackAnswer:
        if False:
            for i in range(10):
                print('nop')
        (pre, disabled, text, show_alert, url, cache_time) = (self.pre, False, self.text, self.show_alert, self.url, self.cache_time)
        if isinstance(properties, dict):
            pre = properties.get('pre', pre)
            disabled = properties.get('disabled', disabled)
            text = properties.get('text', text)
            show_alert = properties.get('show_alert', show_alert)
            url = properties.get('url', url)
            cache_time = properties.get('cache_time', cache_time)
        return CallbackAnswer(answered=pre, disabled=disabled, text=text, show_alert=show_alert, url=url, cache_time=cache_time)

    def answer(self, event: CallbackQuery, callback_answer: CallbackAnswer) -> AnswerCallbackQuery:
        if False:
            for i in range(10):
                print('nop')
        loggers.middlewares.info('Answer to callback query id=%s', event.id)
        return event.answer(text=callback_answer.text, show_alert=callback_answer.show_alert, url=callback_answer.url, cache_time=callback_answer.cache_time)