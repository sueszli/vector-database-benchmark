"""This module contains the StringRegexHandler class."""
import re
from typing import TYPE_CHECKING, Any, Match, Optional, Pattern, TypeVar, Union
from telegram._utils.defaultvalue import DEFAULT_TRUE
from telegram._utils.types import DVType
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT, HandlerCallback
if TYPE_CHECKING:
    from telegram.ext import Application
RT = TypeVar('RT')

class StringRegexHandler(BaseHandler[str, CCT]):
    """Handler class to handle string updates based on a regex which checks the update content.

    Read the documentation of the :mod:`re` module for more information. The :func:`re.match`
    function is used to determine if an update should be handled by this handler.

    Note:
        This handler is not used to handle Telegram :class:`telegram.Update`, but strings manually
        put in the queue. For example to send messages with the bot using command line or API.

    Warning:
        When setting :paramref:`block` to :obj:`False`, you cannot rely on adding custom
        attributes to :class:`telegram.ext.CallbackContext`. See its docs for more info.

    Args:
        pattern (:obj:`str` | :func:`re.Pattern <re.compile>`): The regex pattern.
        callback (:term:`coroutine function`): The callback function for this handler. Will be
            called when :meth:`check_update` has determined that an update should be processed by
            this handler. Callback signature::

                async def callback(update: Update, context: CallbackContext)

            The return value of the callback is usually ignored except for the special case of
            :class:`telegram.ext.ConversationHandler`.
        block (:obj:`bool`, optional): Determines whether the return value of the callback should
            be awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`. Defaults to :obj:`True`.

            .. seealso:: :wiki:`Concurrency`

    Attributes:
        pattern (:obj:`str` | :func:`re.Pattern <re.compile>`): The regex pattern.
        callback (:term:`coroutine function`): The callback function for this handler.
        block (:obj:`bool`): Determines whether the return value of the callback should be
            awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`.

    """
    __slots__ = ('pattern',)

    def __init__(self, pattern: Union[str, Pattern[str]], callback: HandlerCallback[str, CCT, RT], block: DVType[bool]=DEFAULT_TRUE):
        if False:
            while True:
                i = 10
        super().__init__(callback, block=block)
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.pattern: Union[str, Pattern[str]] = pattern

    def check_update(self, update: object) -> Optional[Match[str]]:
        if False:
            print('Hello World!')
        "Determines whether an update should be passed to this handler's :attr:`callback`.\n\n        Args:\n            update (:obj:`object`): The incoming update.\n\n        Returns:\n            :obj:`None` | :obj:`re.match`\n\n        "
        if isinstance(update, str) and (match := re.match(self.pattern, update)):
            return match
        return None

    def collect_additional_context(self, context: CCT, update: str, application: 'Application[Any, CCT, Any, Any, Any, Any]', check_result: Optional[Match[str]]) -> None:
        if False:
            while True:
                i = 10
        'Add the result of ``re.match(pattern, update)`` to :attr:`CallbackContext.matches` as\n        list with one element.\n        '
        if self.pattern and check_result:
            context.matches = [check_result]