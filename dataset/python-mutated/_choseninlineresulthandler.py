"""This module contains the ChosenInlineResultHandler class."""
import re
from typing import TYPE_CHECKING, Any, Match, Optional, Pattern, TypeVar, Union, cast
from telegram import Update
from telegram._utils.defaultvalue import DEFAULT_TRUE
from telegram._utils.types import DVType
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT, HandlerCallback
RT = TypeVar('RT')
if TYPE_CHECKING:
    from telegram.ext import Application

class ChosenInlineResultHandler(BaseHandler[Update, CCT]):
    """Handler class to handle Telegram updates that contain
    :attr:`telegram.Update.chosen_inline_result`.

    Warning:
        When setting :paramref:`block` to :obj:`False`, you cannot rely on adding custom
        attributes to :class:`telegram.ext.CallbackContext`. See its docs for more info.

    Args:
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
        pattern (:obj:`str` | :func:`re.Pattern <re.compile>`, optional): Regex pattern. If not
            :obj:`None`, :func:`re.match`
            is used on :attr:`telegram.ChosenInlineResult.result_id` to determine if an update
            should be handled by this handler. This is accessible in the callback as
            :attr:`telegram.ext.CallbackContext.matches`.

            .. versionadded:: 13.6
    Attributes:
        callback (:term:`coroutine function`): The callback function for this handler.
        block (:obj:`bool`): Determines whether the return value of the callback should be
            awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`.
        pattern (`Pattern`): Optional. Regex pattern to test
            :attr:`telegram.ChosenInlineResult.result_id` against.

            .. versionadded:: 13.6

    """
    __slots__ = ('pattern',)

    def __init__(self, callback: HandlerCallback[Update, CCT, RT], block: DVType[bool]=DEFAULT_TRUE, pattern: Optional[Union[str, Pattern[str]]]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(callback, block=block)
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.pattern: Optional[Union[str, Pattern[str]]] = pattern

    def check_update(self, update: object) -> Optional[Union[bool, object]]:
        if False:
            for i in range(10):
                print('nop')
        "Determines whether an update should be passed to this handler's :attr:`callback`.\n\n        Args:\n            update (:class:`telegram.Update` | :obj:`object`): Incoming update.\n\n        Returns:\n            :obj:`bool` | :obj:`re.match`\n\n        "
        if isinstance(update, Update) and update.chosen_inline_result:
            if self.pattern:
                if (match := re.match(self.pattern, update.chosen_inline_result.result_id)):
                    return match
            else:
                return True
        return None

    def collect_additional_context(self, context: CCT, update: Update, application: 'Application[Any, CCT, Any, Any, Any, Any]', check_result: Union[bool, Match[str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'This function adds the matched regex pattern result to\n        :attr:`telegram.ext.CallbackContext.matches`.\n        '
        if self.pattern:
            check_result = cast(Match, check_result)
            context.matches = [check_result]