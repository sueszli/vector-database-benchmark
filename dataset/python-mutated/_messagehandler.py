"""This module contains the MessageHandler class."""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union
from telegram import Update
from telegram._utils.defaultvalue import DEFAULT_TRUE
from telegram._utils.types import DVType
from telegram.ext import filters as filters_module
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT, HandlerCallback
if TYPE_CHECKING:
    from telegram.ext import Application
RT = TypeVar('RT')

class MessageHandler(BaseHandler[Update, CCT]):
    """Handler class to handle Telegram messages. They might contain text, media or status
    updates.

    Warning:
        When setting :paramref:`block` to :obj:`False`, you cannot rely on adding custom
        attributes to :class:`telegram.ext.CallbackContext`. See its docs for more info.

    Args:
        filters (:class:`telegram.ext.filters.BaseFilter`): A filter inheriting from
            :class:`telegram.ext.filters.BaseFilter`. Standard filters can be found in
            :mod:`telegram.ext.filters`. Filters can be combined using bitwise
            operators (& for and, | for or, ~ for not). Passing :obj:`None` is a shortcut
            to passing :class:`telegram.ext.filters.ALL`.

            .. seealso:: :wiki:`Advanced Filters <Extensions---Advanced-Filters>`
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
        filters (:class:`telegram.ext.filters.BaseFilter`): Only allow updates with these Filters.
            See :mod:`telegram.ext.filters` for a full list of all available filters.
        callback (:term:`coroutine function`): The callback function for this handler.
        block (:obj:`bool`): Determines whether the return value of the callback should be
            awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`.

    """
    __slots__ = ('filters',)

    def __init__(self, filters: filters_module.BaseFilter, callback: HandlerCallback[Update, CCT, RT], block: DVType[bool]=DEFAULT_TRUE):
        if False:
            while True:
                i = 10
        super().__init__(callback, block=block)
        self.filters: filters_module.BaseFilter = filters if filters is not None else filters_module.ALL

    def check_update(self, update: object) -> Optional[Union[bool, Dict[str, List[Any]]]]:
        if False:
            return 10
        "Determines whether an update should be passed to this handler's :attr:`callback`.\n\n        Args:\n            update (:class:`telegram.Update` | :obj:`object`): Incoming update.\n\n        Returns:\n            :obj:`bool`\n\n        "
        if isinstance(update, Update):
            return self.filters.check_update(update) or False
        return None

    def collect_additional_context(self, context: CCT, update: Update, application: 'Application[Any, CCT, Any, Any, Any, Any]', check_result: Optional[Union[bool, Dict[str, object]]]) -> None:
        if False:
            i = 10
            return i + 15
        'Adds possible output of data filters to the :class:`CallbackContext`.'
        if isinstance(check_result, dict):
            context.update(check_result)