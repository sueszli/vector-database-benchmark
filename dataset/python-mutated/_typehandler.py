"""This module contains the TypeHandler class."""
from typing import Optional, Type, TypeVar
from telegram._utils.defaultvalue import DEFAULT_TRUE
from telegram._utils.types import DVType
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT, HandlerCallback
RT = TypeVar('RT')
UT = TypeVar('UT')

class TypeHandler(BaseHandler[UT, CCT]):
    """Handler class to handle updates of custom types.

    Warning:
        When setting :paramref:`block` to :obj:`False`, you cannot rely on adding custom
        attributes to :class:`telegram.ext.CallbackContext`. See its docs for more info.

    Args:
        type (:external:class:`type`): The :external:class:`type` of updates this handler should
            process, as determined by :obj:`isinstance`
        callback (:term:`coroutine function`): The callback function for this handler. Will be
            called when :meth:`check_update` has determined that an update should be processed by
            this handler. Callback signature::

                async def callback(update: Update, context: CallbackContext)

            The return value of the callback is usually ignored except for the special case of
            :class:`telegram.ext.ConversationHandler`.
        strict (:obj:`bool`, optional): Use ``type`` instead of :obj:`isinstance`.
            Default is :obj:`False`.
        block (:obj:`bool`, optional): Determines whether the return value of the callback should
            be awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`. Defaults to :obj:`True`.

            .. seealso:: :wiki:`Concurrency`

    Attributes:
        type (:external:class:`type`): The :external:class:`type` of updates this handler should
            process.
        callback (:term:`coroutine function`): The callback function for this handler.
        strict (:obj:`bool`): Use :external:class:`type` instead of :obj:`isinstance`. Default is
            :obj:`False`.
        block (:obj:`bool`): Determines whether the return value of the callback should be
            awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`.

    """
    __slots__ = ('type', 'strict')

    def __init__(self, type: Type[UT], callback: HandlerCallback[UT, CCT, RT], strict: bool=False, block: DVType[bool]=DEFAULT_TRUE):
        if False:
            return 10
        super().__init__(callback, block=block)
        self.type: Type[UT] = type
        self.strict: Optional[bool] = strict

    def check_update(self, update: object) -> bool:
        if False:
            while True:
                i = 10
        "Determines whether an update should be passed to this handler's :attr:`callback`.\n\n        Args:\n            update (:obj:`object`): Incoming update.\n\n        Returns:\n            :obj:`bool`\n\n        "
        if not self.strict:
            return isinstance(update, self.type)
        return type(update) is self.type