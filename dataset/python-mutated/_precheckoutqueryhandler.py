"""This module contains the PreCheckoutQueryHandler class."""
from telegram import Update
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT

class PreCheckoutQueryHandler(BaseHandler[Update, CCT]):
    """Handler class to handle Telegram :attr:`telegram.Update.pre_checkout_query`.

    Warning:
        When setting :paramref:`block` to :obj:`False`, you cannot rely on adding custom
        attributes to :class:`telegram.ext.CallbackContext`. See its docs for more info.

    Examples:
        :any:`Payment Bot <examples.paymentbot>`

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

    Attributes:
        callback (:term:`coroutine function`): The callback function for this handler.
        block (:obj:`bool`): Determines whether the callback will run in a blocking way..

    """
    __slots__ = ()

    def check_update(self, update: object) -> bool:
        if False:
            while True:
                i = 10
        "Determines whether an update should be passed to this handler's :attr:`callback`.\n\n        Args:\n            update (:class:`telegram.Update` | :obj:`object`): Incoming update.\n\n        Returns:\n            :obj:`bool`\n\n        "
        return isinstance(update, Update) and bool(update.pre_checkout_query)