"""This module contains the StringCommandHandler class."""
from typing import TYPE_CHECKING, Any, List, Optional
from telegram._utils.defaultvalue import DEFAULT_TRUE
from telegram._utils.types import DVType
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT, RT, HandlerCallback
if TYPE_CHECKING:
    from telegram.ext import Application

class StringCommandHandler(BaseHandler[str, CCT]):
    """Handler class to handle string commands. Commands are string updates that start with
    ``/``. The handler will add a :obj:`list` to the
    :class:`CallbackContext` named :attr:`CallbackContext.args`. It will contain a list of strings,
    which is the text following the command split on single whitespace characters.

    Note:
        This handler is not used to handle Telegram :class:`telegram.Update`, but strings manually
        put in the queue. For example to send messages with the bot using command line or API.

    Warning:
        When setting :paramref:`block` to :obj:`False`, you cannot rely on adding custom
        attributes to :class:`telegram.ext.CallbackContext`. See its docs for more info.

    Args:
        command (:obj:`str`): The command this handler should listen for.
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
        command (:obj:`str`): The command this handler should listen for.
        callback (:term:`coroutine function`): The callback function for this handler.
        block (:obj:`bool`): Determines whether the return value of the callback should be
            awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`.

    """
    __slots__ = ('command',)

    def __init__(self, command: str, callback: HandlerCallback[str, CCT, RT], block: DVType[bool]=DEFAULT_TRUE):
        if False:
            return 10
        super().__init__(callback, block=block)
        self.command: str = command

    def check_update(self, update: object) -> Optional[List[str]]:
        if False:
            print('Hello World!')
        "Determines whether an update should be passed to this handler's :attr:`callback`.\n\n        Args:\n            update (:obj:`object`): The incoming update.\n\n        Returns:\n            List[:obj:`str`]: List containing the text command split on whitespace.\n\n        "
        if isinstance(update, str) and update.startswith('/'):
            args = update[1:].split(' ')
            if args[0] == self.command:
                return args[1:]
        return None

    def collect_additional_context(self, context: CCT, update: str, application: 'Application[Any, CCT, Any, Any, Any, Any]', check_result: Optional[List[str]]) -> None:
        if False:
            i = 10
            return i + 15
        'Add text after the command to :attr:`CallbackContext.args` as list, split on single\n        whitespaces.\n        '
        context.args = check_result