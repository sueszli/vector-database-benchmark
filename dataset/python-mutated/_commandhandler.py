"""This module contains the CommandHandler class."""
import re
from typing import TYPE_CHECKING, Any, FrozenSet, List, Optional, Tuple, TypeVar, Union
from telegram import MessageEntity, Update
from telegram._utils.defaultvalue import DEFAULT_TRUE
from telegram._utils.types import SCT, DVType
from telegram.ext import filters as filters_module
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT, FilterDataDict, HandlerCallback
if TYPE_CHECKING:
    from telegram.ext import Application
RT = TypeVar('RT')

class CommandHandler(BaseHandler[Update, CCT]):
    """Handler class to handle Telegram commands.

    Commands are Telegram messages that start with ``/``, optionally followed by an ``@`` and the
    bot's name and/or some additional text. The handler will add a :obj:`list` to the
    :class:`CallbackContext` named :attr:`CallbackContext.args`. It will contain a list of strings,
    which is the text following the command split on single or consecutive whitespace characters.

    By default, the handler listens to messages as well as edited messages. To change this behavior
    use :attr:`~filters.UpdateType.EDITED_MESSAGE <telegram.ext.filters.UpdateType.EDITED_MESSAGE>`
    in the filter argument.

    Note:
        :class:`CommandHandler` does *not* handle (edited) channel posts and does *not* handle
        commands that are part of a caption. Please use :class:`~telegram.ext.MessageHandler`
        with a suitable combination of filters (e.g.
        :attr:`telegram.ext.filters.UpdateType.CHANNEL_POSTS`,
        :attr:`telegram.ext.filters.CAPTION` and :class:`telegram.ext.filters.Regex`) to handle
        those messages.

    Warning:
        When setting :paramref:`block` to :obj:`False`, you cannot rely on adding custom
        attributes to :class:`telegram.ext.CallbackContext`. See its docs for more info.

    Examples:
        * :any:`Timer Bot <examples.timerbot>`
        * :any:`Error Handler Bot <examples.errorhandlerbot>`

    .. versionchanged:: 20.0

        * Renamed the attribute ``command`` to :attr:`commands`, which now is always a
          :class:`frozenset`
        * Updating the commands this handler listens to is no longer possible.

    Args:
        command (:obj:`str` | Collection[:obj:`str`]):
            The command or list of commands this handler should listen for. Case-insensitive.
            Limitations are the same as for :attr:`telegram.BotCommand.command`.
        callback (:term:`coroutine function`): The callback function for this handler. Will be
            called when :meth:`check_update` has determined that an update should be processed by
            this handler. Callback signature::

                async def callback(update: Update, context: CallbackContext)

            The return value of the callback is usually ignored except for the special case of
            :class:`telegram.ext.ConversationHandler`.
        filters (:class:`telegram.ext.filters.BaseFilter`, optional): A filter inheriting from
            :class:`telegram.ext.filters.BaseFilter`. Standard filters can be found in
            :mod:`telegram.ext.filters`. Filters can be combined using bitwise
            operators (``&`` for :keyword:`and`, ``|`` for :keyword:`or`, ``~`` for :keyword:`not`)
        block (:obj:`bool`, optional): Determines whether the return value of the callback should
            be awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`. Defaults to :obj:`True`.

            .. seealso:: :wiki:`Concurrency`
        has_args (:obj:`bool` | :obj:`int`, optional):
            Determines whether the command handler should process the update or not.
            If :obj:`True`, the handler will process any non-zero number of args.
            If :obj:`False`, the handler will only process if there are no args.
            if :obj:`int`, the handler will only process if there are exactly that many args.
            Defaults to :obj:`None`, which means the handler will process any or no args.

            .. versionadded:: 20.5

    Raises:
        :exc:`ValueError`: When the command is too long or has illegal chars.

    Attributes:
        commands (FrozenSet[:obj:`str`]): The set of commands this handler should listen for.
        callback (:term:`coroutine function`): The callback function for this handler.
        filters (:class:`telegram.ext.filters.BaseFilter`): Optional. Only allow updates with these
            filters.
        block (:obj:`bool`): Determines whether the return value of the callback should be
            awaited before processing the next handler in
            :meth:`telegram.ext.Application.process_update`.
        has_args (:obj:`bool` | :obj:`int` | None):
            Optional argument, otherwise all implementations of :class:`CommandHandler` will break.
            Defaults to :obj:`None`, which means the handler will process any args or no args.

            .. versionadded:: 20.5
    """
    __slots__ = ('commands', 'filters', 'has_args')

    def __init__(self, command: SCT[str], callback: HandlerCallback[Update, CCT, RT], filters: Optional[filters_module.BaseFilter]=None, block: DVType[bool]=DEFAULT_TRUE, has_args: Optional[Union[bool, int]]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(callback, block=block)
        if isinstance(command, str):
            commands = frozenset({command.lower()})
        else:
            commands = frozenset((x.lower() for x in command))
        for comm in commands:
            if not re.match('^[\\da-z_]{1,32}$', comm):
                raise ValueError(f'Command `{comm}` is not a valid bot command')
        self.commands: FrozenSet[str] = commands
        self.filters: filters_module.BaseFilter = filters if filters is not None else filters_module.UpdateType.MESSAGES
        self.has_args: Optional[Union[bool, int]] = has_args
        if isinstance(self.has_args, int) and self.has_args < 0:
            raise ValueError('CommandHandler argument has_args cannot be a negative integer')

    def _check_correct_args(self, args: List[str]) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        'Determines whether the args are correct for this handler. Implemented in check_update().\n        Args:\n            args (:obj:`list`): The args for the handler.\n        Returns:\n            :obj:`bool`: Whether the args are valid for this handler.\n        '
        if self.has_args is None or (self.has_args is True and args) or (self.has_args is False and (not args)) or (isinstance(self.has_args, int) and len(args) == self.has_args):
            return True
        return False

    def check_update(self, update: object) -> Optional[Union[bool, Tuple[List[str], Optional[Union[bool, FilterDataDict]]]]]:
        if False:
            while True:
                i = 10
        "Determines whether an update should be passed to this handler's :attr:`callback`.\n\n        Args:\n            update (:class:`telegram.Update` | :obj:`object`): Incoming update.\n\n        Returns:\n            :obj:`list`: The list of args for the handler.\n\n        "
        if isinstance(update, Update) and update.effective_message:
            message = update.effective_message
            if message.entities and message.entities[0].type == MessageEntity.BOT_COMMAND and (message.entities[0].offset == 0) and message.text and message.get_bot():
                command = message.text[1:message.entities[0].length]
                args = message.text.split()[1:]
                command_parts = command.split('@')
                command_parts.append(message.get_bot().username)
                if not (command_parts[0].lower() in self.commands and command_parts[1].lower() == message.get_bot().username.lower()):
                    return None
                if not self._check_correct_args(args):
                    return None
                filter_result = self.filters.check_update(update)
                if filter_result:
                    return (args, filter_result)
                return False
        return None

    def collect_additional_context(self, context: CCT, update: Update, application: 'Application[Any, CCT, Any, Any, Any, Any]', check_result: Optional[Union[bool, Tuple[List[str], Optional[bool]]]]) -> None:
        if False:
            print('Hello World!')
        'Add text after the command to :attr:`CallbackContext.args` as list, split on single\n        whitespaces and add output of data filters to :attr:`CallbackContext` as well.\n        '
        if isinstance(check_result, tuple):
            context.args = check_result[0]
            if isinstance(check_result[1], dict):
                context.update(check_result[1])