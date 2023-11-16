import datetime
import re
from telegram import Chat, Message, MessageEntity, Update, User
from tests.auxil.ci_bots import BOT_INFO_PROVIDER
from tests.auxil.pytest_classes import make_bot
CMD_PATTERN = re.compile('/[\\da-z_]{1,32}(?:@\\w{1,32})?')
DATE = datetime.datetime.now()

def make_message(text, **kwargs):
    if False:
        print('Hello World!')
    '\n    Testing utility factory to create a fake ``telegram.Message`` with\n    reasonable defaults for mimicking a real message.\n    :param text: (str) message text\n    :return: a (fake) ``telegram.Message``\n    '
    bot = kwargs.pop('bot', None)
    if bot is None:
        bot = make_bot(BOT_INFO_PROVIDER.get_info())
    message = Message(message_id=1, from_user=kwargs.pop('user', User(id=1, first_name='', is_bot=False)), date=kwargs.pop('date', DATE), chat=kwargs.pop('chat', Chat(id=1, type='')), text=text, **kwargs)
    message.set_bot(bot)
    return message

def make_command_message(text, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Testing utility factory to create a message containing a single telegram\n    command.\n    Mimics the Telegram API in that it identifies commands within the message\n    and tags the returned ``Message`` object with the appropriate ``MessageEntity``\n    tag (but it does this only for commands).\n\n    :param text: (str) message text containing (or not) the command\n    :return: a (fake) ``telegram.Message`` containing only the command\n    '
    match = re.search(CMD_PATTERN, text)
    entities = [MessageEntity(type=MessageEntity.BOT_COMMAND, offset=match.start(0), length=len(match.group(0)))] if match else []
    return make_message(text, entities=entities, **kwargs)

def make_message_update(message, message_factory=make_message, edited=False, **kwargs):
    if False:
        return 10
    '\n    Testing utility factory to create an update from a message, as either a\n    ``telegram.Message`` or a string. In the latter case ``message_factory``\n    is used to convert ``message`` to a ``telegram.Message``.\n    :param message: either a ``telegram.Message`` or a string with the message text\n    :param message_factory: function to convert the message text into a ``telegram.Message``\n    :param edited: whether the message should be stored as ``edited_message`` (vs. ``message``)\n    :return: ``telegram.Update`` with the given message\n    '
    if not isinstance(message, Message):
        message = message_factory(message, **kwargs)
    update_kwargs = {'message' if not edited else 'edited_message': message}
    return Update(0, **update_kwargs)

def make_command_update(message, edited=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Testing utility factory to create an update from a message that potentially\n    contains a command. See ``make_command_message`` for more details.\n    :param message: message potentially containing a command\n    :param edited: whether the message should be stored as ``edited_message`` (vs. ``message``)\n    :return: ``telegram.Update`` with the given message\n    '
    return make_message_update(message, make_command_message, edited, **kwargs)