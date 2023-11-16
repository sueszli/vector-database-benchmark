"""This module contains subclasses of classes from the python-telegram-bot library that
modify behavior of the respective parent classes in order to make them easier to use in the
pytest framework. A common change is to allow monkeypatching of the class members by not
enforcing slots in the subclasses."""
from telegram import Bot, User
from telegram.ext import Application, ExtBot
from tests.auxil.ci_bots import BOT_INFO_PROVIDER
from tests.auxil.constants import PRIVATE_KEY
from tests.auxil.envvars import TEST_WITH_OPT_DEPS
from tests.auxil.networking import NonchalantHttpxRequest

def _get_bot_user(token: str) -> User:
    if False:
        print('Hello World!')
    'Used to return a mock user in bot.get_me(). This saves API calls on every init.'
    bot_info = BOT_INFO_PROVIDER.get_info()
    user_id = int(token.split(':')[0])
    first_name = bot_info.get('name')
    username = bot_info.get('username').strip('@')
    return User(user_id, first_name, is_bot=True, username=username, can_join_groups=True, can_read_all_group_messages=False, supports_inline_queries=True)

async def _mocked_get_me(bot: Bot):
    if bot._bot_user is None:
        bot._bot_user = _get_bot_user(bot.token)
    return bot._bot_user

class PytestExtBot(ExtBot):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._unfreeze()

    async def get_me(self, *args, **kwargs):
        return await _mocked_get_me(self)

class PytestBot(Bot):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._unfreeze()

    async def get_me(self, *args, **kwargs):
        return await _mocked_get_me(self)

class PytestApplication(Application):
    pass

def make_bot(bot_info=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Tests are executed on tg.ext.ExtBot, as that class only extends the functionality of tg.bot\n    '
    token = kwargs.pop('token', (bot_info or {}).get('token'))
    private_key = kwargs.pop('private_key', PRIVATE_KEY)
    kwargs.pop('token', None)
    return PytestExtBot(token=token, private_key=private_key if TEST_WITH_OPT_DEPS else None, request=NonchalantHttpxRequest(8), get_updates_request=NonchalantHttpxRequest(1), **kwargs)