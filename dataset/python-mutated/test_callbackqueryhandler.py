import asyncio
import pytest
from telegram import Bot, CallbackQuery, Chat, ChosenInlineResult, InlineQuery, Message, PreCheckoutQuery, ShippingQuery, Update, User
from telegram.ext import CallbackContext, CallbackQueryHandler, InvalidCallbackData, JobQueue
from tests.auxil.slots import mro_slots
message = Message(1, None, Chat(1, ''), from_user=User(1, '', False), text='Text')
params = [{'message': message}, {'edited_message': message}, {'channel_post': message}, {'edited_channel_post': message}, {'inline_query': InlineQuery(1, User(1, '', False), '', '')}, {'chosen_inline_result': ChosenInlineResult('id', User(1, '', False), '')}, {'shipping_query': ShippingQuery('id', User(1, '', False), '', None)}, {'pre_checkout_query': PreCheckoutQuery('id', User(1, '', False), '', 0, '')}]
ids = ('message', 'edited_message', 'channel_post', 'edited_channel_post', 'inline_query', 'chosen_inline_result', 'shipping_query', 'pre_checkout_query')

@pytest.fixture(scope='class', params=params, ids=ids)
def false_update(request):
    if False:
        return 10
    return Update(update_id=2, **request.param)

@pytest.fixture()
def callback_query(bot):
    if False:
        for i in range(10):
            print('nop')
    update = Update(0, callback_query=CallbackQuery(2, User(1, '', False), None, data='test data'))
    update._unfreeze()
    update.callback_query._unfreeze()
    return update

class TestCallbackQueryHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            print('Hello World!')
        handler = CallbackQueryHandler(self.callback_data_1)
        for attr in handler.__slots__:
            assert getattr(handler, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(handler)) == len(set(mro_slots(handler))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            i = 10
            return i + 15
        self.test_flag = False

    def callback_basic(self, update, context):
        if False:
            return 10
        test_bot = isinstance(context.bot, Bot)
        test_update = isinstance(update, Update)
        self.test_flag = test_bot and test_update

    def callback_data_1(self, bot, update, user_data=None, chat_data=None):
        if False:
            print('Hello World!')
        self.test_flag = user_data is not None or chat_data is not None

    def callback_data_2(self, bot, update, user_data=None, chat_data=None):
        if False:
            print('Hello World!')
        self.test_flag = user_data is not None and chat_data is not None

    def callback_queue_1(self, bot, update, job_queue=None, update_queue=None):
        if False:
            for i in range(10):
                print('nop')
        self.test_flag = job_queue is not None or update_queue is not None

    def callback_queue_2(self, bot, update, job_queue=None, update_queue=None):
        if False:
            for i in range(10):
                print('nop')
        self.test_flag = job_queue is not None and update_queue is not None

    def callback_group(self, bot, update, groups=None, groupdict=None):
        if False:
            while True:
                i = 10
        if groups is not None:
            self.test_flag = groups == ('t', ' data')
        if groupdict is not None:
            self.test_flag = groupdict == {'begin': 't', 'end': ' data'}

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, Update) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue) and isinstance(context.user_data, dict) and (context.chat_data is None) and isinstance(context.bot_data, dict) and isinstance(update.callback_query, CallbackQuery)

    def callback_pattern(self, update, context):
        if False:
            while True:
                i = 10
        if context.matches[0].groups():
            self.test_flag = context.matches[0].groups() == ('t', ' data')
        if context.matches[0].groupdict():
            self.test_flag = context.matches[0].groupdict() == {'begin': 't', 'end': ' data'}

    def test_with_pattern(self, callback_query):
        if False:
            while True:
                i = 10
        handler = CallbackQueryHandler(self.callback_basic, pattern='.*est.*')
        assert handler.check_update(callback_query)
        callback_query.callback_query.data = 'nothing here'
        assert not handler.check_update(callback_query)
        callback_query.callback_query.data = None
        callback_query.callback_query.game_short_name = 'this is a short game name'
        assert not handler.check_update(callback_query)
        callback_query.callback_query.data = object()
        assert not handler.check_update(callback_query)
        callback_query.callback_query.data = InvalidCallbackData()
        assert not handler.check_update(callback_query)

    def test_with_callable_pattern(self, callback_query):
        if False:
            i = 10
            return i + 15

        class CallbackData:
            pass

        def pattern(callback_data):
            if False:
                print('Hello World!')
            return isinstance(callback_data, CallbackData)
        handler = CallbackQueryHandler(self.callback_basic, pattern=pattern)
        callback_query.callback_query.data = CallbackData()
        assert handler.check_update(callback_query)
        callback_query.callback_query.data = 'callback_data'
        assert not handler.check_update(callback_query)

    def test_with_type_pattern(self, callback_query):
        if False:
            i = 10
            return i + 15

        class CallbackData:
            pass
        handler = CallbackQueryHandler(self.callback_basic, pattern=CallbackData)
        callback_query.callback_query.data = CallbackData()
        assert handler.check_update(callback_query)
        callback_query.callback_query.data = 'callback_data'
        assert not handler.check_update(callback_query)
        handler = CallbackQueryHandler(self.callback_basic, pattern=bool)
        callback_query.callback_query.data = False
        assert handler.check_update(callback_query)
        callback_query.callback_query.data = 'callback_data'
        assert not handler.check_update(callback_query)

    def test_other_update_types(self, false_update):
        if False:
            for i in range(10):
                print('nop')
        handler = CallbackQueryHandler(self.callback_basic)
        assert not handler.check_update(false_update)

    async def test_context(self, app, callback_query):
        handler = CallbackQueryHandler(self.callback)
        app.add_handler(handler)
        async with app:
            await app.process_update(callback_query)
            assert self.test_flag

    async def test_context_pattern(self, app, callback_query):
        handler = CallbackQueryHandler(self.callback_pattern, pattern='(?P<begin>.*)est(?P<end>.*)')
        app.add_handler(handler)
        async with app:
            await app.process_update(callback_query)
            assert self.test_flag
            app.remove_handler(handler)
            handler = CallbackQueryHandler(self.callback_pattern, pattern='(t)est(.*)')
            app.add_handler(handler)
            await app.process_update(callback_query)
            assert self.test_flag

    async def test_context_callable_pattern(self, app, callback_query):

        class CallbackData:
            pass

        def pattern(callback_data):
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(callback_data, CallbackData)

        def callback(update, context):
            if False:
                i = 10
                return i + 15
            assert context.matches is None
        handler = CallbackQueryHandler(callback, pattern=pattern)
        app.add_handler(handler)
        async with app:
            await app.process_update(callback_query)

    def test_async_pattern(self):
        if False:
            return 10

        async def pattern():
            pass
        with pytest.raises(TypeError, match='must not be a coroutine function'):
            CallbackQueryHandler(self.callback, pattern=pattern)