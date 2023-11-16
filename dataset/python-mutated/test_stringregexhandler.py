import asyncio
import re
import pytest
from telegram import Bot, CallbackQuery, Chat, ChosenInlineResult, InlineQuery, Message, PreCheckoutQuery, ShippingQuery, Update, User
from telegram.ext import CallbackContext, JobQueue, StringRegexHandler
from tests.auxil.slots import mro_slots
message = Message(1, None, Chat(1, ''), from_user=User(1, '', False), text='Text')
params = [{'message': message}, {'edited_message': message}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat', message=message)}, {'channel_post': message}, {'edited_channel_post': message}, {'inline_query': InlineQuery(1, User(1, '', False), '', '')}, {'chosen_inline_result': ChosenInlineResult('id', User(1, '', False), '')}, {'shipping_query': ShippingQuery('id', User(1, '', False), '', None)}, {'pre_checkout_query': PreCheckoutQuery('id', User(1, '', False), '', 0, '')}, {'callback_query': CallbackQuery(1, User(1, '', False), 'chat')}]
ids = ('message', 'edited_message', 'callback_query', 'channel_post', 'edited_channel_post', 'inline_query', 'chosen_inline_result', 'shipping_query', 'pre_checkout_query', 'callback_query_without_message')

@pytest.fixture(scope='class', params=params, ids=ids)
def false_update(request):
    if False:
        print('Hello World!')
    return Update(update_id=1, **request.param)

class TestStringRegexHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            i = 10
            return i + 15
        inst = StringRegexHandler('pfft', self.callback)
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            return 10
        self.test_flag = False

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, str) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue)

    async def callback_pattern(self, update, context):
        if context.matches[0].groups():
            self.test_flag = context.matches[0].groups() == ('t', ' message')
        if context.matches[0].groupdict():
            self.test_flag = context.matches[0].groupdict() == {'begin': 't', 'end': ' message'}

    @pytest.mark.parametrize('compile', [True, False])
    async def test_basic(self, app, compile):
        pattern = '(?P<begin>.*)est(?P<end>.*)'
        if compile:
            pattern = re.compile('(?P<begin>.*)est(?P<end>.*)')
        handler = StringRegexHandler(pattern, self.callback)
        app.add_handler(handler)
        assert handler.check_update('test message')
        async with app:
            await app.process_update('test message')
        assert self.test_flag
        assert not handler.check_update('does not match')

    def test_other_update_types(self, false_update):
        if False:
            for i in range(10):
                print('nop')
        handler = StringRegexHandler('test', self.callback)
        assert not handler.check_update(false_update)

    async def test_context_pattern(self, app):
        handler = StringRegexHandler('(t)est(.*)', self.callback_pattern)
        app.add_handler(handler)
        async with app:
            await app.process_update('test message')
            assert self.test_flag
            app.remove_handler(handler)
            handler = StringRegexHandler('(t)est(.*)', self.callback_pattern)
            app.add_handler(handler)
            await app.process_update('test message')
            assert self.test_flag