import asyncio
from collections import OrderedDict
import pytest
from telegram import Bot
from telegram.ext import CallbackContext, JobQueue, TypeHandler
from tests.auxil.slots import mro_slots

class TestTypeHandler:
    test_flag = False

    def test_slot_behaviour(self):
        if False:
            print('Hello World!')
        inst = TypeHandler(dict, self.callback)
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            print('Hello World!')
        self.test_flag = False

    async def callback(self, update, context):
        self.test_flag = isinstance(context, CallbackContext) and isinstance(context.bot, Bot) and isinstance(update, dict) and isinstance(context.update_queue, asyncio.Queue) and isinstance(context.job_queue, JobQueue) and (context.user_data is None) and (context.chat_data is None) and isinstance(context.bot_data, dict)

    async def test_basic(self, app):
        handler = TypeHandler(dict, self.callback)
        app.add_handler(handler)
        assert handler.check_update({'a': 1, 'b': 2})
        assert not handler.check_update('not a dict')
        async with app:
            await app.process_update({'a': 1, 'b': 2})
        assert self.test_flag

    def test_strict(self):
        if False:
            return 10
        handler = TypeHandler(dict, self.callback, strict=True)
        o = OrderedDict({'a': 1, 'b': 2})
        assert handler.check_update({'a': 1, 'b': 2})
        assert not handler.check_update(o)