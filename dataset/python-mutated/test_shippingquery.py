import pytest
from telegram import Bot, ShippingAddress, ShippingQuery, Update, User
from tests.auxil.bot_method_checks import check_defaults_handling, check_shortcut_call, check_shortcut_signature
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def shipping_query(bot):
    if False:
        for i in range(10):
            print('nop')
    sq = ShippingQuery(TestShippingQueryBase.id_, TestShippingQueryBase.from_user, TestShippingQueryBase.invoice_payload, TestShippingQueryBase.shipping_address)
    sq.set_bot(bot)
    return sq

class TestShippingQueryBase:
    id_ = '5'
    invoice_payload = 'invoice_payload'
    from_user = User(0, '', False)
    shipping_address = ShippingAddress('GB', '', 'London', '12 Grimmauld Place', '', 'WC1')

class TestShippingQueryWithoutRequest(TestShippingQueryBase):

    def test_slot_behaviour(self, shipping_query):
        if False:
            return 10
        inst = shipping_query
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            i = 10
            return i + 15
        json_dict = {'id': self.id_, 'invoice_payload': self.invoice_payload, 'from': self.from_user.to_dict(), 'shipping_address': self.shipping_address.to_dict()}
        shipping_query = ShippingQuery.de_json(json_dict, bot)
        assert shipping_query.api_kwargs == {}
        assert shipping_query.id == self.id_
        assert shipping_query.invoice_payload == self.invoice_payload
        assert shipping_query.from_user == self.from_user
        assert shipping_query.shipping_address == self.shipping_address
        assert shipping_query.get_bot() is bot

    def test_to_dict(self, shipping_query):
        if False:
            while True:
                i = 10
        shipping_query_dict = shipping_query.to_dict()
        assert isinstance(shipping_query_dict, dict)
        assert shipping_query_dict['id'] == shipping_query.id
        assert shipping_query_dict['invoice_payload'] == shipping_query.invoice_payload
        assert shipping_query_dict['from'] == shipping_query.from_user.to_dict()
        assert shipping_query_dict['shipping_address'] == shipping_query.shipping_address.to_dict()

    def test_equality(self):
        if False:
            return 10
        a = ShippingQuery(self.id_, self.from_user, self.invoice_payload, self.shipping_address)
        b = ShippingQuery(self.id_, self.from_user, self.invoice_payload, self.shipping_address)
        c = ShippingQuery(self.id_, None, '', None)
        d = ShippingQuery(0, self.from_user, self.invoice_payload, self.shipping_address)
        e = Update(self.id_)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)

    async def test_answer(self, monkeypatch, shipping_query):

        async def make_assertion(*_, **kwargs):
            return kwargs['shipping_query_id'] == shipping_query.id
        assert check_shortcut_signature(ShippingQuery.answer, Bot.answer_shipping_query, ['shipping_query_id'], [])
        assert await check_shortcut_call(shipping_query.answer, shipping_query._bot, 'answer_shipping_query')
        assert await check_defaults_handling(shipping_query.answer, shipping_query._bot)
        monkeypatch.setattr(shipping_query._bot, 'answer_shipping_query', make_assertion)
        assert await shipping_query.answer(ok=True)