import pytest
from telegram import LabeledPrice, ShippingOption, Voice
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def shipping_option():
    if False:
        return 10
    return ShippingOption(TestShippingOptionBase.id_, TestShippingOptionBase.title, TestShippingOptionBase.prices)

class TestShippingOptionBase:
    id_ = 'id'
    title = 'title'
    prices = [LabeledPrice('Fish Container', 100), LabeledPrice('Premium Fish Container', 1000)]

class TestShippingOptionWithoutRequest(TestShippingOptionBase):

    def test_slot_behaviour(self, shipping_option):
        if False:
            while True:
                i = 10
        inst = shipping_option
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, shipping_option):
        if False:
            return 10
        assert shipping_option.id == self.id_
        assert shipping_option.title == self.title
        assert shipping_option.prices == tuple(self.prices)

    def test_to_dict(self, shipping_option):
        if False:
            return 10
        shipping_option_dict = shipping_option.to_dict()
        assert isinstance(shipping_option_dict, dict)
        assert shipping_option_dict['id'] == shipping_option.id
        assert shipping_option_dict['title'] == shipping_option.title
        assert shipping_option_dict['prices'][0] == shipping_option.prices[0].to_dict()
        assert shipping_option_dict['prices'][1] == shipping_option.prices[1].to_dict()

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = ShippingOption(self.id_, self.title, self.prices)
        b = ShippingOption(self.id_, self.title, self.prices)
        c = ShippingOption(self.id_, '', [])
        d = ShippingOption(0, self.title, self.prices)
        e = Voice(self.id_, 'someid', 0)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)