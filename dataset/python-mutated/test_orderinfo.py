import pytest
from telegram import OrderInfo, ShippingAddress
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def order_info():
    if False:
        while True:
            i = 10
    return OrderInfo(TestOrderInfoBase.name, TestOrderInfoBase.phone_number, TestOrderInfoBase.email, TestOrderInfoBase.shipping_address)

class TestOrderInfoBase:
    name = 'name'
    phone_number = 'phone_number'
    email = 'email'
    shipping_address = ShippingAddress('GB', '', 'London', '12 Grimmauld Place', '', 'WC1')

class TestOrderInfoWithoutRequest(TestOrderInfoBase):

    def test_slot_behaviour(self, order_info):
        if False:
            return 10
        for attr in order_info.__slots__:
            assert getattr(order_info, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(order_info)) == len(set(mro_slots(order_info))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            while True:
                i = 10
        json_dict = {'name': self.name, 'phone_number': self.phone_number, 'email': self.email, 'shipping_address': self.shipping_address.to_dict()}
        order_info = OrderInfo.de_json(json_dict, bot)
        assert order_info.api_kwargs == {}
        assert order_info.name == self.name
        assert order_info.phone_number == self.phone_number
        assert order_info.email == self.email
        assert order_info.shipping_address == self.shipping_address

    def test_to_dict(self, order_info):
        if False:
            for i in range(10):
                print('nop')
        order_info_dict = order_info.to_dict()
        assert isinstance(order_info_dict, dict)
        assert order_info_dict['name'] == order_info.name
        assert order_info_dict['phone_number'] == order_info.phone_number
        assert order_info_dict['email'] == order_info.email
        assert order_info_dict['shipping_address'] == order_info.shipping_address.to_dict()

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        a = OrderInfo('name', 'number', 'mail', ShippingAddress('GB', '', 'London', '12 Grimmauld Place', '', 'WC1'))
        b = OrderInfo('name', 'number', 'mail', ShippingAddress('GB', '', 'London', '12 Grimmauld Place', '', 'WC1'))
        c = OrderInfo('name', 'number', 'mail', ShippingAddress('GB', '', 'London', '13 Grimmauld Place', '', 'WC1'))
        d = OrderInfo('name', 'number', 'e-mail', ShippingAddress('GB', '', 'London', '12 Grimmauld Place', '', 'WC1'))
        e = ShippingAddress('GB', '', 'London', '12 Grimmauld Place', '', 'WC1')
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)