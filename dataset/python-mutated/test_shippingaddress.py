import pytest
from telegram import ShippingAddress
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def shipping_address():
    if False:
        print('Hello World!')
    return ShippingAddress(TestShippingAddressBase.country_code, TestShippingAddressBase.state, TestShippingAddressBase.city, TestShippingAddressBase.street_line1, TestShippingAddressBase.street_line2, TestShippingAddressBase.post_code)

class TestShippingAddressBase:
    country_code = 'GB'
    state = 'state'
    city = 'London'
    street_line1 = '12 Grimmauld Place'
    street_line2 = 'street_line2'
    post_code = 'WC1'

class TestShippingAddressWithoutRequest(TestShippingAddressBase):

    def test_slot_behaviour(self, shipping_address):
        if False:
            i = 10
            return i + 15
        inst = shipping_address
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            print('Hello World!')
        json_dict = {'country_code': self.country_code, 'state': self.state, 'city': self.city, 'street_line1': self.street_line1, 'street_line2': self.street_line2, 'post_code': self.post_code}
        shipping_address = ShippingAddress.de_json(json_dict, bot)
        assert shipping_address.api_kwargs == {}
        assert shipping_address.country_code == self.country_code
        assert shipping_address.state == self.state
        assert shipping_address.city == self.city
        assert shipping_address.street_line1 == self.street_line1
        assert shipping_address.street_line2 == self.street_line2
        assert shipping_address.post_code == self.post_code

    def test_to_dict(self, shipping_address):
        if False:
            i = 10
            return i + 15
        shipping_address_dict = shipping_address.to_dict()
        assert isinstance(shipping_address_dict, dict)
        assert shipping_address_dict['country_code'] == shipping_address.country_code
        assert shipping_address_dict['state'] == shipping_address.state
        assert shipping_address_dict['city'] == shipping_address.city
        assert shipping_address_dict['street_line1'] == shipping_address.street_line1
        assert shipping_address_dict['street_line2'] == shipping_address.street_line2
        assert shipping_address_dict['post_code'] == shipping_address.post_code

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = ShippingAddress(self.country_code, self.state, self.city, self.street_line1, self.street_line2, self.post_code)
        b = ShippingAddress(self.country_code, self.state, self.city, self.street_line1, self.street_line2, self.post_code)
        d = ShippingAddress('', self.state, self.city, self.street_line1, self.street_line2, self.post_code)
        d2 = ShippingAddress(self.country_code, '', self.city, self.street_line1, self.street_line2, self.post_code)
        d3 = ShippingAddress(self.country_code, self.state, '', self.street_line1, self.street_line2, self.post_code)
        d4 = ShippingAddress(self.country_code, self.state, self.city, '', self.street_line2, self.post_code)
        d5 = ShippingAddress(self.country_code, self.state, self.city, self.street_line1, '', self.post_code)
        d6 = ShippingAddress(self.country_code, self.state, self.city, self.street_line1, self.street_line2, '')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != d
        assert hash(a) != hash(d)
        assert a != d2
        assert hash(a) != hash(d2)
        assert a != d3
        assert hash(a) != hash(d3)
        assert a != d4
        assert hash(a) != hash(d4)
        assert a != d5
        assert hash(a) != hash(d5)
        assert a != d6
        assert hash(6) != hash(d6)