import pytest
from telegram import LabeledPrice, Location
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def labeled_price():
    if False:
        for i in range(10):
            print('nop')
    return LabeledPrice(TestLabeledPriceBase.label, TestLabeledPriceBase.amount)

class TestLabeledPriceBase:
    label = 'label'
    amount = 100

class TestLabeledPriceWithoutRequest(TestLabeledPriceBase):

    def test_slot_behaviour(self, labeled_price):
        if False:
            while True:
                i = 10
        inst = labeled_price
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, labeled_price):
        if False:
            print('Hello World!')
        assert labeled_price.label == self.label
        assert labeled_price.amount == self.amount

    def test_to_dict(self, labeled_price):
        if False:
            i = 10
            return i + 15
        labeled_price_dict = labeled_price.to_dict()
        assert isinstance(labeled_price_dict, dict)
        assert labeled_price_dict['label'] == labeled_price.label
        assert labeled_price_dict['amount'] == labeled_price.amount

    def test_equality(self):
        if False:
            print('Hello World!')
        a = LabeledPrice('label', 100)
        b = LabeledPrice('label', 100)
        c = LabeledPrice('Label', 101)
        d = Location(123, 456)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)