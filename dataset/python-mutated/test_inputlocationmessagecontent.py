import pytest
from telegram import InputLocationMessageContent, Location
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def input_location_message_content():
    if False:
        print('Hello World!')
    return InputLocationMessageContent(TestInputLocationMessageContentBase.latitude, TestInputLocationMessageContentBase.longitude, live_period=TestInputLocationMessageContentBase.live_period, horizontal_accuracy=TestInputLocationMessageContentBase.horizontal_accuracy, heading=TestInputLocationMessageContentBase.heading, proximity_alert_radius=TestInputLocationMessageContentBase.proximity_alert_radius)

class TestInputLocationMessageContentBase:
    latitude = -23.691288
    longitude = -46.788279
    live_period = 80
    horizontal_accuracy = 50.5
    heading = 90
    proximity_alert_radius = 999

class TestInputLocationMessageContentWithoutRequest(TestInputLocationMessageContentBase):

    def test_slot_behaviour(self, input_location_message_content):
        if False:
            while True:
                i = 10
        inst = input_location_message_content
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, input_location_message_content):
        if False:
            for i in range(10):
                print('nop')
        assert input_location_message_content.longitude == self.longitude
        assert input_location_message_content.latitude == self.latitude
        assert input_location_message_content.live_period == self.live_period
        assert input_location_message_content.horizontal_accuracy == self.horizontal_accuracy
        assert input_location_message_content.heading == self.heading
        assert input_location_message_content.proximity_alert_radius == self.proximity_alert_radius

    def test_to_dict(self, input_location_message_content):
        if False:
            while True:
                i = 10
        input_location_message_content_dict = input_location_message_content.to_dict()
        assert isinstance(input_location_message_content_dict, dict)
        assert input_location_message_content_dict['latitude'] == input_location_message_content.latitude
        assert input_location_message_content_dict['longitude'] == input_location_message_content.longitude
        assert input_location_message_content_dict['live_period'] == input_location_message_content.live_period
        assert input_location_message_content_dict['horizontal_accuracy'] == input_location_message_content.horizontal_accuracy
        assert input_location_message_content_dict['heading'] == input_location_message_content.heading
        assert input_location_message_content_dict['proximity_alert_radius'] == input_location_message_content.proximity_alert_radius

    def test_equality(self):
        if False:
            print('Hello World!')
        a = InputLocationMessageContent(123, 456, 70)
        b = InputLocationMessageContent(123, 456, 90)
        c = InputLocationMessageContent(123, 457, 70)
        d = Location(123, 456)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)