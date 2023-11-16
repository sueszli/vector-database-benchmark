import pytest
from telegram import InputTextMessageContent, MessageEntity
from telegram.constants import ParseMode
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def input_text_message_content():
    if False:
        i = 10
        return i + 15
    return InputTextMessageContent(TestInputTextMessageContentBase.message_text, parse_mode=TestInputTextMessageContentBase.parse_mode, entities=TestInputTextMessageContentBase.entities, disable_web_page_preview=TestInputTextMessageContentBase.disable_web_page_preview)

class TestInputTextMessageContentBase:
    message_text = '*message text*'
    parse_mode = ParseMode.MARKDOWN
    entities = [MessageEntity(MessageEntity.ITALIC, 0, 7)]
    disable_web_page_preview = True

class TestInputTextMessageContentWithoutRequest(TestInputTextMessageContentBase):

    def test_slot_behaviour(self, input_text_message_content):
        if False:
            i = 10
            return i + 15
        inst = input_text_message_content
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, input_text_message_content):
        if False:
            return 10
        assert input_text_message_content.parse_mode == self.parse_mode
        assert input_text_message_content.message_text == self.message_text
        assert input_text_message_content.disable_web_page_preview == self.disable_web_page_preview
        assert input_text_message_content.entities == tuple(self.entities)

    def test_entities_always_tuple(self):
        if False:
            return 10
        input_text_message_content = InputTextMessageContent('text')
        assert input_text_message_content.entities == ()

    def test_to_dict(self, input_text_message_content):
        if False:
            while True:
                i = 10
        input_text_message_content_dict = input_text_message_content.to_dict()
        assert isinstance(input_text_message_content_dict, dict)
        assert input_text_message_content_dict['message_text'] == input_text_message_content.message_text
        assert input_text_message_content_dict['parse_mode'] == input_text_message_content.parse_mode
        assert input_text_message_content_dict['entities'] == [ce.to_dict() for ce in input_text_message_content.entities]
        assert input_text_message_content_dict['disable_web_page_preview'] == input_text_message_content.disable_web_page_preview

    def test_equality(self):
        if False:
            return 10
        a = InputTextMessageContent('text')
        b = InputTextMessageContent('text', parse_mode=ParseMode.HTML)
        c = InputTextMessageContent('label')
        d = ParseMode.HTML
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)