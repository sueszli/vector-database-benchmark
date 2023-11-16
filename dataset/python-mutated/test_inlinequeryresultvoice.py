import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultAudio, InlineQueryResultVoice, InputTextMessageContent, MessageEntity
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def inline_query_result_voice():
    if False:
        while True:
            i = 10
    return InlineQueryResultVoice(id=TestInlineQueryResultVoiceBase.id_, voice_url=TestInlineQueryResultVoiceBase.voice_url, title=TestInlineQueryResultVoiceBase.title, voice_duration=TestInlineQueryResultVoiceBase.voice_duration, caption=TestInlineQueryResultVoiceBase.caption, parse_mode=TestInlineQueryResultVoiceBase.parse_mode, caption_entities=TestInlineQueryResultVoiceBase.caption_entities, input_message_content=TestInlineQueryResultVoiceBase.input_message_content, reply_markup=TestInlineQueryResultVoiceBase.reply_markup)

class TestInlineQueryResultVoiceBase:
    id_ = 'id'
    type_ = 'voice'
    voice_url = 'voice url'
    title = 'title'
    voice_duration = 'voice_duration'
    caption = 'caption'
    parse_mode = 'HTML'
    caption_entities = [MessageEntity(MessageEntity.ITALIC, 0, 7)]
    input_message_content = InputTextMessageContent('input_message_content')
    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton('reply_markup')]])

class TestInlineQueryResultVoiceWithoutRequest(TestInlineQueryResultVoiceBase):

    def test_slot_behaviour(self, inline_query_result_voice):
        if False:
            for i in range(10):
                print('nop')
        inst = inline_query_result_voice
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, inline_query_result_voice):
        if False:
            for i in range(10):
                print('nop')
        assert inline_query_result_voice.type == self.type_
        assert inline_query_result_voice.id == self.id_
        assert inline_query_result_voice.voice_url == self.voice_url
        assert inline_query_result_voice.title == self.title
        assert inline_query_result_voice.voice_duration == self.voice_duration
        assert inline_query_result_voice.caption == self.caption
        assert inline_query_result_voice.parse_mode == self.parse_mode
        assert inline_query_result_voice.caption_entities == tuple(self.caption_entities)
        assert inline_query_result_voice.input_message_content.to_dict() == self.input_message_content.to_dict()
        assert inline_query_result_voice.reply_markup.to_dict() == self.reply_markup.to_dict()

    def test_caption_entities_always_tuple(self):
        if False:
            print('Hello World!')
        result = InlineQueryResultVoice(self.id_, self.voice_url, self.title)
        assert result.caption_entities == ()

    def test_to_dict(self, inline_query_result_voice):
        if False:
            for i in range(10):
                print('nop')
        inline_query_result_voice_dict = inline_query_result_voice.to_dict()
        assert isinstance(inline_query_result_voice_dict, dict)
        assert inline_query_result_voice_dict['type'] == inline_query_result_voice.type
        assert inline_query_result_voice_dict['id'] == inline_query_result_voice.id
        assert inline_query_result_voice_dict['voice_url'] == inline_query_result_voice.voice_url
        assert inline_query_result_voice_dict['title'] == inline_query_result_voice.title
        assert inline_query_result_voice_dict['voice_duration'] == inline_query_result_voice.voice_duration
        assert inline_query_result_voice_dict['caption'] == inline_query_result_voice.caption
        assert inline_query_result_voice_dict['parse_mode'] == inline_query_result_voice.parse_mode
        assert inline_query_result_voice_dict['caption_entities'] == [ce.to_dict() for ce in inline_query_result_voice.caption_entities]
        assert inline_query_result_voice_dict['input_message_content'] == inline_query_result_voice.input_message_content.to_dict()
        assert inline_query_result_voice_dict['reply_markup'] == inline_query_result_voice.reply_markup.to_dict()

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = InlineQueryResultVoice(self.id_, self.voice_url, self.title)
        b = InlineQueryResultVoice(self.id_, self.voice_url, self.title)
        c = InlineQueryResultVoice(self.id_, '', self.title)
        d = InlineQueryResultVoice('', self.voice_url, self.title)
        e = InlineQueryResultAudio(self.id_, '', '')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)