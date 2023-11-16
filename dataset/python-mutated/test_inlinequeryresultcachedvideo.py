import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultCachedVideo, InlineQueryResultCachedVoice, InputTextMessageContent, MessageEntity
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def inline_query_result_cached_video():
    if False:
        for i in range(10):
            print('nop')
    return InlineQueryResultCachedVideo(TestInlineQueryResultCachedVideoBase.id_, TestInlineQueryResultCachedVideoBase.video_file_id, TestInlineQueryResultCachedVideoBase.title, caption=TestInlineQueryResultCachedVideoBase.caption, parse_mode=TestInlineQueryResultCachedVideoBase.parse_mode, caption_entities=TestInlineQueryResultCachedVideoBase.caption_entities, description=TestInlineQueryResultCachedVideoBase.description, input_message_content=TestInlineQueryResultCachedVideoBase.input_message_content, reply_markup=TestInlineQueryResultCachedVideoBase.reply_markup)

class TestInlineQueryResultCachedVideoBase:
    id_ = 'id'
    type_ = 'video'
    video_file_id = 'video file id'
    title = 'title'
    caption = 'caption'
    parse_mode = 'Markdown'
    caption_entities = [MessageEntity(MessageEntity.ITALIC, 0, 7)]
    description = 'description'
    input_message_content = InputTextMessageContent('input_message_content')
    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton('reply_markup')]])

class TestInlineQueryResultCachedVideoWithoutRequest(TestInlineQueryResultCachedVideoBase):

    def test_slot_behaviour(self, inline_query_result_cached_video):
        if False:
            return 10
        inst = inline_query_result_cached_video
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, inline_query_result_cached_video):
        if False:
            return 10
        assert inline_query_result_cached_video.type == self.type_
        assert inline_query_result_cached_video.id == self.id_
        assert inline_query_result_cached_video.video_file_id == self.video_file_id
        assert inline_query_result_cached_video.title == self.title
        assert inline_query_result_cached_video.description == self.description
        assert inline_query_result_cached_video.caption == self.caption
        assert inline_query_result_cached_video.parse_mode == self.parse_mode
        assert inline_query_result_cached_video.caption_entities == tuple(self.caption_entities)
        assert inline_query_result_cached_video.input_message_content.to_dict() == self.input_message_content.to_dict()
        assert inline_query_result_cached_video.reply_markup.to_dict() == self.reply_markup.to_dict()

    def test_caption_entities_always_tuple(self):
        if False:
            i = 10
            return i + 15
        video = InlineQueryResultCachedVideo(self.id_, self.video_file_id, self.title)
        assert video.caption_entities == ()

    def test_to_dict(self, inline_query_result_cached_video):
        if False:
            for i in range(10):
                print('nop')
        inline_query_result_cached_video_dict = inline_query_result_cached_video.to_dict()
        assert isinstance(inline_query_result_cached_video_dict, dict)
        assert inline_query_result_cached_video_dict['type'] == inline_query_result_cached_video.type
        assert inline_query_result_cached_video_dict['id'] == inline_query_result_cached_video.id
        assert inline_query_result_cached_video_dict['video_file_id'] == inline_query_result_cached_video.video_file_id
        assert inline_query_result_cached_video_dict['title'] == inline_query_result_cached_video.title
        assert inline_query_result_cached_video_dict['description'] == inline_query_result_cached_video.description
        assert inline_query_result_cached_video_dict['caption'] == inline_query_result_cached_video.caption
        assert inline_query_result_cached_video_dict['parse_mode'] == inline_query_result_cached_video.parse_mode
        assert inline_query_result_cached_video_dict['caption_entities'] == [ce.to_dict() for ce in inline_query_result_cached_video.caption_entities]
        assert inline_query_result_cached_video_dict['input_message_content'] == inline_query_result_cached_video.input_message_content.to_dict()
        assert inline_query_result_cached_video_dict['reply_markup'] == inline_query_result_cached_video.reply_markup.to_dict()

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = InlineQueryResultCachedVideo(self.id_, self.video_file_id, self.title)
        b = InlineQueryResultCachedVideo(self.id_, self.video_file_id, self.title)
        c = InlineQueryResultCachedVideo(self.id_, '', self.title)
        d = InlineQueryResultCachedVideo('', self.video_file_id, self.title)
        e = InlineQueryResultCachedVoice(self.id_, '', '')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)