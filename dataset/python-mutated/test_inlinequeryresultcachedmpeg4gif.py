import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultCachedMpeg4Gif, InlineQueryResultCachedVoice, InputTextMessageContent, MessageEntity
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def inline_query_result_cached_mpeg4_gif():
    if False:
        i = 10
        return i + 15
    return InlineQueryResultCachedMpeg4Gif(TestInlineQueryResultCachedMpeg4GifBase.id_, TestInlineQueryResultCachedMpeg4GifBase.mpeg4_file_id, title=TestInlineQueryResultCachedMpeg4GifBase.title, caption=TestInlineQueryResultCachedMpeg4GifBase.caption, parse_mode=TestInlineQueryResultCachedMpeg4GifBase.parse_mode, caption_entities=TestInlineQueryResultCachedMpeg4GifBase.caption_entities, input_message_content=TestInlineQueryResultCachedMpeg4GifBase.input_message_content, reply_markup=TestInlineQueryResultCachedMpeg4GifBase.reply_markup)

class TestInlineQueryResultCachedMpeg4GifBase:
    id_ = 'id'
    type_ = 'mpeg4_gif'
    mpeg4_file_id = 'mpeg4 file id'
    title = 'title'
    caption = 'caption'
    parse_mode = 'Markdown'
    caption_entities = [MessageEntity(MessageEntity.ITALIC, 0, 7)]
    input_message_content = InputTextMessageContent('input_message_content')
    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton('reply_markup')]])

class TestInlineQueryResultCachedMpeg4GifWithoutRequest(TestInlineQueryResultCachedMpeg4GifBase):

    def test_slot_behaviour(self, inline_query_result_cached_mpeg4_gif):
        if False:
            for i in range(10):
                print('nop')
        inst = inline_query_result_cached_mpeg4_gif
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, inline_query_result_cached_mpeg4_gif):
        if False:
            return 10
        assert inline_query_result_cached_mpeg4_gif.type == self.type_
        assert inline_query_result_cached_mpeg4_gif.id == self.id_
        assert inline_query_result_cached_mpeg4_gif.mpeg4_file_id == self.mpeg4_file_id
        assert inline_query_result_cached_mpeg4_gif.title == self.title
        assert inline_query_result_cached_mpeg4_gif.caption == self.caption
        assert inline_query_result_cached_mpeg4_gif.parse_mode == self.parse_mode
        assert inline_query_result_cached_mpeg4_gif.caption_entities == tuple(self.caption_entities)
        assert inline_query_result_cached_mpeg4_gif.input_message_content.to_dict() == self.input_message_content.to_dict()
        assert inline_query_result_cached_mpeg4_gif.reply_markup.to_dict() == self.reply_markup.to_dict()

    def test_caption_entities_always_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        result = InlineQueryResultCachedMpeg4Gif(self.id_, self.mpeg4_file_id)
        assert result.caption_entities == ()

    def test_to_dict(self, inline_query_result_cached_mpeg4_gif):
        if False:
            i = 10
            return i + 15
        inline_query_result_cached_mpeg4_gif_dict = inline_query_result_cached_mpeg4_gif.to_dict()
        assert isinstance(inline_query_result_cached_mpeg4_gif_dict, dict)
        assert inline_query_result_cached_mpeg4_gif_dict['type'] == inline_query_result_cached_mpeg4_gif.type
        assert inline_query_result_cached_mpeg4_gif_dict['id'] == inline_query_result_cached_mpeg4_gif.id
        assert inline_query_result_cached_mpeg4_gif_dict['mpeg4_file_id'] == inline_query_result_cached_mpeg4_gif.mpeg4_file_id
        assert inline_query_result_cached_mpeg4_gif_dict['title'] == inline_query_result_cached_mpeg4_gif.title
        assert inline_query_result_cached_mpeg4_gif_dict['caption'] == inline_query_result_cached_mpeg4_gif.caption
        assert inline_query_result_cached_mpeg4_gif_dict['parse_mode'] == inline_query_result_cached_mpeg4_gif.parse_mode
        assert inline_query_result_cached_mpeg4_gif_dict['caption_entities'] == [ce.to_dict() for ce in inline_query_result_cached_mpeg4_gif.caption_entities]
        assert inline_query_result_cached_mpeg4_gif_dict['input_message_content'] == inline_query_result_cached_mpeg4_gif.input_message_content.to_dict()
        assert inline_query_result_cached_mpeg4_gif_dict['reply_markup'] == inline_query_result_cached_mpeg4_gif.reply_markup.to_dict()

    def test_equality(self):
        if False:
            print('Hello World!')
        a = InlineQueryResultCachedMpeg4Gif(self.id_, self.mpeg4_file_id)
        b = InlineQueryResultCachedMpeg4Gif(self.id_, self.mpeg4_file_id)
        c = InlineQueryResultCachedMpeg4Gif(self.id_, '')
        d = InlineQueryResultCachedMpeg4Gif('', self.mpeg4_file_id)
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