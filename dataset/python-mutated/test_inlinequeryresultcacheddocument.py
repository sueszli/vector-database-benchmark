import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultCachedDocument, InlineQueryResultCachedVoice, InputTextMessageContent, MessageEntity
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def inline_query_result_cached_document():
    if False:
        i = 10
        return i + 15
    return InlineQueryResultCachedDocument(TestInlineQueryResultCachedDocumentBase.id_, TestInlineQueryResultCachedDocumentBase.title, TestInlineQueryResultCachedDocumentBase.document_file_id, caption=TestInlineQueryResultCachedDocumentBase.caption, parse_mode=TestInlineQueryResultCachedDocumentBase.parse_mode, caption_entities=TestInlineQueryResultCachedDocumentBase.caption_entities, description=TestInlineQueryResultCachedDocumentBase.description, input_message_content=TestInlineQueryResultCachedDocumentBase.input_message_content, reply_markup=TestInlineQueryResultCachedDocumentBase.reply_markup)

class TestInlineQueryResultCachedDocumentBase:
    id_ = 'id'
    type_ = 'document'
    document_file_id = 'document file id'
    title = 'title'
    caption = 'caption'
    parse_mode = 'Markdown'
    caption_entities = [MessageEntity(MessageEntity.ITALIC, 0, 7)]
    description = 'description'
    input_message_content = InputTextMessageContent('input_message_content')
    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton('reply_markup')]])

class TestInlineQueryResultCachedDocumentWithoutRequest(TestInlineQueryResultCachedDocumentBase):

    def test_slot_behaviour(self, inline_query_result_cached_document):
        if False:
            while True:
                i = 10
        inst = inline_query_result_cached_document
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, inline_query_result_cached_document):
        if False:
            print('Hello World!')
        assert inline_query_result_cached_document.id == self.id_
        assert inline_query_result_cached_document.type == self.type_
        assert inline_query_result_cached_document.document_file_id == self.document_file_id
        assert inline_query_result_cached_document.title == self.title
        assert inline_query_result_cached_document.caption == self.caption
        assert inline_query_result_cached_document.parse_mode == self.parse_mode
        assert inline_query_result_cached_document.caption_entities == tuple(self.caption_entities)
        assert inline_query_result_cached_document.description == self.description
        assert inline_query_result_cached_document.input_message_content.to_dict() == self.input_message_content.to_dict()
        assert inline_query_result_cached_document.reply_markup.to_dict() == self.reply_markup.to_dict()

    def test_caption_entities_always_tuple(self):
        if False:
            print('Hello World!')
        test = InlineQueryResultCachedDocument(self.id_, self.title, self.document_file_id)
        assert test.caption_entities == ()

    def test_to_dict(self, inline_query_result_cached_document):
        if False:
            for i in range(10):
                print('nop')
        inline_query_result_cached_document_dict = inline_query_result_cached_document.to_dict()
        assert isinstance(inline_query_result_cached_document_dict, dict)
        assert inline_query_result_cached_document_dict['id'] == inline_query_result_cached_document.id
        assert inline_query_result_cached_document_dict['type'] == inline_query_result_cached_document.type
        assert inline_query_result_cached_document_dict['document_file_id'] == inline_query_result_cached_document.document_file_id
        assert inline_query_result_cached_document_dict['title'] == inline_query_result_cached_document.title
        assert inline_query_result_cached_document_dict['caption'] == inline_query_result_cached_document.caption
        assert inline_query_result_cached_document_dict['parse_mode'] == inline_query_result_cached_document.parse_mode
        assert inline_query_result_cached_document_dict['caption_entities'] == [ce.to_dict() for ce in inline_query_result_cached_document.caption_entities]
        assert inline_query_result_cached_document_dict['description'] == inline_query_result_cached_document.description
        assert inline_query_result_cached_document_dict['input_message_content'] == inline_query_result_cached_document.input_message_content.to_dict()
        assert inline_query_result_cached_document_dict['reply_markup'] == inline_query_result_cached_document.reply_markup.to_dict()

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        a = InlineQueryResultCachedDocument(self.id_, self.title, self.document_file_id)
        b = InlineQueryResultCachedDocument(self.id_, self.title, self.document_file_id)
        c = InlineQueryResultCachedDocument(self.id_, self.title, '')
        d = InlineQueryResultCachedDocument('', self.title, self.document_file_id)
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