import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InlineQueryResultAudio, InputTextMessageContent
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def inline_query_result_article():
    if False:
        print('Hello World!')
    return InlineQueryResultArticle(TestInlineQueryResultArticleBase.id_, TestInlineQueryResultArticleBase.title, input_message_content=TestInlineQueryResultArticleBase.input_message_content, reply_markup=TestInlineQueryResultArticleBase.reply_markup, url=TestInlineQueryResultArticleBase.url, hide_url=TestInlineQueryResultArticleBase.hide_url, description=TestInlineQueryResultArticleBase.description, thumbnail_url=TestInlineQueryResultArticleBase.thumbnail_url, thumbnail_height=TestInlineQueryResultArticleBase.thumbnail_height, thumbnail_width=TestInlineQueryResultArticleBase.thumbnail_width)

class TestInlineQueryResultArticleBase:
    id_ = 'id'
    type_ = 'article'
    title = 'title'
    input_message_content = InputTextMessageContent('input_message_content')
    reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton('reply_markup')]])
    url = 'url'
    hide_url = True
    description = 'description'
    thumbnail_url = 'thumb url'
    thumbnail_height = 10
    thumbnail_width = 15

class TestInlineQueryResultArticleWithoutRequest(TestInlineQueryResultArticleBase):

    def test_slot_behaviour(self, inline_query_result_article):
        if False:
            print('Hello World!')
        inst = inline_query_result_article
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, inline_query_result_article):
        if False:
            return 10
        assert inline_query_result_article.type == self.type_
        assert inline_query_result_article.id == self.id_
        assert inline_query_result_article.title == self.title
        assert inline_query_result_article.input_message_content.to_dict() == self.input_message_content.to_dict()
        assert inline_query_result_article.reply_markup.to_dict() == self.reply_markup.to_dict()
        assert inline_query_result_article.url == self.url
        assert inline_query_result_article.hide_url == self.hide_url
        assert inline_query_result_article.description == self.description
        assert inline_query_result_article.thumbnail_url == self.thumbnail_url
        assert inline_query_result_article.thumbnail_height == self.thumbnail_height
        assert inline_query_result_article.thumbnail_width == self.thumbnail_width

    def test_to_dict(self, inline_query_result_article):
        if False:
            for i in range(10):
                print('nop')
        inline_query_result_article_dict = inline_query_result_article.to_dict()
        assert isinstance(inline_query_result_article_dict, dict)
        assert inline_query_result_article_dict['type'] == inline_query_result_article.type
        assert inline_query_result_article_dict['id'] == inline_query_result_article.id
        assert inline_query_result_article_dict['title'] == inline_query_result_article.title
        assert inline_query_result_article_dict['input_message_content'] == inline_query_result_article.input_message_content.to_dict()
        assert inline_query_result_article_dict['reply_markup'] == inline_query_result_article.reply_markup.to_dict()
        assert inline_query_result_article_dict['url'] == inline_query_result_article.url
        assert inline_query_result_article_dict['hide_url'] == inline_query_result_article.hide_url
        assert inline_query_result_article_dict['description'] == inline_query_result_article.description
        assert inline_query_result_article_dict['thumbnail_url'] == inline_query_result_article.thumbnail_url
        assert inline_query_result_article_dict['thumbnail_height'] == inline_query_result_article.thumbnail_height
        assert inline_query_result_article_dict['thumbnail_width'] == inline_query_result_article.thumbnail_width

    def test_equality(self):
        if False:
            print('Hello World!')
        a = InlineQueryResultArticle(self.id_, self.title, self.input_message_content)
        b = InlineQueryResultArticle(self.id_, self.title, self.input_message_content)
        c = InlineQueryResultArticle(self.id_, '', self.input_message_content)
        d = InlineQueryResultArticle('', self.title, self.input_message_content)
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