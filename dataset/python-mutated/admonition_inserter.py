import collections.abc
import pytest
import telegram.ext
from docs.auxil.admonition_inserter import AdmonitionInserter

@pytest.fixture(scope='session')
def admonition_inserter():
    if False:
        i = 10
        return i + 15
    return AdmonitionInserter()

class TestAdmonitionInserter:
    """This is a minimal-effort test to ensure that the `AdmonitionInserter`
    used for automatically inserting references in the docs works as expected.

    It does not aim to cover all links in the documentation, but rather checks that several special
    cases (which where discovered during the implementation of `AdmonitionInserter`) are handled
    correctly.
    """

    def test_admonitions_dict(self, admonition_inserter):
        if False:
            i = 10
            return i + 15
        assert len(admonition_inserter.admonitions) == len(admonition_inserter.ALL_ADMONITION_TYPES)
        for admonition_type in admonition_inserter.ALL_ADMONITION_TYPES:
            assert admonition_type in admonition_inserter.admonitions
            assert len(admonition_inserter.admonitions[admonition_type].keys()) > 0
        for admonition_type in admonition_inserter.CLASS_ADMONITION_TYPES:
            for cls in admonition_inserter.admonitions[admonition_type]:
                if 'tests.' in str(cls):
                    continue
                assert isinstance(cls, type)
                assert str(cls).startswith("<class 'telegram."), f'Class {cls} does not belong to Telegram classes. Admonition:\\n{admonition_inserter.admonitions[admonition_type][cls]}'
        for admonition_type in admonition_inserter.METHOD_ADMONITION_TYPES:
            for method in admonition_inserter.admonitions[admonition_type]:
                assert isinstance(method, collections.abc.Callable)
                assert str(method).startswith('<function Bot.'), f'Method {method} does not belong to methods that should get admonitions.Admonition:\n{admonition_inserter.admonitions[admonition_type][method]}'

    @pytest.mark.parametrize(('admonition_type', 'cls', 'link'), [('available_in', telegram.ChatMember, ':attr:`telegram.ChatMemberUpdated.new_chat_member`'), ('available_in', telegram.ChatMemberAdministrator, ':attr:`telegram.ChatMemberUpdated.new_chat_member`'), ('available_in', telegram.Sticker, ':attr:`telegram.StickerSet.stickers`'), ('available_in', telegram.ResidentialAddress, ':attr:`telegram.EncryptedPassportElement.data`'), ('returned_in', telegram.StickerSet, ':meth:`telegram.Bot.get_sticker_set`'), ('returned_in', telegram.ChatMember, ':meth:`telegram.Bot.get_chat_member`'), ('returned_in', telegram.ChatMemberOwner, ':meth:`telegram.Bot.get_chat_member`'), ('returned_in', telegram.Message, ':meth:`telegram.Bot.edit_message_live_location`'), ('returned_in', telegram.ext.Application, ':meth:`telegram.ext.ApplicationBuilder.build`'), ('shortcuts', telegram.Bot.edit_message_caption, ':meth:`telegram.CallbackQuery.edit_message_caption`'), ('use_in', telegram.InlineQueryResult, ':meth:`telegram.Bot.answer_web_app_query`'), ('use_in', telegram.InputMediaPhoto, ':meth:`telegram.Bot.send_media_group`'), ('use_in', telegram.InlineKeyboardMarkup, ':meth:`telegram.Bot.send_message`'), ('use_in', telegram.Sticker, ':meth:`telegram.Bot.get_file`'), ('use_in', telegram.ext.BasePersistence, ':meth:`telegram.ext.ApplicationBuilder.persistence`'), ('use_in', telegram.ext.Defaults, ':meth:`telegram.ext.ApplicationBuilder.defaults`'), ('use_in', telegram.ext.JobQueue, ':meth:`telegram.ext.ApplicationBuilder.job_queue`'), ('use_in', telegram.ext.PicklePersistence, ':meth:`telegram.ext.ApplicationBuilder.persistence`')])
    def test_check_presence(self, admonition_inserter, admonition_type, cls, link):
        if False:
            print('Hello World!')
        'Checks if a given link is present in the admonition of a given type for a given\n        class.\n        '
        admonitions = admonition_inserter.admonitions
        assert cls in admonitions[admonition_type]
        lines_with_link = [line for line in admonitions[admonition_type][cls].splitlines() if line.strip().removeprefix('* ') == link]
        assert lines_with_link, f'Class {cls}, does not have link {link} in a {admonition_type} admonition:\n{admonitions[admonition_type][cls]}'
        assert len(lines_with_link) == 1, f'Class {cls}, must contain only one link {link} in a {admonition_type} admonition:\n{admonitions[admonition_type][cls]}'

    @pytest.mark.parametrize(('admonition_type', 'cls', 'link'), [('returned_in', telegram.ext.CallbackContext, ':meth:`telegram.ext.ApplicationBuilder.build`')])
    def test_check_absence(self, admonition_inserter, admonition_type, cls, link):
        if False:
            print('Hello World!')
        'Checks if a given link is **absent** in the admonition of a given type for a given\n        class.\n\n        If a given class has no admonition of this type at all, the test will also pass.\n        '
        admonitions = admonition_inserter.admonitions
        assert not (cls in admonitions[admonition_type] and [line for line in admonitions[admonition_type][cls].splitlines() if line.strip().removeprefix('* ') == link]), f'Class {cls} is not supposed to have link {link} in a {admonition_type} admonition:\n{admonitions[admonition_type][cls]}'