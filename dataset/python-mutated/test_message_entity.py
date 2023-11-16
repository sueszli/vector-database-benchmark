from aiogram.types import MessageEntity

class TestMessageEntity:

    def test_extract_from(self):
        if False:
            while True:
                i = 10
        entity = MessageEntity(type='hashtag', length=4, offset=5)
        assert entity.extract_from('#foo #bar #baz') == '#bar'