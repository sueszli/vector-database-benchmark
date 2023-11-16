import pytest
from praw.models.reddit.mixins import ThingModerationMixin
from .... import UnitTest

class TestThingModerationMixin(UnitTest):

    def test_must_be_extended(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(NotImplementedError):
            ThingModerationMixin().send_removal_message(message='public', title='title', type='message')