"""Test praw.models.subreddits."""
import pytest
from .. import UnitTest

class TestSubreddits(UnitTest):

    def test_recommended__invalid_omit_subreddits(self, reddit):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError) as excinfo:
            reddit.subreddits.recommended(['earthporn'], 'invalid')
        assert str(excinfo.value) == 'omit_subreddits must be a list or None'

    def test_recommended__invalid_subreddits(self, reddit):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError) as excinfo:
            reddit.subreddits.recommended('earthporn')
        assert str(excinfo.value) == 'subreddits must be a list'

    def test_search__params_not_modified(self, reddit):
        if False:
            i = 10
            return i + 15
        params = {'dummy': 'value'}
        generator = reddit.subreddits.search(None, params=params)
        assert generator.params['dummy'] == 'value'
        assert params == {'dummy': 'value'}