import pytest
from pandas import DataFrame
from project_fully_featured.assets.recommender.comment_stories import comment_stories

@pytest.mark.parametrize('comments, stories, expected', [([[2, 1000, 'bob']], [[1000]], [[2, 1000, 'bob']]), ([[2, 1000, 'bob'], [3, 2, 'alice']], [[1000]], [[2, 1000, 'bob'], [3, 1000, 'alice']])])
def test_comment_stories(comments, stories, expected):
    if False:
        for i in range(10):
            print('nop')
    comments = DataFrame(comments, columns=['id', 'parent', 'user_id'])
    stories = DataFrame(stories, columns=['id'])
    result = comment_stories(stories=stories, comments=comments)
    expected = DataFrame(expected, columns=['comment_id', 'story_id', 'commenter_id']).set_index('comment_id')
    assert result.equals(expected)