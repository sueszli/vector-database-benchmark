import pickle

from praw.models import MoreComments

from ... import UnitTest


class TestComment(UnitTest):
    def test_equality(self, reddit):
        more = MoreComments(reddit, {"children": ["a", "b", "c", "d"], "count": 4})
        more2 = MoreComments(reddit, {"children": ["a", "b", "c", "d"], "count": 4})
        assert more == more2
        assert more != 5

    def test_pickle(self, reddit):
        more = MoreComments(reddit, {"children": ["a", "b"], "count": 4})
        for level in range(pickle.HIGHEST_PROTOCOL + 1):
            other = pickle.loads(pickle.dumps(more, protocol=level))
            assert more == other

    def test_repr(self, reddit):
        more = MoreComments(reddit, {"children": ["a", "b", "c", "d", "e"], "count": 5})
        assert repr(more) == "<MoreComments count=5, children=['a', 'b', 'c', '...']>"

        more = MoreComments(reddit, {"children": ["a", "b", "c", "d"], "count": 4})
        assert repr(more) == "<MoreComments count=4, children=['a', 'b', 'c', 'd']>"
