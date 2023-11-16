import orjson
from .util import read_fixture_obj

class TestDictSortKeys:

    def test_twitter_sorted(self):
        if False:
            i = 10
            return i + 15
        '\n        twitter.json sorted\n        '
        obj = read_fixture_obj('twitter.json.xz')
        assert list(obj.keys()) != sorted(list(obj.keys()))
        serialized = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
        val = orjson.loads(serialized)
        assert list(val.keys()) == sorted(list(val.keys()))

    def test_canada_sorted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        canada.json sorted\n        '
        obj = read_fixture_obj('canada.json.xz')
        assert list(obj.keys()) != sorted(list(obj.keys()))
        serialized = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
        val = orjson.loads(serialized)
        assert list(val.keys()) == sorted(list(val.keys()))

    def test_github_sorted(self):
        if False:
            return 10
        '\n        github.json sorted\n        '
        obj = read_fixture_obj('github.json.xz')
        for each in obj:
            assert list(each.keys()) != sorted(list(each.keys()))
        serialized = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
        val = orjson.loads(serialized)
        for each in val:
            assert list(each.keys()) == sorted(list(each.keys()))

    def test_utf8_sorted(self):
        if False:
            return 10
        '\n        UTF-8 sorted\n        '
        obj = {'a': 1, 'ä': 2, 'A': 3}
        assert list(obj.keys()) != sorted(list(obj.keys()))
        serialized = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
        val = orjson.loads(serialized)
        assert list(val.keys()) == sorted(list(val.keys()))