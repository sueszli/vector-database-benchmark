import orjson
from .util import read_fixture_obj

class TestAppendNewline:

    def test_dumps_newline(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        dumps() OPT_APPEND_NEWLINE\n        '
        assert orjson.dumps([], option=orjson.OPT_APPEND_NEWLINE) == b'[]\n'

    def test_twitter_newline(self):
        if False:
            print('Hello World!')
        '\n        loads(),dumps() twitter.json OPT_APPEND_NEWLINE\n        '
        val = read_fixture_obj('twitter.json.xz')
        assert orjson.loads(orjson.dumps(val, option=orjson.OPT_APPEND_NEWLINE)) == val

    def test_canada(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        loads(), dumps() canada.json OPT_APPEND_NEWLINE\n        '
        val = read_fixture_obj('canada.json.xz')
        assert orjson.loads(orjson.dumps(val, option=orjson.OPT_APPEND_NEWLINE)) == val

    def test_citm_catalog_newline(self):
        if False:
            while True:
                i = 10
        '\n        loads(), dumps() citm_catalog.json OPT_APPEND_NEWLINE\n        '
        val = read_fixture_obj('citm_catalog.json.xz')
        assert orjson.loads(orjson.dumps(val, option=orjson.OPT_APPEND_NEWLINE)) == val

    def test_github_newline(self):
        if False:
            return 10
        '\n        loads(), dumps() github.json OPT_APPEND_NEWLINE\n        '
        val = read_fixture_obj('github.json.xz')
        assert orjson.loads(orjson.dumps(val, option=orjson.OPT_APPEND_NEWLINE)) == val