from unittest import TestCase
import simplejson
from simplejson.compat import text_type

class WonkyTextSubclass(text_type):

    def __getslice__(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__('not what you wanted!')

class TestStrSubclass(TestCase):

    def test_dump_load(self):
        if False:
            return 10
        for s in ['', '"hello"', 'text', u'\\']:
            self.assertEqual(s, simplejson.loads(simplejson.dumps(WonkyTextSubclass(s))))
            self.assertEqual(s, simplejson.loads(simplejson.dumps(WonkyTextSubclass(s), ensure_ascii=False)))