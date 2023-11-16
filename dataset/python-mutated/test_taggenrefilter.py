from test.picardtestcase import PicardTestCase
from picard.track import TagGenreFilter

class TagGenreFilterTest(PicardTestCase):

    def test_no_filter(self):
        if False:
            while True:
                i = 10
        tag_filter = TagGenreFilter('# comment')
        self.assertFalse(tag_filter.skip('jazz'))

    def test_strict_filter(self):
        if False:
            print('Hello World!')
        tag_filter = TagGenreFilter('-jazz')
        self.assertTrue(tag_filter.skip('jazz'))

    def test_strict_filter_allowlist(self):
        if False:
            i = 10
            return i + 15
        filters = '\n            +jazz\n            -jazz\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertFalse(tag_filter.skip('jazz'))

    def test_strict_filter_allowlist_reverseorder(self):
        if False:
            return 10
        filters = '\n            -jazz\n            +jazz\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertFalse(tag_filter.skip('jazz'))

    def test_wildcard_filter_all_but(self):
        if False:
            return 10
        filters = '\n            -*\n            +blues\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertTrue(tag_filter.skip('jazz'))
        self.assertTrue(tag_filter.skip('rock'))
        self.assertFalse(tag_filter.skip('blues'))

    def test_wildcard_filter(self):
        if False:
            print('Hello World!')
        filters = '\n            -jazz*\n            -*rock\n            -*disco*\n            -a*b\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertTrue(tag_filter.skip('jazz'))
        self.assertTrue(tag_filter.skip('jazz blues'))
        self.assertFalse(tag_filter.skip('blues jazz'))
        self.assertTrue(tag_filter.skip('rock'))
        self.assertTrue(tag_filter.skip('blues rock'))
        self.assertFalse(tag_filter.skip('rock blues'))
        self.assertTrue(tag_filter.skip('disco'))
        self.assertTrue(tag_filter.skip('xdisco'))
        self.assertTrue(tag_filter.skip('discox'))
        self.assertTrue(tag_filter.skip('ab'))
        self.assertTrue(tag_filter.skip('axb'))
        self.assertTrue(tag_filter.skip('axxb'))
        self.assertFalse(tag_filter.skip('xab'))

    def test_regex_filter(self):
        if False:
            print('Hello World!')
        filters = '\n            -/^j.zz/\n            -/r[io]ck$/\n            -/disco+/\n            +/discoooo/\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertTrue(tag_filter.skip('jazz'))
        self.assertTrue(tag_filter.skip('jizz'))
        self.assertTrue(tag_filter.skip('jazz blues'))
        self.assertFalse(tag_filter.skip('blues jazz'))
        self.assertTrue(tag_filter.skip('rock'))
        self.assertTrue(tag_filter.skip('blues rock'))
        self.assertTrue(tag_filter.skip('blues rick'))
        self.assertFalse(tag_filter.skip('rock blues'))
        self.assertTrue(tag_filter.skip('disco'))
        self.assertTrue(tag_filter.skip('xdiscox'))
        self.assertTrue(tag_filter.skip('xdiscooox'))
        self.assertFalse(tag_filter.skip('xdiscoooox'))

    def test_regex_filter_keep_all(self):
        if False:
            while True:
                i = 10
        filters = '\n            -/^j.zz/\n            -/r[io]ck$/\n            -/disco+/\n            +/discoooo/\n            +/.*/\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertFalse(tag_filter.skip('jazz'))
        self.assertFalse(tag_filter.skip('jizz'))
        self.assertFalse(tag_filter.skip('jazz blues'))
        self.assertFalse(tag_filter.skip('blues jazz'))
        self.assertFalse(tag_filter.skip('rock'))
        self.assertFalse(tag_filter.skip('blues rock'))
        self.assertFalse(tag_filter.skip('blues rick'))
        self.assertFalse(tag_filter.skip('rock blues'))
        self.assertFalse(tag_filter.skip('disco'))
        self.assertFalse(tag_filter.skip('xdiscox'))
        self.assertFalse(tag_filter.skip('xdiscooox'))
        self.assertFalse(tag_filter.skip('xdiscoooox'))

    def test_uppercased_filter(self):
        if False:
            for i in range(10):
                print('nop')
        filters = '\n            -JAZZ*\n            -ROCK\n            -/^DISCO$/\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertTrue(tag_filter.skip('jazz blues'))
        self.assertTrue(tag_filter.skip('JAZZ BLUES'))
        self.assertTrue(tag_filter.skip('rock'))
        self.assertTrue(tag_filter.skip('ROCK'))
        self.assertTrue(tag_filter.skip('disco'))
        self.assertTrue(tag_filter.skip('DISCO'))

    def test_whitespaces_filter(self):
        if False:
            for i in range(10):
                print('nop')
        filters = '\n            - jazz b*\n            - * ro ck\n            - /^di sco$/\n        '
        tag_filter = TagGenreFilter(filters)
        self.assertTrue(tag_filter.skip('jazz blues'))
        self.assertTrue(tag_filter.skip('blues ro ck'))
        self.assertTrue(tag_filter.skip('di sco'))
        self.assertFalse(tag_filter.skip('bluesro ck'))

    def test_filter_method(self):
        if False:
            for i in range(10):
                print('nop')
        tag_filter = TagGenreFilter('-a*')
        result = list(tag_filter.filter([('ax', 1), ('bx', 2), ('ay', 3), ('by', 4)]))
        self.assertEqual([('bx', 2), ('by', 4)], result)