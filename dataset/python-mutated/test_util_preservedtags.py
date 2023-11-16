from test.picardtestcase import PicardTestCase
from picard import config
from picard.util.preservedtags import PreservedTags

class PreservedTagsTest(PicardTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        config.setting[PreservedTags.opt_name] = ['tag1', 'tag2']

    def test_load_and_contains(self):
        if False:
            while True:
                i = 10
        preserved = PreservedTags()
        self.assertIn('tag1', preserved)
        self.assertIn('tag2', preserved)
        self.assertIn('TAG1', preserved)
        self.assertIn(' tag1', preserved)

    def test_add(self):
        if False:
            print('Hello World!')
        preserved = PreservedTags()
        self.assertNotIn('tag3', preserved)
        preserved.add('tag3')
        self.assertIn('tag3', preserved)
        self.assertIn('tag3', PreservedTags())

    def test_add_case_insensitive(self):
        if False:
            for i in range(10):
                print('nop')
        preserved = PreservedTags()
        self.assertNotIn('tag3', preserved)
        preserved.add('TAG3')
        self.assertIn('tag3', preserved)

    def test_discard(self):
        if False:
            while True:
                i = 10
        preserved = PreservedTags()
        self.assertIn('tag1', preserved)
        preserved.discard('tag1')
        self.assertNotIn('tag1', preserved)
        self.assertNotIn('tag1', PreservedTags())

    def test_discard_case_insensitive(self):
        if False:
            while True:
                i = 10
        preserved = PreservedTags()
        self.assertIn('tag1', preserved)
        preserved.discard('TAG1')
        self.assertNotIn('tag1', preserved)

    def test_order(self):
        if False:
            for i in range(10):
                print('nop')
        preserved = PreservedTags()
        preserved.add('tag3')
        preserved.add('tag2')
        preserved.add('tag1')
        preserved.discard('tag2')
        self.assertEqual(config.setting[PreservedTags.opt_name], ['tag1', 'tag3'])