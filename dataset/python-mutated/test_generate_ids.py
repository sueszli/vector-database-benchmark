"""Tests for bzrlib/generate_ids.py"""
from bzrlib import generate_ids, tests

class TestFileIds(tests.TestCase):
    """Test functions which generate file ids"""

    def assertGenFileId(self, regex, filename):
        if False:
            return 10
        'gen_file_id should create a file id matching the regex.\n\n        The file id should be ascii, and should be an 8-bit string\n        '
        file_id = generate_ids.gen_file_id(filename)
        self.assertContainsRe(file_id, '^' + regex + '$')
        self.assertIsInstance(file_id, str)
        file_id.decode('ascii')

    def test_gen_file_id(self):
        if False:
            return 10
        gen_file_id = generate_ids.gen_file_id
        self.assertStartsWith(gen_file_id('bar'), 'bar-')
        self.assertStartsWith(gen_file_id('Mwoo oof\t m'), 'mwoooofm-')
        self.assertStartsWith(gen_file_id('..gam.py'), 'gam.py-')
        self.assertStartsWith(gen_file_id('..Mwoo oof\t m'), 'mwoooofm-')
        self.assertStartsWith(gen_file_id(u'åµ.txt'), 'txt-')
        fid = gen_file_id('A' * 50 + '.txt')
        self.assertStartsWith(fid, 'a' * 20 + '-')
        self.assertTrue(len(fid) < 60)
        fid = gen_file_id('åµ..aBcd\tefGhijKLMnop\tqrstuvwxyz')
        self.assertStartsWith(fid, 'abcdefghijklmnopqrst-')
        self.assertTrue(len(fid) < 60)

    def test_file_ids_are_ascii(self):
        if False:
            print('Hello World!')
        tail = '-\\d{14}-[a-z0-9]{16}-\\d+'
        self.assertGenFileId('foo' + tail, 'foo')
        self.assertGenFileId('foo' + tail, u'foo')
        self.assertGenFileId('bar' + tail, u'bar')
        self.assertGenFileId('br' + tail, u'bår')

    def test__next_id_suffix_sets_suffix(self):
        if False:
            print('Hello World!')
        generate_ids._gen_file_id_suffix = None
        generate_ids._next_id_suffix()
        self.assertNotEqual(None, generate_ids._gen_file_id_suffix)

    def test__next_id_suffix_increments(self):
        if False:
            while True:
                i = 10
        generate_ids._gen_file_id_suffix = 'foo-'
        generate_ids._gen_file_id_serial = 1
        try:
            self.assertEqual('foo-2', generate_ids._next_id_suffix())
            self.assertEqual('foo-3', generate_ids._next_id_suffix())
            self.assertEqual('foo-4', generate_ids._next_id_suffix())
            self.assertEqual('foo-5', generate_ids._next_id_suffix())
            self.assertEqual('foo-6', generate_ids._next_id_suffix())
            self.assertEqual('foo-7', generate_ids._next_id_suffix())
            self.assertEqual('foo-8', generate_ids._next_id_suffix())
            self.assertEqual('foo-9', generate_ids._next_id_suffix())
            self.assertEqual('foo-10', generate_ids._next_id_suffix())
        finally:
            generate_ids._gen_file_id_suffix = None
            generate_ids._gen_file_id_serial = 0

    def test_gen_root_id(self):
        if False:
            i = 10
            return i + 15
        root_id = generate_ids.gen_root_id()
        self.assertStartsWith(root_id, 'tree_root-')

class TestGenRevisionId(tests.TestCase):
    """Test generating revision ids"""

    def assertGenRevisionId(self, regex, username, timestamp=None):
        if False:
            for i in range(10):
                print('nop')
        'gen_revision_id should create a revision id matching the regex'
        revision_id = generate_ids.gen_revision_id(username, timestamp)
        self.assertContainsRe(revision_id, '^' + regex + '$')
        self.assertIsInstance(revision_id, str)
        revision_id.decode('ascii')

    def test_timestamp(self):
        if False:
            i = 10
            return i + 15
        'passing a timestamp should cause it to be used'
        self.assertGenRevisionId('user@host-\\d{14}-[a-z0-9]{16}', 'user@host')
        self.assertGenRevisionId('user@host-20061102205056-[a-z0-9]{16}', 'user@host', 1162500656.688)
        self.assertGenRevisionId('user@host-20061102205024-[a-z0-9]{16}', 'user@host', 1162500624.0)

    def test_gen_revision_id_email(self):
        if False:
            for i in range(10):
                print('nop')
        'gen_revision_id uses email address if present'
        regex = 'user\\+joe_bar@foo-bar\\.com-\\d{14}-[a-z0-9]{16}'
        self.assertGenRevisionId(regex, 'user+joe_bar@foo-bar.com')
        self.assertGenRevisionId(regex, '<user+joe_bar@foo-bar.com>')
        self.assertGenRevisionId(regex, 'Joe Bar <user+joe_bar@foo-bar.com>')
        self.assertGenRevisionId(regex, 'Joe Bar <user+Joe_Bar@Foo-Bar.com>')
        self.assertGenRevisionId(regex, u'Joe Bår <user+Joe_Bar@Foo-Bar.com>')

    def test_gen_revision_id_user(self):
        if False:
            for i in range(10):
                print('nop')
        'If there is no email, fall back to the whole username'
        tail = '-\\d{14}-[a-z0-9]{16}'
        self.assertGenRevisionId('joe_bar' + tail, 'Joe Bar')
        self.assertGenRevisionId('joebar' + tail, 'joebar')
        self.assertGenRevisionId('joe_br' + tail, u'Joe Bår')
        self.assertGenRevisionId('joe_br_user\\+joe_bar_foo-bar.com' + tail, u'Joe Bår <user+Joe_Bar_Foo-Bar.com>')

    def test_revision_ids_are_ascii(self):
        if False:
            for i in range(10):
                print('nop')
        'gen_revision_id should always return an ascii revision id.'
        tail = '-\\d{14}-[a-z0-9]{16}'
        self.assertGenRevisionId('joe_bar' + tail, 'Joe Bar')
        self.assertGenRevisionId('joe_bar' + tail, u'Joe Bar')
        self.assertGenRevisionId('joe@foo' + tail, u'Joe Bar <joe@foo>')
        self.assertGenRevisionId('joe@f' + tail, u'Joe Bar <joe@f¶>')