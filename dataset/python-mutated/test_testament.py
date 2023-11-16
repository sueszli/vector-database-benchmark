"""Test testaments for gpg signing."""
import os
from bzrlib import osutils
from bzrlib.tests import TestCaseWithTransport
from bzrlib.testament import Testament, StrictTestament, StrictTestament3
from bzrlib.transform import TreeTransform
from bzrlib.tests.features import SymlinkFeature

class TestamentSetup(TestCaseWithTransport):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestamentSetup, self).setUp()
        self.wt = self.make_branch_and_tree('.', format='development-subtree')
        self.wt.set_root_id('TREE_ROT')
        b = self.b = self.wt.branch
        b.nick = 'test branch'
        self.wt.commit(message='initial null commit', committer='test@user', timestamp=1129025423, timezone=0, rev_id='test@user-1')
        self.build_tree_contents([('hello', 'contents of hello file'), ('src/',), ('src/foo.c', 'int main()\n{\n}\n')])
        self.wt.add(['hello', 'src', 'src/foo.c'], ['hello-id', 'src-id', 'foo.c-id'])
        tt = TreeTransform(self.wt)
        trans_id = tt.trans_id_tree_path('hello')
        tt.set_executability(True, trans_id)
        tt.apply()
        self.wt.commit(message='add files and directories', timestamp=1129025483, timezone=36000, rev_id='test@user-2', committer='test@user')

class TestamentTests(TestamentSetup):

    def testament_class(self):
        if False:
            return 10
        return Testament

    def expected(self, key):
        if False:
            return 10
        return texts[self.testament_class()][key]

    def from_revision(self, repository, revision_id):
        if False:
            while True:
                i = 10
        return self.testament_class().from_revision(repository, revision_id)

    def test_null_testament(self):
        if False:
            i = 10
            return i + 15
        'Testament for a revision with no contents.'
        t = self.from_revision(self.b.repository, 'test@user-1')
        ass = self.assertTrue
        eq = self.assertEqual
        ass(isinstance(t, Testament))
        eq(t.revision_id, 'test@user-1')
        eq(t.committer, 'test@user')
        eq(t.timestamp, 1129025423)
        eq(t.timezone, 0)

    def test_testment_text_form(self):
        if False:
            i = 10
            return i + 15
        'Conversion of testament to canonical text form.'
        t = self.from_revision(self.b.repository, 'test@user-1')
        text_form = t.as_text()
        self.log('testament text form:\n' + text_form)
        self.assertEqualDiff(text_form, self.expected('rev_1'))
        short_text_form = t.as_short_text()
        self.assertEqualDiff(short_text_form, self.expected('rev_1_short'))

    def test_testament_with_contents(self):
        if False:
            for i in range(10):
                print('nop')
        'Testament containing a file and a directory.'
        t = self.from_revision(self.b.repository, 'test@user-2')
        text_form = t.as_text()
        self.log('testament text form:\n' + text_form)
        self.assertEqualDiff(text_form, self.expected('rev_2'))
        actual_short = t.as_short_text()
        self.assertEqualDiff(actual_short, self.expected('rev_2_short'))

    def test_testament_symlinks(self):
        if False:
            while True:
                i = 10
        'Testament containing symlink (where possible)'
        self.requireFeature(SymlinkFeature)
        os.symlink('wibble/linktarget', 'link')
        self.wt.add(['link'], ['link-id'])
        self.wt.commit(message='add symlink', timestamp=1129025493, timezone=36000, rev_id='test@user-3', committer='test@user')
        t = self.from_revision(self.b.repository, 'test@user-3')
        self.assertEqualDiff(t.as_text(), self.expected('rev_3'))

    def test_testament_revprops(self):
        if False:
            for i in range(10):
                print('nop')
        'Testament to revision with extra properties'
        props = dict(flavor='sour cherry\ncream cheese', size='medium', empty='')
        self.wt.commit(message='revision with properties', timestamp=1129025493, timezone=36000, rev_id='test@user-3', committer='test@user', revprops=props)
        t = self.from_revision(self.b.repository, 'test@user-3')
        self.assertEqualDiff(t.as_text(), self.expected('rev_props'))

    def test_testament_unicode_commit_message(self):
        if False:
            for i in range(10):
                print('nop')
        self.wt.commit(message=u'non-ascii commit © me', timestamp=1129025493, timezone=36000, rev_id='test@user-3', committer=u'Erik Bågfors <test@user>', revprops={'uni': u'µ'})
        t = self.from_revision(self.b.repository, 'test@user-3')
        self.assertEqualDiff(self.expected('sample_unicode').encode('utf-8'), t.as_text())

    def test_from_tree(self):
        if False:
            while True:
                i = 10
        tree = self.b.repository.revision_tree('test@user-2')
        testament = self.testament_class().from_revision_tree(tree)
        text_1 = testament.as_short_text()
        text_2 = self.from_revision(self.b.repository, 'test@user-2').as_short_text()
        self.assertEqual(text_1, text_2)

    def test___init__(self):
        if False:
            return 10
        revision = self.b.repository.get_revision('test@user-2')
        tree = self.b.repository.revision_tree('test@user-2')
        testament_1 = self.testament_class()(revision, tree)
        text_1 = testament_1.as_short_text()
        text_2 = self.from_revision(self.b.repository, 'test@user-2').as_short_text()
        self.assertEqual(text_1, text_2)

class TestamentTestsStrict(TestamentTests):

    def testament_class(self):
        if False:
            print('Hello World!')
        return StrictTestament

class TestamentTestsStrict2(TestamentTests):

    def testament_class(self):
        if False:
            while True:
                i = 10
        return StrictTestament3
REV_1_TESTAMENT = 'bazaar-ng testament version 1\nrevision-id: test@user-1\ncommitter: test@user\ntimestamp: 1129025423\ntimezone: 0\nparents:\nmessage:\n  initial null commit\ninventory:\nproperties:\n  branch-nick:\n    test branch\n'
REV_1_STRICT_TESTAMENT = 'bazaar-ng testament version 2.1\nrevision-id: test@user-1\ncommitter: test@user\ntimestamp: 1129025423\ntimezone: 0\nparents:\nmessage:\n  initial null commit\ninventory:\nproperties:\n  branch-nick:\n    test branch\n'
REV_1_STRICT_TESTAMENT3 = 'bazaar testament version 3 strict\nrevision-id: test@user-1\ncommitter: test@user\ntimestamp: 1129025423\ntimezone: 0\nparents:\nmessage:\n  initial null commit\ninventory:\n  directory . TREE_ROT test@user-1 no\nproperties:\n  branch-nick:\n    test branch\n'
REV_1_SHORT = 'bazaar-ng testament short form 1\nrevision-id: test@user-1\nsha1: %s\n' % osutils.sha_string(REV_1_TESTAMENT)
REV_1_SHORT_STRICT = 'bazaar-ng testament short form 2.1\nrevision-id: test@user-1\nsha1: %s\n' % osutils.sha_string(REV_1_STRICT_TESTAMENT)
REV_1_SHORT_STRICT3 = 'bazaar testament short form 3 strict\nrevision-id: test@user-1\nsha1: %s\n' % osutils.sha_string(REV_1_STRICT_TESTAMENT3)
REV_2_TESTAMENT = 'bazaar-ng testament version 1\nrevision-id: test@user-2\ncommitter: test@user\ntimestamp: 1129025483\ntimezone: 36000\nparents:\n  test@user-1\nmessage:\n  add files and directories\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73\n  directory src src-id\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24\nproperties:\n  branch-nick:\n    test branch\n'
REV_2_STRICT_TESTAMENT = 'bazaar-ng testament version 2.1\nrevision-id: test@user-2\ncommitter: test@user\ntimestamp: 1129025483\ntimezone: 36000\nparents:\n  test@user-1\nmessage:\n  add files and directories\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n'
REV_2_STRICT_TESTAMENT3 = 'bazaar testament version 3 strict\nrevision-id: test@user-2\ncommitter: test@user\ntimestamp: 1129025483\ntimezone: 36000\nparents:\n  test@user-1\nmessage:\n  add files and directories\ninventory:\n  directory . TREE_ROT test@user-1 no\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n'
REV_2_SHORT = 'bazaar-ng testament short form 1\nrevision-id: test@user-2\nsha1: %s\n' % osutils.sha_string(REV_2_TESTAMENT)
REV_2_SHORT_STRICT = 'bazaar-ng testament short form 2.1\nrevision-id: test@user-2\nsha1: %s\n' % osutils.sha_string(REV_2_STRICT_TESTAMENT)
REV_2_SHORT_STRICT3 = 'bazaar testament short form 3 strict\nrevision-id: test@user-2\nsha1: %s\n' % osutils.sha_string(REV_2_STRICT_TESTAMENT3)
REV_PROPS_TESTAMENT = 'bazaar-ng testament version 1\nrevision-id: test@user-3\ncommitter: test@user\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  revision with properties\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73\n  directory src src-id\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24\nproperties:\n  branch-nick:\n    test branch\n  empty:\n  flavor:\n    sour cherry\n    cream cheese\n  size:\n    medium\n'
REV_PROPS_TESTAMENT_STRICT = 'bazaar-ng testament version 2.1\nrevision-id: test@user-3\ncommitter: test@user\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  revision with properties\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n  empty:\n  flavor:\n    sour cherry\n    cream cheese\n  size:\n    medium\n'
REV_PROPS_TESTAMENT_STRICT3 = 'bazaar testament version 3 strict\nrevision-id: test@user-3\ncommitter: test@user\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  revision with properties\ninventory:\n  directory . TREE_ROT test@user-1 no\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n  empty:\n  flavor:\n    sour cherry\n    cream cheese\n  size:\n    medium\n'
REV_3_TESTAMENT = 'bazaar-ng testament version 1\nrevision-id: test@user-3\ncommitter: test@user\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  add symlink\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73\n  symlink link link-id wibble/linktarget\n  directory src src-id\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24\nproperties:\n  branch-nick:\n    test branch\n'
REV_3_TESTAMENT_STRICT = 'bazaar-ng testament version 2.1\nrevision-id: test@user-3\ncommitter: test@user\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  add symlink\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  symlink link link-id wibble/linktarget test@user-3 no\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n'
REV_3_TESTAMENT_STRICT3 = 'bazaar testament version 3 strict\nrevision-id: test@user-3\ncommitter: test@user\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  add symlink\ninventory:\n  directory . TREE_ROT test@user-1 no\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  symlink link link-id wibble/linktarget test@user-3 no\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n'
SAMPLE_UNICODE_TESTAMENT = u'bazaar-ng testament version 1\nrevision-id: test@user-3\ncommitter: Erik Bågfors <test@user>\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  non-ascii commit © me\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73\n  directory src src-id\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24\nproperties:\n  branch-nick:\n    test branch\n  uni:\n    µ\n'
SAMPLE_UNICODE_TESTAMENT_STRICT = u'bazaar-ng testament version 2.1\nrevision-id: test@user-3\ncommitter: Erik Bågfors <test@user>\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  non-ascii commit © me\ninventory:\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n  uni:\n    µ\n'
SAMPLE_UNICODE_TESTAMENT_STRICT3 = u'bazaar testament version 3 strict\nrevision-id: test@user-3\ncommitter: Erik Bågfors <test@user>\ntimestamp: 1129025493\ntimezone: 36000\nparents:\n  test@user-2\nmessage:\n  non-ascii commit © me\ninventory:\n  directory . TREE_ROT test@user-1 no\n  file hello hello-id 34dd0ac19a24bf80c4d33b5c8960196e8d8d1f73 test@user-2 yes\n  directory src src-id test@user-2 no\n  file src/foo.c foo.c-id a2a049c20f908ae31b231d98779eb63c66448f24 test@user-2 no\nproperties:\n  branch-nick:\n    test branch\n  uni:\n    µ\n'
texts = {Testament: {'rev_1': REV_1_TESTAMENT, 'rev_1_short': REV_1_SHORT, 'rev_2': REV_2_TESTAMENT, 'rev_2_short': REV_2_SHORT, 'rev_3': REV_3_TESTAMENT, 'rev_props': REV_PROPS_TESTAMENT, 'sample_unicode': SAMPLE_UNICODE_TESTAMENT}, StrictTestament: {'rev_1': REV_1_STRICT_TESTAMENT, 'rev_1_short': REV_1_SHORT_STRICT, 'rev_2': REV_2_STRICT_TESTAMENT, 'rev_2_short': REV_2_SHORT_STRICT, 'rev_3': REV_3_TESTAMENT_STRICT, 'rev_props': REV_PROPS_TESTAMENT_STRICT, 'sample_unicode': SAMPLE_UNICODE_TESTAMENT_STRICT}, StrictTestament3: {'rev_1': REV_1_STRICT_TESTAMENT3, 'rev_1_short': REV_1_SHORT_STRICT3, 'rev_2': REV_2_STRICT_TESTAMENT3, 'rev_2_short': REV_2_SHORT_STRICT3, 'rev_3': REV_3_TESTAMENT_STRICT3, 'rev_props': REV_PROPS_TESTAMENT_STRICT3, 'sample_unicode': SAMPLE_UNICODE_TESTAMENT_STRICT3}}