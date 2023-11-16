"""test suite for weave algorithm"""
from pprint import pformat
from bzrlib import errors
from bzrlib.osutils import sha_string
from bzrlib.tests import TestCase, TestCaseInTempDir
from bzrlib.weave import Weave, WeaveFormatError
from bzrlib.weavefile import write_weave, read_weave
TEXT_0 = ['Hello world']
TEXT_1 = ['Hello world', 'A second line']

class TestBase(TestCase):

    def check_read_write(self, k):
        if False:
            while True:
                i = 10
        'Check the weave k can be written & re-read.'
        from tempfile import TemporaryFile
        tf = TemporaryFile()
        write_weave(k, tf)
        tf.seek(0)
        k2 = read_weave(tf)
        if k != k2:
            tf.seek(0)
            self.log('serialized weave:')
            self.log(tf.read())
            self.log('')
            self.log('parents: %s' % (k._parents == k2._parents))
            self.log('         %r' % k._parents)
            self.log('         %r' % k2._parents)
            self.log('')
            self.fail('read/write check failed')

class WeaveContains(TestBase):
    """Weave __contains__ operator"""

    def runTest(self):
        if False:
            print('Hello World!')
        k = Weave(get_scope=lambda : None)
        self.assertFalse('foo' in k)
        k.add_lines('foo', [], TEXT_1)
        self.assertTrue('foo' in k)

class Easy(TestBase):

    def runTest(self):
        if False:
            for i in range(10):
                print('nop')
        k = Weave()

class AnnotateOne(TestBase):

    def runTest(self):
        if False:
            return 10
        k = Weave()
        k.add_lines('text0', [], TEXT_0)
        self.assertEqual(k.annotate('text0'), [('text0', TEXT_0[0])])

class InvalidAdd(TestBase):
    """Try to use invalid version number during add."""

    def runTest(self):
        if False:
            i = 10
            return i + 15
        k = Weave()
        self.assertRaises(errors.RevisionNotPresent, k.add_lines, 'text0', ['69'], ['new text!'])

class RepeatedAdd(TestBase):
    """Add the same version twice; harmless."""

    def test_duplicate_add(self):
        if False:
            i = 10
            return i + 15
        k = Weave()
        idx = k.add_lines('text0', [], TEXT_0)
        idx2 = k.add_lines('text0', [], TEXT_0)
        self.assertEqual(idx, idx2)

class InvalidRepeatedAdd(TestBase):

    def runTest(self):
        if False:
            while True:
                i = 10
        k = Weave()
        k.add_lines('basis', [], TEXT_0)
        idx = k.add_lines('text0', [], TEXT_0)
        self.assertRaises(errors.RevisionAlreadyPresent, k.add_lines, 'text0', [], ['not the same text'])
        self.assertRaises(errors.RevisionAlreadyPresent, k.add_lines, 'text0', ['basis'], TEXT_0)

class InsertLines(TestBase):
    """Store a revision that adds one line to the original.

    Look at the annotations to make sure that the first line is matched
    and not stored repeatedly."""

    def runTest(self):
        if False:
            while True:
                i = 10
        k = Weave()
        k.add_lines('text0', [], ['line 1'])
        k.add_lines('text1', ['text0'], ['line 1', 'line 2'])
        self.assertEqual(k.annotate('text0'), [('text0', 'line 1')])
        self.assertEqual(k.get_lines(1), ['line 1', 'line 2'])
        self.assertEqual(k.annotate('text1'), [('text0', 'line 1'), ('text1', 'line 2')])
        k.add_lines('text2', ['text0'], ['line 1', 'diverged line'])
        self.assertEqual(k.annotate('text2'), [('text0', 'line 1'), ('text2', 'diverged line')])
        text3 = ['line 1', 'middle line', 'line 2']
        k.add_lines('text3', ['text0', 'text1'], text3)
        self.log('k._weave=' + pformat(k._weave))
        self.assertEqual(k.annotate('text3'), [('text0', 'line 1'), ('text3', 'middle line'), ('text1', 'line 2')])
        k.add_lines('text4', ['text0', 'text1', 'text3'], ['line 1', 'aaa', 'middle line', 'bbb', 'line 2', 'ccc'])
        self.assertEqual(k.annotate('text4'), [('text0', 'line 1'), ('text4', 'aaa'), ('text3', 'middle line'), ('text4', 'bbb'), ('text1', 'line 2'), ('text4', 'ccc')])

class DeleteLines(TestBase):
    """Deletion of lines from existing text.

    Try various texts all based on a common ancestor."""

    def runTest(self):
        if False:
            return 10
        k = Weave()
        base_text = ['one', 'two', 'three', 'four']
        k.add_lines('text0', [], base_text)
        texts = [['one', 'two', 'three'], ['two', 'three', 'four'], ['one', 'four'], ['one', 'two', 'three', 'four']]
        i = 1
        for t in texts:
            ver = k.add_lines('text%d' % i, ['text0'], t)
            i += 1
        self.log('final weave:')
        self.log('k._weave=' + pformat(k._weave))
        for i in range(len(texts)):
            self.assertEqual(k.get_lines(i + 1), texts[i])

class SuicideDelete(TestBase):
    """Invalid weave which tries to add and delete simultaneously."""

    def runTest(self):
        if False:
            i = 10
            return i + 15
        k = Weave()
        k._parents = [()]
        k._weave = [('{', 0), 'first line', ('[', 0), 'deleted in 0', (']', 0), ('}', 0)]
        return
        self.assertRaises(WeaveFormatError, k.get_lines, 0)

class CannedDelete(TestBase):
    """Unpack canned weave with deleted lines."""

    def runTest(self):
        if False:
            print('Hello World!')
        k = Weave()
        k._parents = [(), frozenset([0])]
        k._weave = [('{', 0), 'first line', ('[', 1), 'line to be deleted', (']', 1), 'last line', ('}', 0)]
        k._sha1s = [sha_string('first lineline to be deletedlast line'), sha_string('first linelast line')]
        self.assertEqual(k.get_lines(0), ['first line', 'line to be deleted', 'last line'])
        self.assertEqual(k.get_lines(1), ['first line', 'last line'])

class CannedReplacement(TestBase):
    """Unpack canned weave with deleted lines."""

    def runTest(self):
        if False:
            while True:
                i = 10
        k = Weave()
        k._parents = [frozenset(), frozenset([0])]
        k._weave = [('{', 0), 'first line', ('[', 1), 'line to be deleted', (']', 1), ('{', 1), 'replacement line', ('}', 1), 'last line', ('}', 0)]
        k._sha1s = [sha_string('first lineline to be deletedlast line'), sha_string('first linereplacement linelast line')]
        self.assertEqual(k.get_lines(0), ['first line', 'line to be deleted', 'last line'])
        self.assertEqual(k.get_lines(1), ['first line', 'replacement line', 'last line'])

class BadWeave(TestBase):
    """Test that we trap an insert which should not occur."""

    def runTest(self):
        if False:
            i = 10
            return i + 15
        k = Weave()
        k._parents = [frozenset()]
        k._weave = ['bad line', ('{', 0), 'foo {', ('{', 1), '  added in version 1', ('{', 2), '  added in v2', ('}', 2), '  also from v1', ('}', 1), '}', ('}', 0)]
        return
        self.assertRaises(WeaveFormatError, k.get, 0)

class BadInsert(TestBase):
    """Test that we trap an insert which should not occur."""

    def runTest(self):
        if False:
            i = 10
            return i + 15
        k = Weave()
        k._parents = [frozenset(), frozenset([0]), frozenset([0]), frozenset([0, 1, 2])]
        k._weave = [('{', 0), 'foo {', ('{', 1), '  added in version 1', ('{', 1), '  more in 1', ('}', 1), ('}', 1), ('}', 0)]
        return
        self.assertRaises(WeaveFormatError, k.get, 0)
        self.assertRaises(WeaveFormatError, k.get, 1)

class InsertNested(TestBase):
    """Insertion with nested instructions."""

    def runTest(self):
        if False:
            print('Hello World!')
        k = Weave()
        k._parents = [frozenset(), frozenset([0]), frozenset([0]), frozenset([0, 1, 2])]
        k._weave = [('{', 0), 'foo {', ('{', 1), '  added in version 1', ('{', 2), '  added in v2', ('}', 2), '  also from v1', ('}', 1), '}', ('}', 0)]
        k._sha1s = [sha_string('foo {}'), sha_string('foo {  added in version 1  also from v1}'), sha_string('foo {  added in v2}'), sha_string('foo {  added in version 1  added in v2  also from v1}')]
        self.assertEqual(k.get_lines(0), ['foo {', '}'])
        self.assertEqual(k.get_lines(1), ['foo {', '  added in version 1', '  also from v1', '}'])
        self.assertEqual(k.get_lines(2), ['foo {', '  added in v2', '}'])
        self.assertEqual(k.get_lines(3), ['foo {', '  added in version 1', '  added in v2', '  also from v1', '}'])

class DeleteLines2(TestBase):
    """Test recording revisions that delete lines.

    This relies on the weave having a way to represent lines knocked
    out by a later revision."""

    def runTest(self):
        if False:
            print('Hello World!')
        k = Weave()
        k.add_lines('text0', [], ['line the first', 'line 2', 'line 3', 'fine'])
        self.assertEqual(len(k.get_lines(0)), 4)
        k.add_lines('text1', ['text0'], ['line the first', 'fine'])
        self.assertEqual(k.get_lines(1), ['line the first', 'fine'])
        self.assertEqual(k.annotate('text1'), [('text0', 'line the first'), ('text0', 'fine')])

class IncludeVersions(TestBase):
    """Check texts that are stored across multiple revisions.

    Here we manually create a weave with particular encoding and make
    sure it unpacks properly.

    Text 0 includes nothing; text 1 includes text 0 and adds some
    lines.
    """

    def runTest(self):
        if False:
            while True:
                i = 10
        k = Weave()
        k._parents = [frozenset(), frozenset([0])]
        k._weave = [('{', 0), 'first line', ('}', 0), ('{', 1), 'second line', ('}', 1)]
        k._sha1s = [sha_string('first line'), sha_string('first linesecond line')]
        self.assertEqual(k.get_lines(1), ['first line', 'second line'])
        self.assertEqual(k.get_lines(0), ['first line'])

class DivergedIncludes(TestBase):
    """Weave with two diverged texts based on version 0.
    """

    def runTest(self):
        if False:
            print('Hello World!')
        k = Weave()
        k._names = ['0', '1', '2']
        k._name_map = {'0': 0, '1': 1, '2': 2}
        k._parents = [frozenset(), frozenset([0]), frozenset([0])]
        k._weave = [('{', 0), 'first line', ('}', 0), ('{', 1), 'second line', ('}', 1), ('{', 2), 'alternative second line', ('}', 2)]
        k._sha1s = [sha_string('first line'), sha_string('first linesecond line'), sha_string('first linealternative second line')]
        self.assertEqual(k.get_lines(0), ['first line'])
        self.assertEqual(k.get_lines(1), ['first line', 'second line'])
        self.assertEqual(k.get_lines('2'), ['first line', 'alternative second line'])
        self.assertEqual(list(k.get_ancestry(['2'])), ['0', '2'])

class ReplaceLine(TestBase):

    def runTest(self):
        if False:
            while True:
                i = 10
        k = Weave()
        text0 = ['cheddar', 'stilton', 'gruyere']
        text1 = ['cheddar', 'blue vein', 'neufchatel', 'chevre']
        k.add_lines('text0', [], text0)
        k.add_lines('text1', ['text0'], text1)
        self.log('k._weave=' + pformat(k._weave))
        self.assertEqual(k.get_lines(0), text0)
        self.assertEqual(k.get_lines(1), text1)

class Merge(TestBase):
    """Storage of versions that merge diverged parents"""

    def runTest(self):
        if False:
            print('Hello World!')
        k = Weave()
        texts = [['header'], ['header', '', 'line from 1'], ['header', '', 'line from 2', 'more from 2'], ['header', '', 'line from 1', 'fixup line', 'line from 2']]
        k.add_lines('text0', [], texts[0])
        k.add_lines('text1', ['text0'], texts[1])
        k.add_lines('text2', ['text0'], texts[2])
        k.add_lines('merge', ['text0', 'text1', 'text2'], texts[3])
        for (i, t) in enumerate(texts):
            self.assertEqual(k.get_lines(i), t)
        self.assertEqual(k.annotate('merge'), [('text0', 'header'), ('text1', ''), ('text1', 'line from 1'), ('merge', 'fixup line'), ('text2', 'line from 2')])
        self.assertEqual(list(k.get_ancestry(['merge'])), ['text0', 'text1', 'text2', 'merge'])
        self.log('k._weave=' + pformat(k._weave))
        self.check_read_write(k)

class Conflicts(TestBase):
    """Test detection of conflicting regions during a merge.

    A base version is inserted, then two descendents try to
    insert different lines in the same place.  These should be
    reported as a possible conflict and forwarded to the user."""

    def runTest(self):
        if False:
            for i in range(10):
                print('nop')
        return
        k = Weave()
        k.add_lines([], ['aaa', 'bbb'])
        k.add_lines([0], ['aaa', '111', 'bbb'])
        k.add_lines([1], ['aaa', '222', 'bbb'])
        merged = k.merge([1, 2])
        self.assertEqual([[['aaa']], [['111'], ['222']], [['bbb']]])

class NonConflict(TestBase):
    """Two descendants insert compatible changes.

    No conflict should be reported."""

    def runTest(self):
        if False:
            while True:
                i = 10
        return
        k = Weave()
        k.add_lines([], ['aaa', 'bbb'])
        k.add_lines([0], ['111', 'aaa', 'ccc', 'bbb'])
        k.add_lines([1], ['aaa', 'ccc', 'bbb', '222'])

class Khayyam(TestBase):
    """Test changes to multi-line texts, and read/write"""

    def test_multi_line_merge(self):
        if False:
            while True:
                i = 10
        rawtexts = ['A Book of Verses underneath the Bough,\n            A Jug of Wine, a Loaf of Bread, -- and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise enow!', 'A Book of Verses underneath the Bough,\n            A Jug of Wine, a Loaf of Bread, -- and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise now!', 'A Book of poems underneath the tree,\n            A Jug of Wine, a Loaf of Bread,\n            and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise now!\n\n            -- O. Khayyam', 'A Book of Verses underneath the Bough,\n            A Jug of Wine, a Loaf of Bread,\n            and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise now!']
        texts = [[l.strip() for l in t.split('\n')] for t in rawtexts]
        k = Weave()
        parents = set()
        i = 0
        for t in texts:
            ver = k.add_lines('text%d' % i, list(parents), t)
            parents.add('text%d' % i)
            i += 1
        self.log('k._weave=' + pformat(k._weave))
        for (i, t) in enumerate(texts):
            self.assertEqual(k.get_lines(i), t)
        self.check_read_write(k)

class JoinWeavesTests(TestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(JoinWeavesTests, self).setUp()
        self.weave1 = Weave()
        self.lines1 = ['hello\n']
        self.lines3 = ['hello\n', 'cruel\n', 'world\n']
        self.weave1.add_lines('v1', [], self.lines1)
        self.weave1.add_lines('v2', ['v1'], ['hello\n', 'world\n'])
        self.weave1.add_lines('v3', ['v2'], self.lines3)

    def test_written_detection(self):
        if False:
            while True:
                i = 10
        from cStringIO import StringIO
        w = Weave()
        w.add_lines('v1', [], ['hello\n'])
        w.add_lines('v2', ['v1'], ['hello\n', 'there\n'])
        tmpf = StringIO()
        write_weave(w, tmpf)
        self.assertEqual('# bzr weave file v5\ni\n1 f572d396fae9206628714fb2ce00f72e94f2258f\nn v1\n\ni 0\n1 90f265c6e75f1c8f9ab76dcf85528352c5f215ef\nn v2\n\nw\n{ 0\n. hello\n}\n{ 1\n. there\n}\nW\n', tmpf.getvalue())
        tmpf = StringIO('# bzr weave file v5\ni\n1 f572d396fae9206628714fb2ce00f72e94f2258f\nn v1\n\ni 0\n1 90f265c6e75f1c8f9ab76dcf85528352c5f215ef\nn v2\n\nw\n{ 0\n. hello\n}\n{ 1\n. There\n}\nW\n')
        w = read_weave(tmpf)
        self.assertEqual('hello\n', w.get_text('v1'))
        self.assertRaises(errors.WeaveInvalidChecksum, w.get_text, 'v2')
        self.assertRaises(errors.WeaveInvalidChecksum, w.get_lines, 'v2')
        self.assertRaises(errors.WeaveInvalidChecksum, w.check)
        tmpf = StringIO('# bzr weave file v5\ni\n1 f572d396fae9206628714fb2ce00f72e94f2258f\nn v1\n\ni 0\n1 f0f265c6e75f1c8f9ab76dcf85528352c5f215ef\nn v2\n\nw\n{ 0\n. hello\n}\n{ 1\n. there\n}\nW\n')
        w = read_weave(tmpf)
        self.assertEqual('hello\n', w.get_text('v1'))
        self.assertRaises(errors.WeaveInvalidChecksum, w.get_text, 'v2')
        self.assertRaises(errors.WeaveInvalidChecksum, w.get_lines, 'v2')
        self.assertRaises(errors.WeaveInvalidChecksum, w.check)

class TestWeave(TestCase):

    def test_allow_reserved_false(self):
        if False:
            print('Hello World!')
        w = Weave('name', allow_reserved=False)
        w.add_lines('name:', [], TEXT_1)
        self.assertRaises(errors.ReservedId, w.get_lines, 'name:')

    def test_allow_reserved_true(self):
        if False:
            for i in range(10):
                print('nop')
        w = Weave('name', allow_reserved=True)
        w.add_lines('name:', [], TEXT_1)
        self.assertEqual(TEXT_1, w.get_lines('name:'))

class InstrumentedWeave(Weave):
    """Keep track of how many times functions are called."""

    def __init__(self, weave_name=None):
        if False:
            print('Hello World!')
        self._extract_count = 0
        Weave.__init__(self, weave_name=weave_name)

    def _extract(self, versions):
        if False:
            for i in range(10):
                print('nop')
        self._extract_count += 1
        return Weave._extract(self, versions)

class TestNeedsReweave(TestCase):
    """Internal corner cases for when reweave is needed."""

    def test_compatible_parents(self):
        if False:
            i = 10
            return i + 15
        w1 = Weave('a')
        my_parents = set([1, 2, 3])
        self.assertTrue(w1._compatible_parents(my_parents, set([3])))
        self.assertTrue(w1._compatible_parents(my_parents, set(my_parents)))
        self.assertTrue(w1._compatible_parents(set(), set()))
        self.assertFalse(w1._compatible_parents(set(), set([1])))
        self.assertFalse(w1._compatible_parents(my_parents, set([1, 2, 3, 4])))
        self.assertFalse(w1._compatible_parents(my_parents, set([4])))

class TestWeaveFile(TestCaseInTempDir):

    def test_empty_file(self):
        if False:
            i = 10
            return i + 15
        f = open('empty.weave', 'wb+')
        try:
            self.assertRaises(errors.WeaveFormatError, read_weave, f)
        finally:
            f.close()