"""Tests for VersionedFile classes"""
from bzrlib import errors, groupcompress, multiparent, tests, versionedfile

class Test_MPDiffGenerator(tests.TestCaseWithMemoryTransport):

    def make_vf(self):
        if False:
            while True:
                i = 10
        t = self.get_transport('')
        factory = groupcompress.make_pack_factory(True, True, 1)
        return factory(t)

    def make_three_vf(self):
        if False:
            while True:
                i = 10
        vf = self.make_vf()
        vf.add_lines(('one',), (), ['first\n'])
        vf.add_lines(('two',), [('one',)], ['first\n', 'second\n'])
        vf.add_lines(('three',), [('one',), ('two',)], ['first\n', 'second\n', 'third\n'])
        return vf

    def test_finds_parents(self):
        if False:
            return 10
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [('three',)])
        (needed_keys, refcount) = gen._find_needed_keys()
        self.assertEqual(sorted([('one',), ('two',), ('three',)]), sorted(needed_keys))
        self.assertEqual({('one',): 1, ('two',): 1}, refcount)

    def test_ignores_ghost_parents(self):
        if False:
            i = 10
            return i + 15
        vf = self.make_vf()
        vf.add_lines(('two',), [('one',)], ['first\n', 'second\n'])
        gen = versionedfile._MPDiffGenerator(vf, [('two',)])
        (needed_keys, refcount) = gen._find_needed_keys()
        self.assertEqual(sorted([('two',)]), sorted(needed_keys))
        self.assertEqual({('one',): 1}, refcount)
        self.assertEqual([('one',)], sorted(gen.ghost_parents))
        self.assertEqual([], sorted(gen.present_parents))

    def test_raises_on_ghost_keys(self):
        if False:
            print('Hello World!')
        vf = self.make_vf()
        gen = versionedfile._MPDiffGenerator(vf, [('one',)])
        self.assertRaises(errors.RevisionNotPresent, gen._find_needed_keys)

    def test_refcount_multiple_children(self):
        if False:
            while True:
                i = 10
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [('two',), ('three',)])
        (needed_keys, refcount) = gen._find_needed_keys()
        self.assertEqual(sorted([('one',), ('two',), ('three',)]), sorted(needed_keys))
        self.assertEqual({('one',): 2, ('two',): 1}, refcount)
        self.assertEqual([('one',)], sorted(gen.present_parents))

    def test_process_contents(self):
        if False:
            print('Hello World!')
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [('two',), ('three',)])
        gen._find_needed_keys()
        self.assertEqual({('two',): (('one',),), ('three',): (('one',), ('two',))}, gen.parent_map)
        self.assertEqual({('one',): 2, ('two',): 1}, gen.refcounts)
        self.assertEqual(sorted([('one',), ('two',), ('three',)]), sorted(gen.needed_keys))
        stream = vf.get_record_stream(gen.needed_keys, 'topological', True)
        record = stream.next()
        self.assertEqual(('one',), record.key)
        gen._process_one_record(record.key, record.get_bytes_as('chunked'))
        self.assertEqual([('one',)], gen.chunks.keys())
        self.assertEqual({('one',): 2, ('two',): 1}, gen.refcounts)
        self.assertEqual([], gen.diffs.keys())
        record = stream.next()
        self.assertEqual(('two',), record.key)
        gen._process_one_record(record.key, record.get_bytes_as('chunked'))
        self.assertEqual(sorted([('one',), ('two',)]), sorted(gen.chunks.keys()))
        self.assertEqual({('one',): 1, ('two',): 1}, gen.refcounts)
        self.assertEqual([('two',)], gen.diffs.keys())
        self.assertEqual({('three',): (('one',), ('two',))}, gen.parent_map)
        record = stream.next()
        self.assertEqual(('three',), record.key)
        gen._process_one_record(record.key, record.get_bytes_as('chunked'))
        self.assertEqual([], gen.chunks.keys())
        self.assertEqual({}, gen.refcounts)
        self.assertEqual(sorted([('two',), ('three',)]), sorted(gen.diffs.keys()))

    def test_compute_diffs(self):
        if False:
            for i in range(10):
                print('nop')
        vf = self.make_three_vf()
        gen = versionedfile._MPDiffGenerator(vf, [('two',), ('three',), ('one',)])
        diffs = gen.compute_diffs()
        expected_diffs = [multiparent.MultiParent([multiparent.ParentText(0, 0, 0, 1), multiparent.NewText(['second\n'])]), multiparent.MultiParent([multiparent.ParentText(1, 0, 0, 2), multiparent.NewText(['third\n'])]), multiparent.MultiParent([multiparent.NewText(['first\n'])])]
        self.assertEqual(expected_diffs, diffs)