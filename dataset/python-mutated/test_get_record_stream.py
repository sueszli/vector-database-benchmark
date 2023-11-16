"""Tests that get_record_stream() behaves itself properly when stacked."""
from bzrlib import errors, knit
from bzrlib.tests.per_repository_reference import TestCaseWithExternalReferenceRepository

class TestGetRecordStream(TestCaseWithExternalReferenceRepository):

    def setUp(self):
        if False:
            return 10
        super(TestGetRecordStream, self).setUp()
        builder = self.make_branch_builder('all')
        builder.start_series()
        builder.build_snapshot('A', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('file', 'f-id', 'file', 'initial content\n'))])
        builder.build_snapshot('B', ['A'], [('modify', ('f-id', 'initial content\nand B content\n'))])
        builder.build_snapshot('C', ['A'], [('modify', ('f-id', 'initial content\nand C content\n'))])
        builder.build_snapshot('D', ['B', 'C'], [('modify', ('f-id', 'initial content\nand B content\nand C content\n'))])
        builder.build_snapshot('E', ['C'], [('modify', ('f-id', 'initial content\nand C content\nand E content\n'))])
        builder.build_snapshot('F', ['D'], [('modify', ('f-id', 'initial content\nand B content\nand C content\nand F content\n'))])
        builder.build_snapshot('G', ['E', 'D'], [('modify', ('f-id', 'initial content\nand B content\nand C content\nand E content\n'))])
        builder.finish_series()
        self.all_repo = builder.get_branch().repository
        self.all_repo.lock_read()
        self.addCleanup(self.all_repo.unlock)
        self.base_repo = self.make_repository('base')
        self.stacked_repo = self.make_referring('referring', self.base_repo)

    def make_simple_split(self):
        if False:
            print('Hello World!')
        'Set up the repositories so that everything is in base except F'
        self.base_repo.fetch(self.all_repo, revision_id='G')
        self.stacked_repo.fetch(self.all_repo, revision_id='F')

    def make_complex_split(self):
        if False:
            for i in range(10):
                print('nop')
        'intermix the revisions so that base holds left stacked holds right.\n\n        base will hold\n            A B D F (and C because it is a parent of D)\n        referring will hold\n            C E G (only)\n        '
        self.base_repo.fetch(self.all_repo, revision_id='B')
        self.stacked_repo.fetch(self.all_repo, revision_id='C')
        self.base_repo.fetch(self.all_repo, revision_id='F')
        self.stacked_repo.fetch(self.all_repo, revision_id='G')

    def test_unordered_fetch_simple_split(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_simple_split()
        keys = [('f-id', r) for r in 'ABCDF']
        self.stacked_repo.lock_read()
        self.addCleanup(self.stacked_repo.unlock)
        stream = self.stacked_repo.texts.get_record_stream(keys, 'unordered', False)
        record_keys = set()
        for record in stream:
            if record.storage_kind == 'absent':
                raise ValueError('absent record: %s' % (record.key,))
            record_keys.add(record.key)
        self.assertEqual(keys, sorted(record_keys))

    def test_unordered_fetch_complex_split(self):
        if False:
            while True:
                i = 10
        self.make_complex_split()
        keys = [('f-id', r) for r in 'ABCDEG']
        self.stacked_repo.lock_read()
        self.addCleanup(self.stacked_repo.unlock)
        stream = self.stacked_repo.texts.get_record_stream(keys, 'unordered', False)
        record_keys = set()
        for record in stream:
            if record.storage_kind == 'absent':
                raise ValueError('absent record: %s' % (record.key,))
            record_keys.add(record.key)
        self.assertEqual(keys, sorted(record_keys))

    def test_ordered_no_closure(self):
        if False:
            while True:
                i = 10
        self.make_complex_split()
        keys = [('f-id', r) for r in 'ABCDEG']
        alt_1 = [('f-id', r) for r in 'ACBDEG']
        alt_2 = [('f-id', r) for r in 'ABCEDG']
        alt_3 = [('f-id', r) for r in 'ACBEDG']
        self.stacked_repo.lock_read()
        self.addCleanup(self.stacked_repo.unlock)
        stream = self.stacked_repo.texts.get_record_stream(keys, 'topological', False)
        record_keys = []
        for record in stream:
            if record.storage_kind == 'absent':
                raise ValueError('absent record: %s' % (record.key,))
            record_keys.append(record.key)
        self.assertTrue(record_keys in (keys, alt_1, alt_2, alt_3))

    def test_ordered_fulltext_simple(self):
        if False:
            while True:
                i = 10
        self.make_simple_split()
        keys = [('f-id', r) for r in 'ABCDF']
        alt_1 = [('f-id', r) for r in 'ACBDF']
        self.stacked_repo.lock_read()
        self.addCleanup(self.stacked_repo.unlock)
        stream = self.stacked_repo.texts.get_record_stream(keys, 'topological', True)
        record_keys = []
        for record in stream:
            if record.storage_kind == 'absent':
                raise ValueError('absent record: %s' % (record.key,))
            record_keys.append(record.key)
        self.assertTrue(record_keys in (keys, alt_1))

    def test_ordered_fulltext_complex(self):
        if False:
            while True:
                i = 10
        self.make_complex_split()
        keys = [('f-id', r) for r in 'ABCDEG']
        alt_1 = [('f-id', r) for r in 'ACBDEG']
        alt_2 = [('f-id', r) for r in 'ABCEDG']
        alt_3 = [('f-id', r) for r in 'ACBEDG']
        self.stacked_repo.lock_read()
        self.addCleanup(self.stacked_repo.unlock)
        stream = self.stacked_repo.texts.get_record_stream(keys, 'topological', True)
        record_keys = []
        for record in stream:
            if record.storage_kind == 'absent':
                raise ValueError('absent record: %s' % (record.key,))
            record_keys.append(record.key)
        if isinstance(self.stacked_repo.texts, knit.KnitVersionedFiles):
            self.expectFailure('KVF does not weave fulltexts from fallback repositories to preserve perfect order', self.assertTrue, record_keys in (keys, alt_1, alt_2, alt_3))
        self.assertTrue(record_keys in (keys, alt_1, alt_2, alt_3))