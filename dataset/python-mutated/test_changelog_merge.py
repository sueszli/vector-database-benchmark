from bzrlib import merge, tests
from bzrlib.tests import test_merge_core
from bzrlib.plugins.changelog_merge import changelog_merge
sample_base_entries = ['Base entry B1', 'Base entry B2', 'Base entry B3']
sample_this_entries = ['This entry T1', 'This entry T2', 'Base entry B1', 'Base entry B2', 'Base entry B3']
sample_other_entries = ['Other entry O1', 'Base entry B1', 'Base entry B2 updated', 'Base entry B3']
sample2_base_entries = ['Base entry B1', 'Base entry B2', 'Base entry B3']
sample2_this_entries = ['This entry T1', 'This entry T2', 'Base entry B1', 'Base entry B2']
sample2_other_entries = ['Other entry O1', 'Base entry B1 edit', 'Base entry B2']

class TestMergeCoreLogic(tests.TestCase):

    def test_new_in_other_floats_to_top(self):
        if False:
            i = 10
            return i + 15
        "Changes at the top of 'other' float to the top.\n\n        Given a changelog in THIS containing::\n\n          NEW-1\n          OLD-1\n\n        and a changelog in OTHER containing::\n\n          NEW-2\n          OLD-1\n\n        it will merge as::\n\n          NEW-2\n          NEW-1\n          OLD-1\n        "
        base_entries = ['OLD-1']
        this_entries = ['NEW-1', 'OLD-1']
        other_entries = ['NEW-2', 'OLD-1']
        result_entries = changelog_merge.merge_entries(base_entries, this_entries, other_entries)
        self.assertEqual(['NEW-2', 'NEW-1', 'OLD-1'], result_entries)

    def test_acceptance_bug_723968(self):
        if False:
            i = 10
            return i + 15
        'Merging a branch that:\n\n         1. adds a new entry, and\n         2. edits an old entry (e.g. to fix a typo or twiddle formatting)\n\n        will:\n\n         1. add the new entry to the top\n         2. keep the edit, without duplicating the edited entry or moving it.\n        '
        result_entries = changelog_merge.merge_entries(sample_base_entries, sample_this_entries, sample_other_entries)
        self.assertEqual(['Other entry O1', 'This entry T1', 'This entry T2', 'Base entry B1', 'Base entry B2 updated', 'Base entry B3'], list(result_entries))

    def test_more_complex_conflict(self):
        if False:
            return 10
        'Like test_acceptance_bug_723968, but with a more difficult conflict:\n        the new entry and the edited entry are adjacent.\n        '

        def guess_edits(new, deleted):
            if False:
                return 10
            return changelog_merge.default_guess_edits(new, deleted, entry_as_str=lambda x: x)
        result_entries = changelog_merge.merge_entries(sample2_base_entries, sample2_this_entries, sample2_other_entries, guess_edits=guess_edits)
        self.assertEqual(['Other entry O1', 'This entry T1', 'This entry T2', 'Base entry B1 edit', 'Base entry B2'], list(result_entries))

    def test_too_hard(self):
        if False:
            print('Hello World!')
        'A conflict this plugin cannot resolve raises EntryConflict.\n        '
        self.assertRaises(changelog_merge.EntryConflict, changelog_merge.merge_entries, sample2_base_entries, [], sample2_other_entries)

    def test_default_guess_edits(self):
        if False:
            while True:
                i = 10
        'default_guess_edits matches a new entry only once.\n        \n        (Even when that entry is the best match for multiple old entries.)\n        '
        new_in_other = [('AAAAA',), ('BBBBB',)]
        deleted_in_other = [('DDDDD',), ('BBBBBx',), ('BBBBBxx',)]
        result = changelog_merge.default_guess_edits(new_in_other, deleted_in_other)
        self.assertEqual(([('AAAAA',)], [('DDDDD',), ('BBBBBxx',)], [(('BBBBBx',), ('BBBBB',))]), result)

class TestChangeLogMerger(tests.TestCaseWithTransport):
    """Tests for ChangeLogMerger class.
    
    Most tests should be unit tests for merge_entries (and its helpers).
    This class is just to cover the handful of lines of code in ChangeLogMerger
    itself.
    """

    def make_builder(self):
        if False:
            return 10
        builder = test_merge_core.MergeBuilder(self.test_base_dir)
        self.addCleanup(builder.cleanup)
        return builder

    def make_changelog_merger(self, base_text, this_text, other_text):
        if False:
            for i in range(10):
                print('nop')
        builder = self.make_builder()
        builder.add_file('clog-id', builder.tree_root, 'ChangeLog', base_text, True)
        builder.change_contents('clog-id', other=other_text, this=this_text)
        merger = builder.make_merger(merge.Merge3Merger, ['clog-id'])
        merger.this_branch.get_config().set_user_option('changelog_merge_files', 'ChangeLog')
        merge_hook_params = merge.MergeFileHookParams(merger, 'clog-id', None, 'file', 'file', 'conflict')
        changelog_merger = changelog_merge.ChangeLogMerger(merger)
        return (changelog_merger, merge_hook_params)

    def test_merge_text_returns_not_applicable(self):
        if False:
            while True:
                i = 10
        'A conflict this plugin cannot resolve returns (not_applicable, None).\n        '

        def entries_as_str(entries):
            if False:
                print('Hello World!')
            return ''.join((entry + '\n' for entry in entries))
        (changelog_merger, merge_hook_params) = self.make_changelog_merger(entries_as_str(sample2_base_entries), '', entries_as_str(sample2_other_entries))
        self.assertEqual(('not_applicable', None), changelog_merger.merge_contents(merge_hook_params))

    def test_merge_text_returns_success(self):
        if False:
            print('Hello World!')
        "A successful merge returns ('success', lines)."
        (changelog_merger, merge_hook_params) = self.make_changelog_merger('', 'this text\n', 'other text\n')
        (status, lines) = changelog_merger.merge_contents(merge_hook_params)
        self.assertEqual(('success', ['other text\n', 'this text\n']), (status, list(lines)))