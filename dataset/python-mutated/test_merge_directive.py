"""Tests for how merge directives interact with various repository formats.

Bundles contain the serialized form, so changes in serialization based on
repository effects the final bundle.
"""
from bzrlib import chk_map, merge_directive
from bzrlib.tests.scenarios import load_tests_apply_scenarios
from bzrlib.tests.per_repository_vf import TestCaseWithRepository, all_repository_vf_format_scenarios
load_tests = load_tests_apply_scenarios

class TestMergeDirective(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def make_two_branches(self):
        if False:
            while True:
                i = 10
        builder = self.make_branch_builder('source')
        builder.start_series()
        builder.build_snapshot('A', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('f', 'f-id', 'file', 'initial content\n'))])
        builder.build_snapshot('B', 'A', [('modify', ('f-id', 'new content\n'))])
        builder.finish_series()
        b1 = builder.get_branch()
        b2 = b1.bzrdir.sprout('target', revision_id='A').open_branch()
        return (b1, b2)

    def create_merge_directive(self, source_branch, submit_url):
        if False:
            for i in range(10):
                print('nop')
        return merge_directive.MergeDirective2.from_objects(source_branch.repository, source_branch.last_revision(), time=1247775710, timezone=0, target_branch=submit_url)

    def test_create_merge_directive(self):
        if False:
            i = 10
            return i + 15
        (source_branch, target_branch) = self.make_two_branches()
        directive = self.create_merge_directive(source_branch, target_branch.base)
        self.assertIsInstance(directive, merge_directive.MergeDirective2)

    def test_create_and_install_directive(self):
        if False:
            for i in range(10):
                print('nop')
        (source_branch, target_branch) = self.make_two_branches()
        directive = self.create_merge_directive(source_branch, target_branch.base)
        chk_map.clear_cache()
        directive.install_revisions(target_branch.repository)
        rt = target_branch.repository.revision_tree('B')
        rt.lock_read()
        self.assertEqualDiff('new content\n', rt.get_file_text('f-id'))
        rt.unlock()