from bzrlib import merge_directive, tests

class TestBundleInfo(tests.TestCaseWithTransport):

    def test_bundle_info(self):
        if False:
            while True:
                i = 10
        source = self.make_branch_and_tree('source')
        self.build_tree(['source/foo'])
        source.add('foo')
        source.commit('added file', rev_id='rev1')
        bundle = open('bundle', 'wb')
        try:
            source.branch.repository.create_bundle('rev1', 'null:', bundle, '4')
        finally:
            bundle.close()
        info = self.run_bzr('bundle-info bundle')[0]
        self.assertContainsRe(info, 'file: [12] .0 multiparent.')
        self.assertContainsRe(info, 'nicks: source')
        self.assertNotContainsRe(info, 'foo')
        self.run_bzr_error(['--verbose requires a merge directive'], 'bundle-info -v bundle')
        target = self.make_branch('target')
        md = merge_directive.MergeDirective2.from_objects(source.branch.repository, 'rev1', 0, 0, 'target', base_revision_id='null:')
        directive = open('directive', 'wb')
        try:
            directive.writelines(md.to_lines())
        finally:
            directive.close()
        info = self.run_bzr('bundle-info -v directive')[0]
        self.assertContainsRe(info, 'foo')