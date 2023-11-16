from bzrlib.tests import TestCaseWithTransport

class TestCatRevision(TestCaseWithTransport):

    def test_cat_unicode_revision(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        tree.commit('This revision', rev_id='abcd')
        (output, errors) = self.run_bzr(['cat-revision', u'abcd'])
        self.assertContainsRe(output, 'This revision')
        self.assertEqual('', errors)

    def test_cat_revision(self):
        if False:
            i = 10
            return i + 15
        'Test bzr cat-revision.\n        '
        wt = self.make_branch_and_tree('.')
        r = wt.branch.repository
        wt.commit('Commit one', rev_id='a@r-0-1')
        wt.commit('Commit two', rev_id='a@r-0-2')
        wt.commit('Commit three', rev_id='a@r-0-3')
        r.lock_read()
        try:
            revs = {}
            for i in (1, 2, 3):
                revid = 'a@r-0-%d' % i
                stream = r.revisions.get_record_stream([(revid,)], 'unordered', False)
                revs[i] = stream.next().get_bytes_as('fulltext')
        finally:
            r.unlock()
        for i in [1, 2, 3]:
            self.assertEqual(revs[i], self.run_bzr('cat-revision -r revid:a@r-0-%d' % i)[0])
            self.assertEqual(revs[i], self.run_bzr('cat-revision a@r-0-%d' % i)[0])
            self.assertEqual(revs[i], self.run_bzr('cat-revision -r %d' % i)[0])

    def test_cat_no_such_revid(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        err = self.run_bzr('cat-revision abcd', retcode=3)[1]
        self.assertContainsRe(err, 'The repository .* contains no revision abcd.')

    def test_cat_revision_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test --directory option'
        tree = self.make_branch_and_tree('a')
        tree.commit('This revision', rev_id='abcd')
        (output, errors) = self.run_bzr(['cat-revision', '-d', 'a', u'abcd'])
        self.assertContainsRe(output, 'This revision')
        self.assertEqual('', errors)

    def test_cat_tree_less_branch(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        tree.commit('This revision', rev_id='abcd')
        tree.bzrdir.destroy_workingtree()
        (output, errors) = self.run_bzr(['cat-revision', '-d', 'a', u'abcd'])
        self.assertContainsRe(output, 'This revision')
        self.assertEqual('', errors)