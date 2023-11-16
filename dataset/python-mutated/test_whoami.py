"""Black-box tests for bzr whoami."""
import bzrlib
from bzrlib import branch, config, errors, tests

class TestWhoami(tests.TestCaseWithTransport):

    def assertWhoAmI(self, expected, *cmd_args, **kwargs):
        if False:
            return 10
        (out, err) = self.run_bzr(('whoami',) + cmd_args, **kwargs)
        self.assertEqual('', err)
        lines = out.splitlines()
        self.assertLength(1, lines)
        self.assertEqual(expected, lines[0].rstrip())

    def test_whoami_no_args_no_conf(self):
        if False:
            i = 10
            return i + 15
        out = self.run_bzr('whoami')[0]
        self.assertTrue(len(out) > 0)
        self.assertEqual(1, out.count('@'))

    def test_whoami_email_no_args(self):
        if False:
            print('Hello World!')
        out = self.run_bzr('whoami --email')[0]
        self.assertTrue(len(out) > 0)
        self.assertEqual(1, out.count('@'))

    def test_whoami_email_arg(self):
        if False:
            for i in range(10):
                print('nop')
        out = self.run_bzr("whoami --email 'foo <foo@example.com>'", 3)[0]
        self.assertEqual('', out)

    def set_branch_email(self, b, email):
        if False:
            i = 10
            return i + 15
        b.get_config_stack().set('email', email)

    def test_whoami_branch(self):
        if False:
            while True:
                i = 10
        'branch specific user identity works.'
        wt = self.make_branch_and_tree('.')
        b = bzrlib.branch.Branch.open('.')
        self.set_branch_email(b, 'Branch Identity <branch@identi.ty>')
        self.assertWhoAmI('Branch Identity <branch@identi.ty>')
        self.assertWhoAmI('branch@identi.ty', '--email')
        self.overrideEnv('BZR_EMAIL', 'Different ID <other@environ.ment>')
        self.assertWhoAmI('Different ID <other@environ.ment>')
        self.assertWhoAmI('other@environ.ment', '--email')

    def test_whoami_utf8(self):
        if False:
            return 10
        'verify that an identity can be in utf-8.'
        self.run_bzr(['whoami', u'Branch Identity € <branch@identi.ty>'], encoding='utf-8')
        self.assertWhoAmI('Branch Identity â\x82¬ <branch@identi.ty>', encoding='utf-8')
        self.assertWhoAmI('branch@identi.ty', '--email')

    def test_whoami_ascii(self):
        if False:
            while True:
                i = 10
        "\n        verify that whoami doesn't totally break when in utf-8, using an ascii\n        encoding.\n        "
        wt = self.make_branch_and_tree('.')
        b = bzrlib.branch.Branch.open('.')
        self.set_branch_email(b, u'Branch Identity € <branch@identi.ty>')
        self.assertWhoAmI('Branch Identity ? <branch@identi.ty>', encoding='ascii')
        self.assertWhoAmI('branch@identi.ty', '--email', encoding='ascii')

    def test_warning(self):
        if False:
            for i in range(10):
                print('nop')
        'verify that a warning is displayed if no email is given.'
        self.make_branch_and_tree('.')
        display = self.run_bzr(['whoami', 'Branch Identity'])[1]
        self.assertEqual('"Branch Identity" does not seem to contain an email address.  This is allowed, but not recommended.\n', display)

    def test_whoami_not_set(self):
        if False:
            print('Hello World!')
        'Ensure whoami error if username is not set and not inferred.\n        '
        self.overrideEnv('EMAIL', None)
        self.overrideEnv('BZR_EMAIL', None)
        self.overrideAttr(config, '_auto_user_id', lambda : (None, None))
        (out, err) = self.run_bzr(['whoami'], 3)
        self.assertContainsRe(err, 'Unable to determine your name')

    def test_whoami_directory(self):
        if False:
            return 10
        'Test --directory option.'
        wt = self.make_branch_and_tree('subdir')
        self.set_branch_email(wt.branch, 'Branch Identity <branch@identi.ty>')
        self.assertWhoAmI('Branch Identity <branch@identi.ty>', '--directory', 'subdir')
        self.run_bzr(['whoami', '--directory', 'subdir', '--branch', 'Changed Identity <changed@identi.ty>'])
        wt = wt.bzrdir.open_workingtree()
        c = wt.branch.get_config_stack()
        self.assertEqual('Changed Identity <changed@identi.ty>', c.get('email'))

    def test_whoami_remote_directory(self):
        if False:
            print('Hello World!')
        'Test --directory option with a remote directory.'
        wt = self.make_branch_and_tree('subdir')
        self.set_branch_email(wt.branch, 'Branch Identity <branch@identi.ty>')
        url = self.get_readonly_url() + '/subdir'
        self.assertWhoAmI('Branch Identity <branch@identi.ty>', '--directory', url)
        url = self.get_url('subdir')
        self.run_bzr(['whoami', '--directory', url, '--branch', 'Changed Identity <changed@identi.ty>'])
        c = branch.Branch.open(url).get_config_stack()
        self.assertEqual('Changed Identity <changed@identi.ty>', c.get('email'))
        self.overrideEnv('BZR_EMAIL', None)
        self.overrideEnv('EMAIL', None)
        self.overrideAttr(config, '_auto_user_id', lambda : (None, None))
        global_conf = config.GlobalStack()
        self.assertRaises(errors.NoWhoami, global_conf.get, 'email')

    def test_whoami_nonbranch_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test --directory mentioning a non-branch directory.'
        wt = self.build_tree(['subdir/'])
        (out, err) = self.run_bzr('whoami --directory subdir', retcode=3)
        self.assertContainsRe(err, 'ERROR: Not a branch')