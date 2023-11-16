"""Tests for the launchpad-login command."""
from bzrlib.plugins.launchpad import account
from bzrlib.tests import TestCaseWithTransport

class TestLaunchpadLogin(TestCaseWithTransport):
    """Tests for launchpad-login."""

    def test_login_without_name_when_not_logged_in(self):
        if False:
            while True:
                i = 10
        (out, err) = self.run_bzr(['launchpad-login', '--no-check'], retcode=1)
        self.assertEqual('No Launchpad user ID configured.\n', out)
        self.assertEqual('', err)

    def test_login_with_name_sets_login(self):
        if False:
            while True:
                i = 10
        self.run_bzr(['launchpad-login', '--no-check', 'foo'])
        self.assertEqual('foo', account.get_lp_login())

    def test_login_without_name_when_logged_in(self):
        if False:
            return 10
        account.set_lp_login('foo')
        (out, err) = self.run_bzr(['launchpad-login', '--no-check'])
        self.assertEqual('foo\n', out)
        self.assertEqual('', err)

    def test_login_with_name_no_output_by_default(self):
        if False:
            while True:
                i = 10
        (out, err) = self.run_bzr(['launchpad-login', '--no-check', 'foo'])
        self.assertEqual('', out)
        self.assertEqual('', err)

    def test_login_with_name_verbose(self):
        if False:
            i = 10
            return i + 15
        (out, err) = self.run_bzr(['launchpad-login', '-v', '--no-check', 'foo'])
        self.assertEqual("Launchpad user ID set to 'foo'.\n", out)
        self.assertEqual('', err)