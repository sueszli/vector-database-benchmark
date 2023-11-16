from bzrlib import branch, urlutils
from bzrlib.tests import script

class TestRememberMixin(object):
    """--remember and --no-remember set locations or not."""
    command = []
    working_dir = None
    first_use_args = []
    next_uses_args = []

    def do_command(self, *args):
        if False:
            i = 10
            return i + 15
        (out, err) = self.run_bzr(self.command + list(args), working_dir=self.working_dir)

    def test_first_use_no_option(self):
        if False:
            print('Hello World!')
        self.do_command(*self.first_use_args)
        self.assertLocations(self.first_use_args)

    def test_first_use_remember(self):
        if False:
            while True:
                i = 10
        self.do_command('--remember', *self.first_use_args)
        self.assertLocations(self.first_use_args)

    def test_first_use_no_remember(self):
        if False:
            for i in range(10):
                print('nop')
        self.do_command('--no-remember', *self.first_use_args)
        self.assertLocations([])

    def test_next_uses_no_option(self):
        if False:
            return 10
        self.setup_next_uses()
        self.do_command(*self.next_uses_args)
        self.assertLocations(self.first_use_args)

    def test_next_uses_remember(self):
        if False:
            i = 10
            return i + 15
        self.setup_next_uses()
        self.do_command('--remember', *self.next_uses_args)
        self.assertLocations(self.next_uses_args)

    def test_next_uses_no_remember(self):
        if False:
            return 10
        self.setup_next_uses()
        self.do_command('--no-remember', *self.next_uses_args)
        self.assertLocations(self.first_use_args)

class TestSendRemember(script.TestCaseWithTransportAndScript, TestRememberMixin):
    working_dir = 'work'
    command = ['send', '-o-']
    first_use_args = ['../parent', '../grand_parent']
    next_uses_args = ['../new_parent', '../new_grand_parent']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestSendRemember, self).setUp()
        self.run_script("\n            $ bzr init grand_parent\n            $ cd grand_parent\n            $ echo grand_parent > file\n            $ bzr add\n            $ bzr commit -m 'initial commit'\n            $ cd ..\n            $ bzr branch grand_parent parent\n            $ cd parent\n            $ echo parent > file\n            $ bzr commit -m 'parent'\n            $ cd ..\n            $ bzr branch parent %(working_dir)s\n            $ cd %(working_dir)s\n            $ echo %(working_dir)s > file\n            $ bzr commit -m '%(working_dir)s'\n            $ cd ..\n            " % {'working_dir': self.working_dir}, null_output_matches_anything=True)

    def setup_next_uses(self):
        if False:
            i = 10
            return i + 15
        self.do_command(*self.first_use_args)
        self.run_script('\n            $ bzr branch grand_parent new_grand_parent\n            $ bzr branch parent new_parent\n            ', null_output_matches_anything=True)

    def assertLocations(self, expected_locations):
        if False:
            i = 10
            return i + 15
        if not expected_locations:
            (expected_submit_branch, expected_public_branch) = (None, None)
        else:
            (expected_submit_branch, expected_public_branch) = expected_locations
        (br, _) = branch.Branch.open_containing(self.working_dir)
        self.assertEqual(expected_submit_branch, br.get_submit_branch())
        self.assertEqual(expected_public_branch, br.get_public_branch())

class TestPushRemember(script.TestCaseWithTransportAndScript, TestRememberMixin):
    working_dir = 'work'
    command = ['push']
    first_use_args = ['../target']
    next_uses_args = ['../new_target']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestPushRemember, self).setUp()
        self.run_script("\n            $ bzr init %(working_dir)s\n            $ cd %(working_dir)s\n            $ echo some content > file\n            $ bzr add\n            $ bzr commit -m 'initial commit'\n            $ cd ..\n            " % {'working_dir': self.working_dir}, null_output_matches_anything=True)

    def setup_next_uses(self):
        if False:
            i = 10
            return i + 15
        self.do_command(*self.first_use_args)
        self.run_script("\n            $ cd %(working_dir)s\n            $ echo new content > file\n            $ bzr commit -m 'new content'\n            $ cd ..\n            " % {'working_dir': self.working_dir}, null_output_matches_anything=True)

    def assertLocations(self, expected_locations):
        if False:
            while True:
                i = 10
        (br, _) = branch.Branch.open_containing(self.working_dir)
        if not expected_locations:
            self.assertEqual(None, br.get_push_location())
        else:
            expected_push_location = expected_locations[0]
            push_location = urlutils.relative_url(br.base, br.get_push_location())
            self.assertIsSameRealPath(expected_push_location, push_location)

class TestPullRemember(script.TestCaseWithTransportAndScript, TestRememberMixin):
    working_dir = 'work'
    command = ['pull']
    first_use_args = ['../parent']
    next_uses_args = ['../new_parent']

    def setUp(self):
        if False:
            return 10
        super(TestPullRemember, self).setUp()
        self.run_script("\n            $ bzr init parent\n            $ cd parent\n            $ echo parent > file\n            $ bzr add\n            $ bzr commit -m 'initial commit'\n            $ cd ..\n            $ bzr init %(working_dir)s\n            " % {'working_dir': self.working_dir}, null_output_matches_anything=True)

    def setup_next_uses(self):
        if False:
            for i in range(10):
                print('nop')
        self.do_command(*self.first_use_args)
        self.run_script("\n            $ bzr branch parent new_parent\n            $ cd new_parent\n            $ echo new parent > file\n            $ bzr commit -m 'new parent'\n            $ cd ..\n            " % {'working_dir': self.working_dir}, null_output_matches_anything=True)

    def assertLocations(self, expected_locations):
        if False:
            for i in range(10):
                print('nop')
        (br, _) = branch.Branch.open_containing(self.working_dir)
        if not expected_locations:
            self.assertEqual(None, br.get_parent())
        else:
            expected_pull_location = expected_locations[0]
            pull_location = urlutils.relative_url(br.base, br.get_parent())
            self.assertIsSameRealPath(expected_pull_location, pull_location)