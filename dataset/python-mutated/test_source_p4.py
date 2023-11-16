import platform
import textwrap
from twisted.internet import error
from twisted.python import reflect
from twisted.trial import unittest
from buildbot.process.results import RETRY
from buildbot.process.results import SUCCESS
from buildbot.steps.source.p4 import P4
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectRmdir
from buildbot.test.steps import ExpectShell
from buildbot.test.util import sourcesteps
from buildbot.test.util.config import ConfigErrorsMixin
from buildbot.test.util.properties import ConstantRenderable
_is_windows = platform.system() == 'Windows'

class TestP4(sourcesteps.SourceStepMixin, TestReactorMixin, ConfigErrorsMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        return self.setUpSourceStep()

    def tearDown(self):
        if False:
            while True:
                i = 10
        return self.tearDownSourceStep()

    def setup_step(self, step, args=None, patch=None, **kwargs):
        if False:
            while True:
                i = 10
        if args is None:
            args = {}
        step = super().setup_step(step, args={}, patch=None, **kwargs)
        self.build.getSourceStamp().revision = args.get('revision', None)
        workspace_dir = '/home/user/workspace'
        if _is_windows:
            workspace_dir = 'C:\\Users\\username\\Workspace'
            self.build.path_module = reflect.namedModule('ntpath')
        self.properties.setProperty('builddir', workspace_dir, 'P4')

    def test_no_empty_step_config(self):
        if False:
            return 10
        with self.assertRaisesConfigError('You must provide p4base or p4viewspec'):
            P4()

    def test_p4base_has_whitespace(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesConfigError('p4base should not end with a trailing / [p4base = //depot with space/]'):
            P4(p4base='//depot with space/')

    def test_p4branch_has_whitespace(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError('p4base should not end with a trailing / [p4base = //depot/]'):
            P4(p4base='//depot/', p4branch='branch with space')

    def test_no_p4base_has_leading_slash_step_config(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesConfigError('p4base should start with // [p4base = depot/]'):
            P4(p4base='depot/')

    def test_no_multiple_type_step_config(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesConfigError('Either provide p4viewspec or p4base and p4branch (and optionally p4extra_views)'):
            P4(p4viewspec=('//depot/trunk', ''), p4base='//depot', p4branch='trunk', p4extra_views=['src', 'doc'])

    def test_no_p4viewspec_is_string_step_config(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesConfigError('p4viewspec must not be a string, and should be a sequence of 2 element sequences'):
            P4(p4viewspec='a_bad_idea')

    def test_no_p4base_has_trailing_slash_step_config(self):
        if False:
            return 10
        with self.assertRaisesConfigError('p4base should not end with a trailing / [p4base = //depot/]'):
            P4(p4base='//depot/')

    def test_no_p4branch_has_trailing_slash_step_config(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError('p4branch should not end with a trailing / [p4branch = blah/]'):
            P4(p4base='//depot', p4branch='blah/')

    def test_no_p4branch_with_no_p4base_step_config(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesConfigError('You must provide p4base or p4viewspec'):
            P4(p4branch='blah')

    def test_no_p4extra_views_with_no_p4base_step_config(self):
        if False:
            return 10
        with self.assertRaisesConfigError('You must provide p4base or p4viewspec'):
            P4(p4extra_views='blah')

    def test_incorrect_mode(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError("mode invalid is not an IRenderable, or one of ('incremental', 'full')"):
            P4(p4base='//depot', mode='invalid')

    def test_mode_incremental_p4base_with_revision(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass'), {'revision': '101'})
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self.expect_commands(ExpectShell(workdir='wkdir', command=['p4', '-V']).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', 'client', '-i'], initial_stdin=client_spec).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', '-ztag', 'changes', '-m1', '//p4_client1/...@101']).stdout('... change 100').exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', 'sync', '//p4_client1/...@100']).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'P4')
        return self.run_step()

    def _incremental(self, client_stdin='', extra_args=None, workdir='wkdir', timeout=20 * 60):
        if False:
            for i in range(10):
                print('nop')
        if extra_args is None:
            extra_args = []
        self.expect_commands(ExpectShell(workdir=workdir, command=['p4', '-V']).exit(0), ExpectShell(workdir=workdir, timeout=timeout, command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', 'client', '-i'], initial_stdin=client_stdin).exit(0), ExpectShell(workdir=workdir, timeout=timeout, command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', '-ztag', 'changes', '-m1', '//p4_client1/...#head']).stdout('... change 100').exit(0), ExpectShell(workdir=workdir, timeout=timeout, command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1'] + extra_args + ['sync', '//p4_client1/...@100']).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'P4')
        return self.run_step()

    def test_mode_incremental_p4base(self):
        if False:
            return 10
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._incremental(client_stdin=client_spec)

    def test_mode_incremental_p4base_with_no_branch(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot/trunk', p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._incremental(client_stdin=client_spec)

    def test_mode_incremental_p4base_with_p4extra_views(self):
        if False:
            print('Hello World!')
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4extra_views=[('-//depot/trunk/test', 'test'), ('-//depot/trunk/doc', 'doc'), ('-//depot/trunk/white space', 'white space')], p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        \t-//depot/trunk/test/... //p4_client1/test/...\n        \t-//depot/trunk/doc/... //p4_client1/doc/...\n        \t"-//depot/trunk/white space/..." "//p4_client1/white space/..."\n        ')
        self._incremental(client_stdin=client_spec)

    def test_mode_incremental_p4viewspec(self):
        if False:
            print('Hello World!')
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4viewspec=[('//depot/trunk/', ''), ('//depot/white space/', 'white space/'), ('-//depot/white space/excluded/', 'white space/excluded/')], p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        \t"//depot/white space/..." "//p4_client1/white space/..."\n        \t"-//depot/white space/excluded/..." "//p4_client1/white space/excluded/..."\n        ')
        self._incremental(client_stdin=client_spec)

    def test_mode_incremental_p4viewspec_suffix(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4viewspec_suffix=None, p4viewspec=[('//depot/trunk/foo.xml', 'bar.xml'), ('//depot/white space/...', 'white space/...'), ('-//depot/white space/excluded/...', 'white space/excluded/...')], p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/foo.xml //p4_client1/bar.xml\n        \t"//depot/white space/..." "//p4_client1/white space/..."\n        \t"-//depot/white space/excluded/..." "//p4_client1/white space/excluded/..."\n        ')
        self._incremental(client_stdin=client_spec)

    def test_mode_incremental_p4client_spec_options(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4client_spec_options='rmdir compress', p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\trmdir compress\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._incremental(client_stdin=client_spec)

    def test_mode_incremental_parent_workdir(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', workdir='../another_wkdir'))
        root_dir = '/home/user/another_wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\another_wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._incremental(client_stdin=client_spec, workdir='../another_wkdir')

    def test_mode_incremental_p4extra_args(self):
        if False:
            return 10
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', p4extra_args=['-Zproxyload']))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._incremental(client_stdin=client_spec, extra_args=['-Zproxyload'])

    def test_mode_incremental_timeout(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', timeout=60 * 60))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._incremental(client_stdin=client_spec, timeout=60 * 60)

    def test_mode_incremental_stream(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', stream=True))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        Stream:\t//depot/trunk\n        ')
        self._incremental(client_stdin=client_spec)

    def _full(self, client_stdin='', p4client='p4_client1', p4user='user', workdir='wkdir', extra_args=None, obfuscated_pass=True):
        if False:
            i = 10
            return i + 15
        if extra_args is None:
            extra_args = []
        if obfuscated_pass:
            expected_pass = ('obfuscated', 'pass', 'XXXXXX')
        else:
            expected_pass = 'pass'
        self.expect_commands(ExpectShell(workdir=workdir, command=['p4', '-V']).exit(0), ExpectShell(workdir=workdir, command=['p4', '-p', 'localhost:12000', '-u', p4user, '-P', expected_pass, '-c', p4client, 'client', '-i'], initial_stdin=client_stdin).exit(0), ExpectShell(workdir=workdir, command=['p4', '-p', 'localhost:12000', '-u', p4user, '-P', expected_pass, '-c', p4client, '-ztag', 'changes', '-m1', f'//{p4client}/...#head']).stdout('... change 100').exit(0), ExpectShell(workdir=workdir, command=['p4', '-p', 'localhost:12000', '-u', p4user, '-P', expected_pass, '-c', p4client] + extra_args + ['sync', '#none']).exit(0), ExpectRmdir(dir=workdir, log_environ=True).exit(0), ExpectShell(workdir=workdir, command=['p4', '-p', 'localhost:12000', '-u', p4user, '-P', expected_pass, '-c', p4client] + extra_args + ['sync', f'//{p4client}/...@100']).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'P4')
        return self.run_step()

    def test_mode_full_p4base(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_stdin = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n')
        self._full(client_stdin=client_stdin)

    def test_mode_full_p4base_not_obfuscated(self):
        if False:
            print('Hello World!')
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass'), worker_version={'*': '2.15'})
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_stdin = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n')
        self._full(client_stdin=client_stdin, obfuscated_pass=False)

    def test_mode_full_p4base_with_no_branch(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot/trunk', p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._full(client_stdin=client_spec)

    def test_mode_full_p4viewspec(self):
        if False:
            return 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4viewspec=[('//depot/main/', ''), ('//depot/main/white space/', 'white space/'), ('-//depot/main/white space/excluded/', 'white space/excluded/')], p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_stdin = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/main/... //p4_client1/...\n        \t"//depot/main/white space/..." "//p4_client1/white space/..."\n        \t"-//depot/main/white space/excluded/..." "//p4_client1/white space/excluded/..."\n        ')
        self._full(client_stdin=client_stdin)

    def test_mode_full_renderable_p4base(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base=ConstantRenderable('//depot'), p4branch='release/1.0', p4user='user', p4client='p4_client2', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_stdin = textwrap.dedent(f'        Client: p4_client2\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/release/1.0/... //p4_client2/...\n')
        self._full(client_stdin=client_stdin, p4client='p4_client2')

    def test_mode_full_renderable_p4client(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch='trunk', p4user='user', p4client=ConstantRenderable('p4_client_render'), p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_stdin = textwrap.dedent(f'        Client: p4_client_render\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client_render/...\n')
        self._full(client_stdin=client_stdin, p4client='p4_client_render')

    def test_mode_full_renderable_p4branch(self):
        if False:
            print('Hello World!')
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch=ConstantRenderable('render_branch'), p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_stdin = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/render_branch/... //p4_client1/...\n')
        self._full(client_stdin=client_stdin)

    def test_mode_full_renderable_p4viewspec(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4viewspec=[(ConstantRenderable('//depot/render_trunk/'), '')], p4user='different_user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_stdin = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: different_user\n\n        Description:\n        \tCreated by different_user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/render_trunk/... //p4_client1/...\n')
        self._full(client_stdin=client_stdin, p4user='different_user')

    def test_mode_full_p4viewspec_suffix(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4viewspec_suffix=None, p4viewspec=[('//depot/trunk/foo.xml', 'bar.xml'), ('//depot/trunk/white space/...', 'white space/...'), ('-//depot/trunk/white space/excluded/...', 'white space/excluded/...')], p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/foo.xml //p4_client1/bar.xml\n        \t"//depot/trunk/white space/..." "//p4_client1/white space/..."\n        \t"-//depot/trunk/white space/excluded/..." "//p4_client1/white space/excluded/..."\n        ')
        self._full(client_stdin=client_spec)

    def test_mode_full_p4client_spec_options(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch='trunk', p4client_spec_options='rmdir compress', p4user='user', p4client='p4_client1', p4passwd='pass'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\trmdir compress\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._full(client_stdin=client_spec)

    def test_mode_full_parent_workdir(self):
        if False:
            return 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', workdir='../another_wkdir'))
        root_dir = '/home/user/another_wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\another_wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._full(client_stdin=client_spec, workdir='../another_wkdir')

    def test_mode_full_p4extra_args(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', p4extra_args=['-Zproxyload']))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self._full(client_stdin=client_spec, extra_args=['-Zproxyload'])

    def test_mode_full_stream(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', stream=True))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        Stream:\t//depot/trunk\n        ')
        self._full(client_stdin=client_spec)

    def test_mode_full_stream_renderable_p4base(self):
        if False:
            return 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base=ConstantRenderable('//depot'), p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', stream=True))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        Stream:\t//depot/trunk\n        ')
        self._full(client_stdin=client_spec)

    def test_mode_full_stream_renderable_p4branch(self):
        if False:
            return 10
        self.setup_step(P4(p4port='localhost:12000', mode='full', p4base='//depot', p4branch=ConstantRenderable('render_branch'), p4user='user', p4client='p4_client1', p4passwd='pass', stream=True))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        Stream:\t//depot/render_branch\n        ')
        self._full(client_stdin=client_spec)

    def test_worker_connection_lost(self):
        if False:
            while True:
                i = 10
        self.setup_step(P4(p4port='localhost:12000', mode='incremental', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['p4', '-V']).error(error.ConnectionLost()))
        self.expect_outcome(result=RETRY, state_string='update (retry)')
        return self.run_step()

    def test_ticket_auth(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(P4(p4port='localhost:12000', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', use_tickets=True))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self.expect_commands(ExpectShell(workdir='wkdir', command=['p4', '-V']).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-c', 'p4_client1', 'login'], initial_stdin='pass\n').exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-c', 'p4_client1', 'client', '-i'], initial_stdin=client_spec).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-c', 'p4_client1', '-ztag', 'changes', '-m1', '//p4_client1/...#head']).stdout('... change 100').exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-c', 'p4_client1', 'sync', '//p4_client1/...@100']).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_client_type_readonly(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(P4(p4port='localhost:12000', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', p4client_type='readonly'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        Type:\treadonly\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self.expect_commands(ExpectShell(workdir='wkdir', command=['p4', '-V']).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', 'client', '-i'], initial_stdin=client_spec).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', '-ztag', 'changes', '-m1', '//p4_client1/...#head']).stdout('... change 100').exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', 'sync', '//p4_client1/...@100']).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_client_type_partitioned(self):
        if False:
            print('Hello World!')
        self.setup_step(P4(p4port='localhost:12000', p4base='//depot', p4branch='trunk', p4user='user', p4client='p4_client1', p4passwd='pass', p4client_type='partitioned'))
        root_dir = '/home/user/workspace/wkdir'
        if _is_windows:
            root_dir = 'C:\\Users\\username\\Workspace\\wkdir'
        client_spec = textwrap.dedent(f'        Client: p4_client1\n\n        Owner: user\n\n        Description:\n        \tCreated by user\n\n        Root:\t{root_dir}\n\n        Options:\tallwrite rmdir\n\n        LineEnd:\tlocal\n\n        Type:\tpartitioned\n\n        View:\n        \t//depot/trunk/... //p4_client1/...\n        ')
        self.expect_commands(ExpectShell(workdir='wkdir', command=['p4', '-V']).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', 'client', '-i'], initial_stdin=client_spec).exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', '-ztag', 'changes', '-m1', '//p4_client1/...#head']).stdout('... change 100').exit(0), ExpectShell(workdir='wkdir', command=['p4', '-p', 'localhost:12000', '-u', 'user', '-P', ('obfuscated', 'pass', 'XXXXXX'), '-c', 'p4_client1', 'sync', '//p4_client1/...@100']).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()