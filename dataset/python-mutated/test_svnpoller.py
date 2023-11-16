import os
import xml.dom.minidom
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.changes import svnpoller
from buildbot.process.properties import Interpolate
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.runprocess import ExpectMasterShell
from buildbot.test.runprocess import MasterRunProcessMixin
from buildbot.test.util import changesource
prefix_output = b'<?xml version="1.0"?>\n<info>\n<entry\n   kind="dir"\n   path="trunk"\n   revision="18354">\n<url>svn+ssh://svn.twistedmatrix.com/svn/Twisted/trunk</url>\n<repository>\n<root>svn+ssh://svn.twistedmatrix.com/svn/Twisted</root>\n<uuid>bbbe8e31-12d6-0310-92fd-ac37d47ddeeb</uuid>\n</repository>\n<commit\n   revision="18352">\n<author>jml</author>\n<date>2006-10-01T02:37:34.063255Z</date>\n</commit>\n</entry>\n</info>\n'
prefix_output_2 = b'<?xml version="1.0"?>\n<info>\n</info>\n'
prefix_output_3 = b'<?xml version="1.0"?>\n<info>\n<entry\n   kind="dir"\n   path="SVN-Repository"\n   revision="3">\n<url>file:///home/warner/stuff/Projects/Buildbot/trees/svnpoller/_trial_temp/test_vc/repositories/SVN-Repository</url>\n<repository>\n<root>file:///home/warner/stuff/Projects/Buildbot/trees/svnpoller/_trial_temp/test_vc/repositories/SVN-Repository</root>\n<uuid>c0f47ff4-ba1e-0410-96b5-d44cc5c79e7f</uuid>\n</repository>\n<commit\n   revision="3">\n<author>warner</author>\n<date>2006-10-01T07:37:04.182499Z</date>\n</commit>\n</entry>\n</info>\n'
prefix_output_4 = b'<?xml version="1.0"?>\n<info>\n<entry\n   kind="dir"\n   path="trunk"\n   revision="3">\n<url>file:///home/warner/stuff/Projects/Buildbot/trees/svnpoller/_trial_temp/test_vc/repositories/SVN-Repository/sample/trunk</url>\n<repository>\n<root>file:///home/warner/stuff/Projects/Buildbot/trees/svnpoller/_trial_temp/test_vc/repositories/SVN-Repository</root>\n<uuid>c0f47ff4-ba1e-0410-96b5-d44cc5c79e7f</uuid>\n</repository>\n<commit\n   revision="1">\n<author>warner</author>\n<date>2006-10-01T07:37:02.286440Z</date>\n</commit>\n</entry>\n</info>\n'
sample_base = 'file:///usr/home/warner/stuff/Projects/Buildbot/trees/misc/' + '_trial_temp/test_vc/repositories/SVN-Repository/sample'
sample_logentries = [None] * 6
sample_logentries[5] = b'<logentry\n   revision="6">\n<author>warner</author>\n<date>2006-10-01T19:35:16.165664Z</date>\n<paths>\n<path\n   action="D">/sample/branch/version.c</path>\n</paths>\n<msg>revised_to_2</msg>\n</logentry>\n'
sample_logentries[4] = b'<logentry\n   revision="5">\n<author>warner</author>\n<date>2006-10-01T19:35:16.165664Z</date>\n<paths>\n<path\n   action="D">/sample/branch</path>\n</paths>\n<msg>revised_to_2</msg>\n</logentry>\n'
sample_logentries[3] = b'<logentry\n   revision="4">\n<author>warner</author>\n<date>2006-10-01T19:35:16.165664Z</date>\n<paths>\n<path\n   action="M">/sample/trunk/version.c</path>\n</paths>\n<msg>revised_to_2</msg>\n</logentry>\n'
sample_logentries[2] = b'<logentry\n   revision="3">\n<author>warner</author>\n<date>2006-10-01T19:35:10.215692Z</date>\n<paths>\n<path\n   action="M">/sample/branch/c\xcc\xa7main.c</path>\n</paths>\n<msg>commit_on_branch</msg>\n</logentry>\n'
sample_logentries[1] = b'<logentry\n   revision="2">\n<author>warner</author>\n<date>2006-10-01T19:35:09.154973Z</date>\n<paths>\n<path\n   copyfrom-path="/sample/trunk"\n   copyfrom-rev="1"\n   action="A">/sample/branch</path>\n</paths>\n<msg>make_branch</msg>\n</logentry>\n'
sample_logentries[0] = b'<logentry\n   revision="1">\n<author>warner</author>\n<date>2006-10-01T19:35:08.642045Z</date>\n<paths>\n<path\n   action="A">/sample</path>\n<path\n   action="A">/sample/trunk</path>\n<path\n   action="A">/sample/trunk/subdir/subdir.c</path>\n<path\n   action="A">/sample/trunk/main.c</path>\n<path\n   action="A">/sample/trunk/version.c</path>\n<path\n   action="A">/sample/trunk/subdir</path>\n</paths>\n<msg>sample_project_files</msg>\n</logentry>\n'
sample_info_output = b'<?xml version="1.0"?>\n<info>\n<entry\n   kind="dir"\n   path="sample"\n   revision="4">\n<url>file:///usr/home/warner/stuff/Projects/Buildbot/trees/misc/_trial_temp/test_vc/repositories/SVN-Repository/sample</url>\n<repository>\n<root>file:///usr/home/warner/stuff/Projects/Buildbot/trees/misc/_trial_temp/test_vc/repositories/SVN-Repository</root>\n<uuid>4f94adfc-c41e-0410-92d5-fbf86b7c7689</uuid>\n</repository>\n<commit\n   revision="4">\n<author>warner</author>\n<date>2006-10-01T19:35:16.165664Z</date>\n</commit>\n</entry>\n</info>\n'

def make_changes_output(maxrevision):
    if False:
        return 10
    logs = sample_logentries[0:maxrevision]
    assert len(logs) == maxrevision
    logs.reverse()
    output = b'<?xml version="1.0"?>\n<log>' + b''.join(logs) + b'</log>'
    return output

def make_logentry_elements(maxrevision):
    if False:
        while True:
            i = 10
    'return the corresponding logentry elements for the given revisions'
    doc = xml.dom.minidom.parseString(make_changes_output(maxrevision))
    return doc.getElementsByTagName('logentry')

def split_file(path):
    if False:
        print('Hello World!')
    pieces = path.split('/')
    if pieces[0] == 'branch':
        return {'branch': 'branch', 'path': '/'.join(pieces[1:])}
    if pieces[0] == 'trunk':
        return {'path': '/'.join(pieces[1:])}
    raise RuntimeError(f"there shouldn't be any files like {repr(path)}")

class TestSVNPoller(MasterRunProcessMixin, changesource.ChangeSourceMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.setup_master_run_process()
        return self.setUpChangeSource()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tearDownChangeSource()

    @defer.inlineCallbacks
    def attachSVNPoller(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        s = svnpoller.SVNPoller(*args, **kwargs)
        yield self.attachChangeSource(s)
        return s

    @defer.inlineCallbacks
    def test_describe(self):
        if False:
            i = 10
            return i + 15
        s = (yield self.attachSVNPoller('file://'))
        self.assertSubstring('SVNPoller', s.describe())

    @defer.inlineCallbacks
    def test_name(self):
        if False:
            while True:
                i = 10
        s = (yield self.attachSVNPoller('file://'))
        self.assertEqual('file://', s.name)
        s = (yield self.attachSVNPoller('file://', name='MyName'))
        self.assertEqual('MyName', s.name)

    @defer.inlineCallbacks
    def test_strip_repourl(self):
        if False:
            while True:
                i = 10
        base = 'svn+ssh://svn.twistedmatrix.com/svn/Twisted/trunk'
        s = (yield self.attachSVNPoller(base + '/'))
        self.assertEqual(s.repourl, base)

    @defer.inlineCallbacks
    def do_test_get_prefix(self, base, output, expected):
        if False:
            return 10
        s = (yield self.attachSVNPoller(base))
        self.expect_commands(ExpectMasterShell(['svn', 'info', '--xml', '--non-interactive', base]).stdout(output))
        prefix = (yield s.get_prefix())
        self.assertEqual(prefix, expected)
        self.assert_all_commands_ran()

    def test_get_prefix_1(self):
        if False:
            while True:
                i = 10
        base = 'svn+ssh://svn.twistedmatrix.com/svn/Twisted/trunk'
        return self.do_test_get_prefix(base, prefix_output, 'trunk')

    def test_get_prefix_2(self):
        if False:
            for i in range(10):
                print('nop')
        base = 'svn+ssh://svn.twistedmatrix.com/svn/Twisted'
        return self.do_test_get_prefix(base, prefix_output_2, '')

    def test_get_prefix_3(self):
        if False:
            return 10
        base = 'file:///home/warner/stuff/Projects/Buildbot/trees/' + 'svnpoller/_trial_temp/test_vc/repositories/SVN-Repository'
        return self.do_test_get_prefix(base, prefix_output_3, '')

    def test_get_prefix_4(self):
        if False:
            i = 10
            return i + 15
        base = 'file:///home/warner/stuff/Projects/Buildbot/trees/' + 'svnpoller/_trial_temp/test_vc/repositories/SVN-Repository/sample/trunk'
        return self.do_test_get_prefix(base, prefix_output_3, 'sample/trunk')

    @defer.inlineCallbacks
    def test_log_parsing(self):
        if False:
            print('Hello World!')
        s = (yield self.attachSVNPoller('file:///foo'))
        output = make_changes_output(4)
        entries = s.parse_logs(output)
        self.assertEqual(len(entries), 4)

    @defer.inlineCallbacks
    def test_get_new_logentries(self):
        if False:
            for i in range(10):
                print('nop')
        s = (yield self.attachSVNPoller('file:///foo'))
        entries = make_logentry_elements(4)
        s.last_change = 4
        new = s.get_new_logentries(entries)
        self.assertEqual(s.last_change, 4)
        self.assertEqual(len(new), 0)
        s.last_change = 3
        new = s.get_new_logentries(entries)
        self.assertEqual(s.last_change, 4)
        self.assertEqual(len(new), 1)
        s.last_change = 1
        new = s.get_new_logentries(entries)
        self.assertEqual(s.last_change, 4)
        self.assertEqual(len(new), 3)
        s.last_change = None
        new = s.get_new_logentries(entries)
        self.assertEqual(s.last_change, 4)
        self.assertEqual(len(new), 0)

    @defer.inlineCallbacks
    def test_get_text(self):
        if False:
            print('Hello World!')
        doc = xml.dom.minidom.parseString('\n            <parent>\n                <child>\n                    hi\n                    <grandchild>1</grandchild>\n                    <grandchild>2</grandchild>\n                </child>\n            </parent>'.strip())
        s = (yield self.attachSVNPoller('http://', split_file=split_file))
        self.assertEqual(s._get_text(doc, 'grandchild'), '1')
        self.assertEqual(s._get_text(doc, 'nonexistent'), 'unknown')

    @defer.inlineCallbacks
    def test_create_changes(self):
        if False:
            return 10
        base = 'file:///home/warner/stuff/Projects/Buildbot/trees/' + 'svnpoller/_trial_temp/test_vc/repositories/SVN-Repository/sample'
        s = (yield self.attachSVNPoller(base, split_file=split_file))
        s._prefix = 'sample'
        logentries = dict(zip(range(1, 7), reversed(make_logentry_elements(6))))
        changes = s.create_changes(reversed([logentries[3], logentries[2]]))
        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0]['branch'], 'branch')
        self.assertEqual(changes[0]['revision'], '2')
        self.assertEqual(changes[0]['project'], '')
        self.assertEqual(changes[0]['repository'], base)
        self.assertEqual(changes[1]['branch'], 'branch')
        self.assertEqual(changes[1]['files'], ['çmain.c'])
        self.assertEqual(changes[1]['revision'], '3')
        self.assertEqual(changes[1]['project'], '')
        self.assertEqual(changes[1]['repository'], base)
        changes = s.create_changes([logentries[4]])
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]['branch'], None)
        self.assertEqual(changes[0]['revision'], '4')
        self.assertEqual(changes[0]['files'], ['version.c'])
        changes = s.create_changes([logentries[5]])
        self.assertEqual(len(changes), 0)
        changes = s.create_changes([logentries[6]])
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]['branch'], 'branch')
        self.assertEqual(changes[0]['revision'], '6')
        self.assertEqual(changes[0]['files'], ['version.c'])

    def makeInfoExpect(self, password='bbrocks'):
        if False:
            return 10
        args = ['svn', 'info', '--xml', '--non-interactive', sample_base, '--username=dustin']
        if password is not None:
            args.append('--password=' + password)
        return ExpectMasterShell(args)

    def makeLogExpect(self, password='bbrocks'):
        if False:
            print('Hello World!')
        args = ['svn', 'log', '--xml', '--verbose', '--non-interactive', '--username=dustin']
        if password is not None:
            args.append('--password=' + password)
        args.extend(['--limit=100', sample_base])
        return ExpectMasterShell(args)

    @defer.inlineCallbacks
    def test_create_changes_overridden_project(self):
        if False:
            i = 10
            return i + 15

        def custom_split_file(path):
            if False:
                return 10
            f = split_file(path)
            if f:
                f['project'] = 'overridden-project'
                f['repository'] = 'overridden-repository'
                f['codebase'] = 'overridden-codebase'
            return f
        base = 'file:///home/warner/stuff/Projects/Buildbot/trees/' + 'svnpoller/_trial_temp/test_vc/repositories/SVN-Repository/sample'
        s = (yield self.attachSVNPoller(base, split_file=custom_split_file))
        s._prefix = 'sample'
        logentries = dict(zip(range(1, 7), reversed(make_logentry_elements(6))))
        changes = s.create_changes(reversed([logentries[3], logentries[2]]))
        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0]['branch'], 'branch')
        self.assertEqual(changes[0]['revision'], '2')
        self.assertEqual(changes[0]['project'], 'overridden-project')
        self.assertEqual(changes[0]['repository'], 'overridden-repository')
        self.assertEqual(changes[0]['codebase'], 'overridden-codebase')
        self.assertEqual(changes[1]['branch'], 'branch')
        self.assertEqual(changes[1]['files'], ['çmain.c'])
        self.assertEqual(changes[1]['revision'], '3')
        self.assertEqual(changes[1]['project'], 'overridden-project')
        self.assertEqual(changes[1]['repository'], 'overridden-repository')
        self.assertEqual(changes[1]['codebase'], 'overridden-codebase')

    @defer.inlineCallbacks
    def test_poll(self):
        if False:
            return 10
        s = (yield self.attachSVNPoller(sample_base, split_file=split_file, svnuser='dustin', svnpasswd='bbrocks'))
        self.expect_commands(self.makeInfoExpect().stdout(sample_info_output), self.makeLogExpect().stdout(make_changes_output(1)), self.makeLogExpect().stdout(make_changes_output(1)), self.makeLogExpect().stdout(make_changes_output(2)), self.makeLogExpect().stdout(make_changes_output(4)))
        yield s.poll()
        self.assertEqual(self.master.data.updates.changesAdded, [])
        self.assertEqual(s.last_change, 1)
        yield s.poll()
        self.assertEqual(self.master.data.updates.changesAdded, [])
        self.assertEqual(s.last_change, 1)
        yield s.poll()
        self.assertEqual(self.master.data.updates.changesAdded, [{'author': 'warner', 'committer': None, 'branch': 'branch', 'category': None, 'codebase': None, 'comments': 'make_branch', 'files': [''], 'project': '', 'properties': {}, 'repository': 'file:///usr/home/warner/stuff/Projects/Buildbot/trees/misc/_trial_temp/test_vc/repositories/SVN-Repository/sample', 'revision': '2', 'revlink': '', 'src': 'svn', 'when_timestamp': None}])
        self.assertEqual(s.last_change, 2)
        self.master.data.updates.changesAdded = []
        yield s.poll()
        self.assertEqual(self.master.data.updates.changesAdded, [{'author': 'warner', 'committer': None, 'branch': 'branch', 'category': None, 'codebase': None, 'comments': 'commit_on_branch', 'files': ['çmain.c'], 'project': '', 'properties': {}, 'repository': 'file:///usr/home/warner/stuff/Projects/Buildbot/trees/misc/_trial_temp/test_vc/repositories/SVN-Repository/sample', 'revision': '3', 'revlink': '', 'src': 'svn', 'when_timestamp': None}, {'author': 'warner', 'committer': None, 'branch': None, 'category': None, 'codebase': None, 'comments': 'revised_to_2', 'files': ['version.c'], 'project': '', 'properties': {}, 'repository': 'file:///usr/home/warner/stuff/Projects/Buildbot/trees/misc/_trial_temp/test_vc/repositories/SVN-Repository/sample', 'revision': '4', 'revlink': '', 'src': 'svn', 'when_timestamp': None}])
        self.assertEqual(s.last_change, 4)
        self.assert_all_commands_ran()

    @defer.inlineCallbacks
    def test_poll_empty_password(self):
        if False:
            for i in range(10):
                print('nop')
        s = (yield self.attachSVNPoller(sample_base, split_file=split_file, svnuser='dustin', svnpasswd=''))
        self.expect_commands(self.makeInfoExpect(password='').stdout(sample_info_output), self.makeLogExpect(password='').stdout(make_changes_output(1)), self.makeLogExpect(password='').stdout(make_changes_output(1)), self.makeLogExpect(password='').stdout(make_changes_output(2)), self.makeLogExpect(password='').stdout(make_changes_output(4)))
        yield s.poll()

    @defer.inlineCallbacks
    def test_poll_no_password(self):
        if False:
            while True:
                i = 10
        s = (yield self.attachSVNPoller(sample_base, split_file=split_file, svnuser='dustin'))
        self.expect_commands(self.makeInfoExpect(password=None).stdout(sample_info_output), self.makeLogExpect(password=None).stdout(make_changes_output(1)), self.makeLogExpect(password=None).stdout(make_changes_output(1)), self.makeLogExpect(password=None).stdout(make_changes_output(2)), self.makeLogExpect(password=None).stdout(make_changes_output(4)))
        yield s.poll()

    @defer.inlineCallbacks
    def test_poll_interpolated_password(self):
        if False:
            i = 10
            return i + 15
        s = (yield self.attachSVNPoller(sample_base, split_file=split_file, svnuser='dustin', svnpasswd=Interpolate('pa$$')))
        self.expect_commands(self.makeInfoExpect(password='pa$$').stdout(sample_info_output), self.makeLogExpect(password='pa$$').stdout(make_changes_output(1)), self.makeLogExpect(password='pa$$').stdout(make_changes_output(1)), self.makeLogExpect(password='pa$$').stdout(make_changes_output(2)), self.makeLogExpect(password='pa$$').stdout(make_changes_output(4)))
        yield s.poll()

    @defer.inlineCallbacks
    def test_poll_get_prefix_exception(self):
        if False:
            while True:
                i = 10
        s = (yield self.attachSVNPoller(sample_base, split_file=split_file, svnuser='dustin', svnpasswd='bbrocks'))
        self.expect_commands(self.makeInfoExpect().stderr(b'error'))
        yield s.poll()
        self.assertEqual(len(self.flushLoggedErrors(EnvironmentError)), 1)
        self.assert_all_commands_ran()

    @defer.inlineCallbacks
    def test_poll_get_logs_exception(self):
        if False:
            print('Hello World!')
        s = (yield self.attachSVNPoller(sample_base, split_file=split_file, svnuser='dustin', svnpasswd='bbrocks'))
        s._prefix = 'abc'
        self.expect_commands(self.makeLogExpect().stderr(b'some error'))
        yield s.poll()
        self.assertEqual(len(self.flushLoggedErrors(EnvironmentError)), 1)
        self.assert_all_commands_ran()

    @defer.inlineCallbacks
    def test_cachepath_empty(self):
        if False:
            print('Hello World!')
        cachepath = os.path.abspath('revcache')
        if os.path.exists(cachepath):
            os.unlink(cachepath)
        s = (yield self.attachSVNPoller(sample_base, cachepath=cachepath))
        self.assertEqual(s.last_change, None)

    @defer.inlineCallbacks
    def test_cachepath_full(self):
        if False:
            for i in range(10):
                print('nop')
        cachepath = os.path.abspath('revcache')
        with open(cachepath, 'w', encoding='utf-8') as f:
            f.write('33')
        s = (yield self.attachSVNPoller(sample_base, cachepath=cachepath))
        self.assertEqual(s.last_change, 33)
        s.last_change = 44
        s.finished_ok(None)
        with open(cachepath, encoding='utf-8') as f:
            self.assertEqual(f.read().strip(), '44')

    @defer.inlineCallbacks
    def test_cachepath_bogus(self):
        if False:
            print('Hello World!')
        cachepath = os.path.abspath('revcache')
        with open(cachepath, 'w', encoding='utf-8') as f:
            f.write('nine')
        s = (yield self.attachSVNPoller(sample_base, cachepath=cachepath))
        self.assertEqual(s.last_change, None)
        self.assertEqual(s.cachepath, None)
        self.assertEqual(len(self.flushLoggedErrors(ValueError)), 1)

    def test_constructor_pollinterval(self):
        if False:
            while True:
                i = 10
        return self.attachSVNPoller(sample_base, pollinterval=100)

    @defer.inlineCallbacks
    def test_extra_args(self):
        if False:
            for i in range(10):
                print('nop')
        extra_args = ['--no-auth-cache']
        base = 'svn+ssh://svn.twistedmatrix.com/svn/Twisted/trunk'
        s = (yield self.attachSVNPoller(repourl=base, extra_args=extra_args))
        self.assertEqual(s.extra_args, extra_args)

    @defer.inlineCallbacks
    def test_use_svnurl(self):
        if False:
            i = 10
            return i + 15
        base = 'svn+ssh://svn.twistedmatrix.com/svn/Twisted/trunk'
        with self.assertRaises(TypeError):
            yield self.attachSVNPoller(svnurl=base)

class TestSplitFile(unittest.TestCase):

    def test_split_file_alwaystrunk(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(svnpoller.split_file_alwaystrunk('foo'), {'path': 'foo'})

    def test_split_file_branches_trunk(self):
        if False:
            while True:
                i = 10
        self.assertEqual(svnpoller.split_file_branches('trunk/'), (None, ''))

    def test_split_file_branches_trunk_subdir(self):
        if False:
            return 10
        self.assertEqual(svnpoller.split_file_branches('trunk/subdir/'), (None, 'subdir/'))

    def test_split_file_branches_trunk_subfile(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(svnpoller.split_file_branches('trunk/subdir/file.c'), (None, 'subdir/file.c'))

    def test_split_file_branches_trunk_invalid(self):
        if False:
            return 10
        self.assertEqual(svnpoller.split_file_branches('trunk'), None)

    def test_split_file_branches_branch(self):
        if False:
            print('Hello World!')
        self.assertEqual(svnpoller.split_file_branches('branches/1.5.x/'), ('branches/1.5.x', ''))

    def test_split_file_branches_branch_subdir(self):
        if False:
            while True:
                i = 10
        self.assertEqual(svnpoller.split_file_branches('branches/1.5.x/subdir/'), ('branches/1.5.x', 'subdir/'))

    def test_split_file_branches_branch_subfile(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(svnpoller.split_file_branches('branches/1.5.x/subdir/file.c'), ('branches/1.5.x', 'subdir/file.c'))

    def test_split_file_branches_branch_invalid(self):
        if False:
            while True:
                i = 10
        self.assertEqual(svnpoller.split_file_branches('branches/1.5.x'), None)

    def test_split_file_branches_otherdir(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(svnpoller.split_file_branches('tags/testthis/subdir/'), None)

    def test_split_file_branches_otherfile(self):
        if False:
            while True:
                i = 10
        self.assertEqual(svnpoller.split_file_branches('tags/testthis/subdir/file.c'), None)

    def test_split_file_projects_branches(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(svnpoller.split_file_projects_branches('buildbot/trunk/subdir/file.c'), {'project': 'buildbot', 'path': 'subdir/file.c'})
        self.assertEqual(svnpoller.split_file_projects_branches('buildbot/branches/1.5.x/subdir/file.c'), {'project': 'buildbot', 'branch': 'branches/1.5.x', 'path': 'subdir/file.c'})
        self.assertEqual(svnpoller.split_file_projects_branches('buildbot/tags/testthis/subdir/file.c'), None)