from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process import results
from buildbot.steps.source import Source
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import TestBuildStepMixin
from buildbot.test.util import sourcesteps

class OldStyleSourceStep(Source):

    def startVC(self):
        if False:
            print('Hello World!')
        self.finished(results.SUCCESS)

class TestSource(sourcesteps.SourceStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tear_down_test_build_step()

    def setup_deferred_mock(self):
        if False:
            return 10
        m = mock.Mock()

        def wrapper(*args, **kwargs):
            if False:
                return 10
            m(*args, **kwargs)
            return results.SUCCESS
        wrapper.mock = m
        return wrapper

    def test_start_alwaysUseLatest_True(self):
        if False:
            while True:
                i = 10
        step = self.setup_step(Source(alwaysUseLatest=True), {'branch': 'other-branch', 'revision': 'revision'}, patch='patch')
        step.branch = 'branch'
        step.run_vc = self.setup_deferred_mock()
        step.startStep(mock.Mock())
        self.assertEqual(step.run_vc.mock.call_args, (('branch', None, None), {}))

    def test_start_alwaysUseLatest_False(self):
        if False:
            print('Hello World!')
        step = self.setup_step(Source(), {'branch': 'other-branch', 'revision': 'revision'}, patch='patch')
        step.branch = 'branch'
        step.run_vc = self.setup_deferred_mock()
        step.startStep(mock.Mock())
        self.assertEqual(step.run_vc.mock.call_args, (('other-branch', 'revision', 'patch'), {}))

    def test_start_alwaysUseLatest_False_binary_patch(self):
        if False:
            while True:
                i = 10
        args = {'branch': 'other-branch', 'revision': 'revision'}
        step = self.setup_step(Source(), args, patch=(1, b'patch\xf8'))
        step.branch = 'branch'
        step.run_vc = self.setup_deferred_mock()
        step.startStep(mock.Mock())
        self.assertEqual(step.run_vc.mock.call_args, (('other-branch', 'revision', (1, b'patch\xf8')), {}))

    def test_start_alwaysUseLatest_False_no_branch(self):
        if False:
            while True:
                i = 10
        step = self.setup_step(Source())
        step.branch = 'branch'
        step.run_vc = self.setup_deferred_mock()
        step.startStep(mock.Mock())
        self.assertEqual(step.run_vc.mock.call_args, (('branch', None, None), {}))

    def test_start_no_codebase(self):
        if False:
            i = 10
            return i + 15
        step = self.setup_step(Source())
        step.branch = 'branch'
        step.run_vc = self.setup_deferred_mock()
        step.build.getSourceStamp = mock.Mock()
        step.build.getSourceStamp.return_value = None
        self.assertEqual(step.getCurrentSummary(), {'step': 'updating'})
        self.assertEqual(step.name, Source.name)
        step.startStep(mock.Mock())
        self.assertEqual(step.build.getSourceStamp.call_args[0], ('',))
        self.assertEqual(step.getCurrentSummary(), {'step': 'updating'})

    @defer.inlineCallbacks
    def test_start_with_codebase(self):
        if False:
            i = 10
            return i + 15
        step = self.setup_step(Source(codebase='codebase'))
        step.branch = 'branch'
        step.run_vc = self.setup_deferred_mock()
        step.build.getSourceStamp = mock.Mock()
        step.build.getSourceStamp.return_value = None
        self.assertEqual(step.getCurrentSummary(), {'step': 'updating codebase'})
        step.name = (yield step.build.render(step.name))
        self.assertEqual(step.name, Source.name + '-codebase')
        step.startStep(mock.Mock())
        self.assertEqual(step.build.getSourceStamp.call_args[0], ('codebase',))
        self.assertEqual(step.getResultSummary(), {'step': 'Codebase codebase not in build codebase (failure)'})

    @defer.inlineCallbacks
    def test_start_with_codebase_and_descriptionSuffix(self):
        if False:
            i = 10
            return i + 15
        step = self.setup_step(Source(codebase='my-code', descriptionSuffix='suffix'))
        step.branch = 'branch'
        step.run_vc = self.setup_deferred_mock()
        step.build.getSourceStamp = mock.Mock()
        step.build.getSourceStamp.return_value = None
        self.assertEqual(step.getCurrentSummary(), {'step': 'updating suffix'})
        step.name = (yield step.build.render(step.name))
        self.assertEqual(step.name, Source.name + '-my-code')
        step.startStep(mock.Mock())
        self.assertEqual(step.build.getSourceStamp.call_args[0], ('my-code',))
        self.assertEqual(step.getResultSummary(), {'step': 'Codebase my-code not in build suffix (failure)'})

    def test_old_style_source_step_throws_exception(self):
        if False:
            i = 10
            return i + 15
        step = self.setup_step(OldStyleSourceStep())
        step.startStep(mock.Mock())
        self.expect_outcome(result=results.EXCEPTION)
        self.flushLoggedErrors(NotImplementedError)

class TestSourceDescription(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tear_down_test_build_step()

    def test_constructor_args_strings(self):
        if False:
            i = 10
            return i + 15
        step = Source(workdir='build', description='svn update (running)', descriptionDone='svn update')
        self.assertEqual(step.description, ['svn update (running)'])
        self.assertEqual(step.descriptionDone, ['svn update'])

    def test_constructor_args_lists(self):
        if False:
            i = 10
            return i + 15
        step = Source(workdir='build', description=['svn', 'update', '(running)'], descriptionDone=['svn', 'update'])
        self.assertEqual(step.description, ['svn', 'update', '(running)'])
        self.assertEqual(step.descriptionDone, ['svn', 'update'])

class AttrGroup(Source):

    def other_method(self):
        if False:
            return 10
        pass

    def mode_full(self):
        if False:
            print('Hello World!')
        pass

    def mode_incremental(self):
        if False:
            i = 10
            return i + 15
        pass

class TestSourceAttrGroup(sourcesteps.SourceStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tear_down_test_build_step()

    def test_attrgroup_hasattr(self):
        if False:
            return 10
        step = AttrGroup()
        self.assertTrue(step._hasAttrGroupMember('mode', 'full'))
        self.assertTrue(step._hasAttrGroupMember('mode', 'incremental'))
        self.assertFalse(step._hasAttrGroupMember('mode', 'nothing'))

    def test_attrgroup_getattr(self):
        if False:
            i = 10
            return i + 15
        step = AttrGroup()
        self.assertEqual(step._getAttrGroupMember('mode', 'full'), step.mode_full)
        self.assertEqual(step._getAttrGroupMember('mode', 'incremental'), step.mode_incremental)
        with self.assertRaises(AttributeError):
            step._getAttrGroupMember('mode', 'nothing')

    def test_attrgroup_listattr(self):
        if False:
            return 10
        step = AttrGroup()
        self.assertEqual(sorted(step._listAttrGroupMembers('mode')), ['full', 'incremental'])