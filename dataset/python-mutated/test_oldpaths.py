import warnings
from twisted.trial import unittest
from buildbot.warnings import DeprecatedApiWarning

def deprecatedImport(fn):
    if False:
        while True:
            i = 10

    def wrapper(self):
        if False:
            i = 10
            return i + 15
        fn(self)
        warnings = self.flushWarnings()
        if len(warnings) == 2 and warnings[0] == warnings[1]:
            del warnings[1]
        self.assertEqual(len(warnings), 1, f'got: {repr(warnings)}')
        self.assertEqual(warnings[0]['category'], DeprecatedApiWarning)
    return wrapper

class OldImportPaths(unittest.TestCase):
    """
    Test that old, deprecated import paths still work.
    """

    def test_scheduler_Scheduler(self):
        if False:
            i = 10
            return i + 15
        from buildbot.scheduler import Scheduler
        assert Scheduler

    def test_schedulers_basic_Scheduler(self):
        if False:
            i = 10
            return i + 15
        from buildbot.schedulers.basic import Scheduler
        assert Scheduler

    def test_scheduler_AnyBranchScheduler(self):
        if False:
            while True:
                i = 10
        from buildbot.scheduler import AnyBranchScheduler
        assert AnyBranchScheduler

    def test_scheduler_basic_Dependent(self):
        if False:
            while True:
                i = 10
        from buildbot.schedulers.basic import Dependent
        assert Dependent

    def test_scheduler_Dependent(self):
        if False:
            for i in range(10):
                print('nop')
        from buildbot.scheduler import Dependent
        assert Dependent

    def test_scheduler_Periodic(self):
        if False:
            i = 10
            return i + 15
        from buildbot.scheduler import Periodic
        assert Periodic

    def test_scheduler_Nightly(self):
        if False:
            i = 10
            return i + 15
        from buildbot.scheduler import Nightly
        assert Nightly

    def test_scheduler_Triggerable(self):
        if False:
            for i in range(10):
                print('nop')
        from buildbot.scheduler import Triggerable
        assert Triggerable

    def test_scheduler_Try_Jobdir(self):
        if False:
            for i in range(10):
                print('nop')
        from buildbot.scheduler import Try_Jobdir
        assert Try_Jobdir

    def test_scheduler_Try_Userpass(self):
        if False:
            print('Hello World!')
        from buildbot.scheduler import Try_Userpass
        assert Try_Userpass

    def test_schedulers_filter_ChangeFilter(self):
        if False:
            i = 10
            return i + 15
        from buildbot.schedulers.filter import ChangeFilter
        assert ChangeFilter

    def test_process_base_Build(self):
        if False:
            print('Hello World!')
        from buildbot.process.base import Build
        assert Build

    def test_buildrequest_BuildRequest(self):
        if False:
            print('Hello World!')
        from buildbot.buildrequest import BuildRequest
        assert BuildRequest

    def test_process_subunitlogobserver_SubunitShellCommand(self):
        if False:
            return 10
        from buildbot.process.subunitlogobserver import SubunitShellCommand
        assert SubunitShellCommand

    def test_steps_source_Source(self):
        if False:
            print('Hello World!')
        from buildbot.steps.source import Source
        assert Source

    def test_buildstep_remotecommand(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecatedApiWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
            from buildbot.process.buildstep import LoggedRemoteCommand
            from buildbot.process.buildstep import RemoteCommand
            from buildbot.process.buildstep import RemoteShellCommand
            assert RemoteCommand
            assert LoggedRemoteCommand
            assert RemoteShellCommand

    def test_buildstep_logobserver(self):
        if False:
            return 10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecatedApiWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
            from buildbot.process.buildstep import LogLineObserver
            from buildbot.process.buildstep import LogObserver
            from buildbot.process.buildstep import OutputProgressObserver
        assert LogObserver
        assert LogLineObserver
        assert OutputProgressObserver