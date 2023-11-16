from unittest import mock
from buildbot.test.steps import TestBuildStepMixin

class SourceStepMixin(TestBuildStepMixin):
    """
    Support for testing source steps.  Aside from the capabilities of
    L{TestBuildStepMixin}, this adds:

     - fake sourcestamps

    The following instance variables are available after C{setupSourceStep}, in
    addition to those made available by L{TestBuildStepMixin}:

    @ivar sourcestamp: fake SourceStamp for the build
    """

    def setUpSourceStep(self):
        if False:
            print('Hello World!')
        return super().setup_test_build_step()

    def tearDownSourceStep(self):
        if False:
            return 10
        return super().tear_down_test_build_step()

    def setup_step(self, step, args=None, patch=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Set up C{step} for testing.  This calls L{TestBuildStepMixin}'s C{setup_step}\n        and then does setup specific to a Source step.\n        "
        step = super().setup_step(step, **kwargs)
        if args is None:
            args = {}
        ss = self.sourcestamp = mock.Mock(name='sourcestamp')
        ss.ssid = 9123
        ss.branch = args.get('branch', None)
        ss.revision = args.get('revision', None)
        ss.project = ''
        ss.repository = ''
        ss.patch = patch
        ss.patch_info = None
        ss.changes = []
        self.build.getSourceStamp = lambda x=None: ss
        return step