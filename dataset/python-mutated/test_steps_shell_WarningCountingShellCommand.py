import re
from twisted.trial import unittest
from buildbot.steps.shell import WarningCountingShellCommand

class TestWarningCountingShellCommand(unittest.TestCase):

    def testSuppressingLinelessWarningsPossible(self):
        if False:
            while True:
                i = 10
        w = WarningCountingShellCommand(warningExtractor=WarningCountingShellCommand.warnExtractWholeLine, command='echo')
        fileRe = None
        warnRe = '.*SUPPRESS.*'
        start = None
        end = None
        suppression = (fileRe, warnRe, start, end)
        w.addSuppression([suppression])
        warnings = []
        line = 'this warning should be SUPPRESSed'
        match = re.match('.*warning.*', line)
        w.maybeAddWarning(warnings, line, match)
        expectedWarnings = 0
        self.assertEqual(len(warnings), expectedWarnings)