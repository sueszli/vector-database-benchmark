import testslide
from ...language_server import daemon_connection
from ..daemon_query import DaemonQueryFailure
from ..daemon_query_failer import DaemonQueryNoOpFailer, RegexDaemonQueryFailer

class DaemonQueryAutoFailerTest(testslide.TestCase):

    def test_noop(self) -> None:
        if False:
            return 10
        self.assertIsNone(DaemonQueryNoOpFailer().query_failure('/path'))

class DaemonQueryAutoFailerPatternTest(testslide.TestCase):

    def test_passes(self) -> None:
        if False:
            i = 10
            return i + 15
        daemonQueryAutoFailerPattern = RegexDaemonQueryFailer('/a/b/c/*')
        self.assertIsNone(daemonQueryAutoFailerPattern.query_failure('/path'))
        self.assertIsNone(daemonQueryAutoFailerPattern.query_failure('/a/b/something.py'))
        self.assertIsNone(daemonQueryAutoFailerPattern.query_failure('/a/b/something/something.py'))
        self.assertIsNone(daemonQueryAutoFailerPattern.query_connection_failure('/path'))
        self.assertIsNone(daemonQueryAutoFailerPattern.query_connection_failure('/a/b/something.py'))
        self.assertIsNone(daemonQueryAutoFailerPattern.query_connection_failure('/a/b/something/something.py'))

    def test_rejects(self) -> None:
        if False:
            while True:
                i = 10
        daemonQueryAutoFailerPattern = RegexDaemonQueryFailer('/a/b/c/*')
        self.assertEqual(daemonQueryAutoFailerPattern.query_failure('/a/b/c/something.py'), DaemonQueryFailure('Not querying daemon for path: /a/b/c/something.py as matches regex: /a/b/c/*'))
        self.assertEqual(daemonQueryAutoFailerPattern.query_failure('/a/b/c/something/something.py'), DaemonQueryFailure('Not querying daemon for path: /a/b/c/something/something.py as matches regex: /a/b/c/*'))
        self.assertEqual(daemonQueryAutoFailerPattern.query_connection_failure('/a/b/c/something.py'), daemon_connection.DaemonConnectionFailure('Not querying daemon for path: /a/b/c/something.py as matches regex: /a/b/c/*'))
        self.assertEqual(daemonQueryAutoFailerPattern.query_connection_failure('/a/b/c/something/something.py'), daemon_connection.DaemonConnectionFailure('Not querying daemon for path: /a/b/c/something/something.py as matches regex: /a/b/c/*'))