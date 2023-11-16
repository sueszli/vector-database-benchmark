from twisted.trial.unittest import TestCase
from buildbot.process.workerforbuilder import AbstractWorkerForBuilder
from buildbot.worker.base import AbstractWorker

class TestAbstractWorkerForBuilder(TestCase):
    """
    Tests for ``AbstractWorkerForBuilder``.
    """

    def test_buildStarted_called(self):
        if False:
            while True:
                i = 10
        '\n        If the worker associated to worker builder has a ``buildStarted`` method,\n        calling ``buildStarted`` on the worker builder calls the method on the\n        worker with the workerforbuilder as an argument.\n        '

        class ConcreteWorker(AbstractWorker):
            _buildStartedCalls = []

            def buildStarted(self, workerforbuilder):
                if False:
                    i = 10
                    return i + 15
                self._buildStartedCalls.append(workerforbuilder)
        worker = ConcreteWorker('worker', 'pass')
        workerforbuilder = AbstractWorkerForBuilder()
        workerforbuilder.worker = worker
        workerforbuilder.buildStarted()
        self.assertEqual(ConcreteWorker._buildStartedCalls, [workerforbuilder])

    def test_buildStarted_missing(self):
        if False:
            print('Hello World!')
        "\n        If the worker associated to worker builder doesn't not have a\n        ``buildStarted`` method, calling ``buildStarted`` on the worker builder\n        doesn't raise an exception.\n        "

        class ConcreteWorker(AbstractWorker):
            pass
        worker = ConcreteWorker('worker', 'pass')
        workerforbuilder = AbstractWorkerForBuilder()
        workerforbuilder.worker = worker
        workerforbuilder.buildStarted()