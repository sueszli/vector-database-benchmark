"""UnitTests for DoFn lifecycle and bundle methods"""
import unittest
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline

class CallSequenceEnforcingDoFn(beam.DoFn):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._setup_called = False
        self._start_bundle_calls = 0
        self._finish_bundle_calls = 0
        self._teardown_called = False

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self._setup_called, 'setup should not be called twice'
        assert self._start_bundle_calls == 0, 'setup should be called before start_bundle'
        assert self._finish_bundle_calls == 0, 'setup should be called before finish_bundle'
        assert not self._teardown_called, 'setup should be called before teardown'
        self._setup_called = True

    def start_bundle(self):
        if False:
            print('Hello World!')
        assert self._setup_called, 'setup should have been called'
        assert self._start_bundle_calls == self._finish_bundle_calls, 'there should be as many start_bundle calls as finish_bundle calls'
        assert not self._teardown_called, 'teardown should not have been called'
        self._start_bundle_calls += 1

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        assert self._setup_called, 'setup should have been called'
        assert self._start_bundle_calls > 0, 'start_bundle should have been called'
        assert self._start_bundle_calls == self._finish_bundle_calls + 1, 'there should be one start_bundle call with no call to finish_bundle'
        assert not self._teardown_called, 'teardown should not have been called'
        return [element * element]

    def finish_bundle(self):
        if False:
            while True:
                i = 10
        assert self._setup_called, 'setup should have been called'
        assert self._start_bundle_calls > 0, 'start_bundle should have been called'
        assert self._start_bundle_calls == self._finish_bundle_calls + 1, 'there should be one start_bundle call with no call to finish_bundle'
        assert not self._teardown_called, 'teardown should not have been called'
        self._finish_bundle_calls += 1

    def teardown(self):
        if False:
            print('Hello World!')
        assert self._setup_called, 'setup should have been called'
        assert self._start_bundle_calls == self._finish_bundle_calls, 'there should be as many start_bundle calls as finish_bundle calls'
        assert not self._teardown_called, 'teardown should not be called twice'
        self._teardown_called = True

@pytest.mark.it_validatesrunner
class DoFnLifecycleTest(unittest.TestCase):

    def test_dofn_lifecycle(self):
        if False:
            return 10
        with TestPipeline() as p:
            _ = p | 'Start' >> beam.Create([1, 2, 3]) | 'Do' >> beam.ParDo(CallSequenceEnforcingDoFn())

class LocalDoFnLifecycleTest(unittest.TestCase):

    def test_dofn_lifecycle(self):
        if False:
            print('Hello World!')
        from apache_beam.runners.direct import direct_runner
        from apache_beam.runners.portability import fn_api_runner
        runners = [direct_runner.BundleBasedDirectRunner(), fn_api_runner.FnApiRunner()]
        for r in runners:
            with TestPipeline(runner=r) as p:
                _ = p | 'Start' >> beam.Create([1, 2, 3]) | 'Do' >> beam.ParDo(CallSequenceEnforcingDoFn())
if __name__ == '__main__':
    unittest.main()