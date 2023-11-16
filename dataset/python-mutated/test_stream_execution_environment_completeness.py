from pyflink.datastream import StreamExecutionEnvironment
from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase

class StreamExecutionEnvironmentCompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):

    @classmethod
    def python_class(cls):
        if False:
            for i in range(10):
                print('nop')
        return StreamExecutionEnvironment

    @classmethod
    def java_class(cls):
        if False:
            return 10
        return 'org.apache.flink.streaming.api.environment.StreamExecutionEnvironment'

    @classmethod
    def excluded_methods(cls):
        if False:
            i = 10
            return i + 15
        return {'getLastJobExecutionResult', 'getId', 'getIdString', 'createCollectionsEnvironment', 'createLocalEnvironment', 'createRemoteEnvironment', 'addOperator', 'fromElements', 'resetContextEnvironment', 'getCachedFiles', 'generateSequence', 'getNumberOfExecutionRetries', 'getStreamGraph', 'fromParallelCollection', 'readFileStream', 'isForceCheckpointing', 'readFile', 'clean', 'createInput', 'createLocalEnvironmentWithWebUI', 'fromCollection', 'socketTextStream', 'initializeContextEnvironment', 'readTextFile', 'setNumberOfExecutionRetries', 'executeAsync', 'registerJobListener', 'clearJobListeners', 'getJobListeners', 'fromSequence', 'getConfiguration', 'generateStreamGraph', 'getTransformations', 'areExplicitEnvironmentsAllowed', 'registerCollectIterator', 'listCompletedClusterDatasets', 'invalidateClusterDataset', 'registerCacheTransformation', 'close'}
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)