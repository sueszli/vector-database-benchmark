import pytest
from sacred import Experiment
from sacred.stflow import LogFileWriter
import sacred.optional as opt

@pytest.fixture
def ex():
    if False:
        return 10
    return Experiment('tensorflow_tests')

@pytest.fixture()
def tf():
    if False:
        i = 10
        return i + 15
    '\n    Creates a simplified tensorflow interface if necessary,\n    so `tensorflow` is not required during the tests.\n    '
    from sacred.optional import has_tensorflow
    if has_tensorflow:
        return opt.get_tensorflow()
    else:

        class tensorflow:

            class summary:

                class FileWriter:

                    def __init__(self, logdir, graph):
                        if False:
                            while True:
                                i = 10
                        self.logdir = logdir
                        self.graph = graph
                        print('Mocked FileWriter got logdir=%s, graph=%s' % (logdir, graph))

            class Session:

                def __init__(self):
                    if False:
                        i = 10
                        return i + 15
                    self.graph = None

                def __enter__(self):
                    if False:
                        return 10
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if False:
                        print('Hello World!')
                    pass
        import sacred.stflow.method_interception
        sacred.stflow.method_interception.tf = tensorflow
        return tensorflow

def test_log_file_writer(ex, tf):
    if False:
        print('Hello World!')
    '\n    Tests whether logdir is stored into the info dictionary when creating a new FileWriter object.\n    '
    TEST_LOG_DIR = '/tmp/sacred_test1'
    TEST_LOG_DIR2 = '/tmp/sacred_test2'

    @ex.main
    @LogFileWriter(ex)
    def run_experiment(_run):
        if False:
            return 10
        assert _run.info.get('tensorflow', None) is None
        with tf.Session() as s:
            with LogFileWriter(ex):
                swr = tf.summary.FileWriter(logdir=TEST_LOG_DIR, graph=s.graph)
            assert swr is not None
            assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR]
            tf.summary.FileWriter(TEST_LOG_DIR2, s.graph)
            assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR, TEST_LOG_DIR2]
    ex.run()

def test_log_summary_writer_as_context_manager(ex, tf):
    if False:
        i = 10
        return i + 15
    '\n    Check that Tensorflow log directory is captured by LogFileWriter context manager.\n    '
    TEST_LOG_DIR = '/tmp/sacred_test1'
    TEST_LOG_DIR2 = '/tmp/sacred_test2'

    @ex.main
    def run_experiment(_run):
        if False:
            i = 10
            return i + 15
        assert _run.info.get('tensorflow', None) is None
        with tf.Session() as s:
            swr = tf.summary.FileWriter(logdir=TEST_LOG_DIR, graph=s.graph)
            assert swr is not None
            assert _run.info.get('tensorflow', None) is None
            with LogFileWriter(ex):
                swr = tf.summary.FileWriter(logdir=TEST_LOG_DIR, graph=s.graph)
                assert swr is not None
                assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR]
                tf.summary.FileWriter(TEST_LOG_DIR2, s.graph)
                assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR, TEST_LOG_DIR2]
            tf.summary.FileWriter('/tmp/whatever', s.graph)
            assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR, TEST_LOG_DIR2]
    ex.run()

def test_log_file_writer_as_context_manager_with_exception(ex, tf):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that Tensorflow log directory is captured by LogFileWriter context manager.\n    '
    TEST_LOG_DIR = '/tmp/sacred_test'

    @ex.main
    def run_experiment(_run):
        if False:
            return 10
        assert _run.info.get('tensorflow', None) is None
        with tf.Session() as s:
            try:
                with LogFileWriter(ex):
                    swr = tf.summary.FileWriter(logdir=TEST_LOG_DIR, graph=s.graph)
                    assert swr is not None
                    assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR]
                    raise ValueError('I want to be raised!')
            except ValueError:
                pass
            tf.summary.FileWriter('/tmp/whatever', s.graph)
            assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR]
    ex.run()

def test_log_summary_writer_class(ex, tf):
    if False:
        return 10
    '\n    Tests whether logdir is stored into the info dictionary when creating a new FileWriter object,\n    but this time on a method of a class.\n    '
    TEST_LOG_DIR = '/tmp/sacred_test1'
    TEST_LOG_DIR2 = '/tmp/sacred_test2'

    class FooClass:

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass

        @LogFileWriter(ex)
        def hello(self, argument):
            if False:
                i = 10
                return i + 15
            with tf.Session() as s:
                tf.summary.FileWriter(argument, s.graph)

    @ex.main
    def run_experiment(_run):
        if False:
            for i in range(10):
                print('nop')
        assert _run.info.get('tensorflow', None) is None
        foo = FooClass()
        with tf.Session() as s:
            swr = tf.summary.FileWriter(TEST_LOG_DIR, s.graph)
            assert swr is not None
            assert _run.info.get('tensorflow', None) is None
        foo.hello(TEST_LOG_DIR2)
        assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR2]
        with tf.Session() as s:
            swr = tf.summary.FileWriter(TEST_LOG_DIR, s.graph)
            assert _run.info['tensorflow']['logdirs'] == [TEST_LOG_DIR2]
    ex.run()
if __name__ == '__main__':
    test_log_file_writer(ex(), tf())