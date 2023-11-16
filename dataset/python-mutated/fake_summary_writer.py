"""Fake summary writer for unit tests."""
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache

class FakeSummaryWriter(object):
    """Fake summary writer."""
    _replaced_summary_writer = None

    @classmethod
    def install(cls):
        if False:
            while True:
                i = 10
        if cls._replaced_summary_writer:
            raise ValueError('FakeSummaryWriter already installed.')
        cls._replaced_summary_writer = writer.FileWriter
        writer.FileWriter = FakeSummaryWriter
        writer_cache.FileWriter = FakeSummaryWriter

    @classmethod
    def uninstall(cls):
        if False:
            return 10
        if not cls._replaced_summary_writer:
            raise ValueError('FakeSummaryWriter not installed.')
        writer.FileWriter = cls._replaced_summary_writer
        writer_cache.FileWriter = cls._replaced_summary_writer
        cls._replaced_summary_writer = None

    def __init__(self, logdir, graph=None):
        if False:
            return 10
        self._logdir = logdir
        self._graph = graph
        self._summaries = {}
        self._added_graphs = []
        self._added_meta_graphs = []
        self._added_session_logs = []
        self._added_run_metadata = {}

    @property
    def summaries(self):
        if False:
            print('Hello World!')
        return self._summaries

    def assert_summaries(self, test_case, expected_logdir=None, expected_graph=None, expected_summaries=None, expected_added_graphs=None, expected_added_meta_graphs=None, expected_session_logs=None):
        if False:
            while True:
                i = 10
        'Assert expected items have been added to summary writer.'
        if expected_logdir is not None:
            test_case.assertEqual(expected_logdir, self._logdir)
        if expected_graph is not None:
            test_case.assertTrue(expected_graph is self._graph)
        expected_summaries = expected_summaries or {}
        for step in expected_summaries:
            test_case.assertTrue(step in self._summaries, msg='Missing step %s from %s.' % (step, self._summaries.keys()))
            actual_simple_values = {}
            for step_summary in self._summaries[step]:
                for v in step_summary.value:
                    if 'global_step/sec' != v.tag:
                        actual_simple_values[v.tag] = v.simple_value
            test_case.assertEqual(expected_summaries[step], actual_simple_values)
        if expected_added_graphs is not None:
            test_case.assertEqual(expected_added_graphs, self._added_graphs)
        if expected_added_meta_graphs is not None:
            test_case.assertEqual(len(expected_added_meta_graphs), len(self._added_meta_graphs))
            for (expected, actual) in zip(expected_added_meta_graphs, self._added_meta_graphs):
                test_util.assert_meta_graph_protos_equal(test_case, expected, actual)
        if expected_session_logs is not None:
            test_case.assertEqual(expected_session_logs, self._added_session_logs)

    def add_summary(self, summ, current_global_step):
        if False:
            return 10
        'Add summary.'
        if isinstance(summ, bytes):
            summary_proto = summary_pb2.Summary()
            summary_proto.ParseFromString(summ)
            summ = summary_proto
        if current_global_step in self._summaries:
            step_summaries = self._summaries[current_global_step]
        else:
            step_summaries = []
            self._summaries[current_global_step] = step_summaries
        step_summaries.append(summ)

    def add_graph(self, graph, global_step=None, graph_def=None):
        if False:
            return 10
        'Add graph.'
        if global_step is not None and global_step < 0:
            raise ValueError('Invalid global_step %s.' % global_step)
        if graph_def is not None:
            raise ValueError('Unexpected graph_def %s.' % graph_def)
        self._added_graphs.append(graph)

    def add_meta_graph(self, meta_graph_def, global_step=None):
        if False:
            return 10
        'Add metagraph.'
        if global_step is not None and global_step < 0:
            raise ValueError('Invalid global_step %s.' % global_step)
        self._added_meta_graphs.append(meta_graph_def)

    def add_session_log(self, session_log, global_step=None):
        if False:
            i = 10
            return i + 15
        self._added_session_logs.append(session_log)

    def add_run_metadata(self, run_metadata, tag, global_step=None):
        if False:
            i = 10
            return i + 15
        if global_step is not None and global_step < 0:
            raise ValueError('Invalid global_step %s.' % global_step)
        self._added_run_metadata[tag] = run_metadata

    def flush(self):
        if False:
            print('Hello World!')
        pass

    def reopen(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass