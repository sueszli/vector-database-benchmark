"""Model Analyzer.

Analyze model, including shape, params, time, memory, structure, etc.
"""
import sys
from google.protobuf import message
from tensorflow.core.profiler import tfprof_options_pb2
from tensorflow.core.profiler import tfprof_output_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util.tf_export import tf_export
_DEFAULT_PROFILE_OPTIONS = 0
_DEFAULT_ADVISE_OPTIONS = 0
ALL_ADVICE = {'ExpensiveOperationChecker': {}, 'AcceleratorUtilizationChecker': {}, 'JobChecker': {}, 'OperationChecker': {}}

def _graph_string(graph):
    if False:
        for i in range(10):
            print('nop')
    'Helper to serialize a graph to string.'
    if graph:
        return graph.as_graph_def(add_shapes=True).SerializeToString()
    else:
        return b''

def _build_options(options):
    if False:
        print('Hello World!')
    'Build tfprof.OptionsProto.\n\n  Args:\n    options: A dictionary of options.\n\n  Returns:\n    tfprof.OptionsProto.\n  '
    opts = tfprof_options_pb2.OptionsProto()
    opts.max_depth = options.get('max_depth', 10)
    opts.min_bytes = options.get('min_bytes', 0)
    opts.min_peak_bytes = options.get('min_peak_bytes', 0)
    opts.min_residual_bytes = options.get('min_residual_bytes', 0)
    opts.min_output_bytes = options.get('min_output_bytes', 0)
    opts.min_micros = options.get('min_micros', 0)
    opts.min_accelerator_micros = options.get('min_accelerator_micros', 0)
    opts.min_cpu_micros = options.get('min_cpu_micros', 0)
    opts.min_params = options.get('min_params', 0)
    opts.min_float_ops = options.get('min_float_ops', 0)
    opts.min_occurrence = options.get('min_occurrence', 0)
    opts.step = options.get('step', -1)
    opts.order_by = options.get('order_by', 'name')
    for p in options.get('account_type_regexes', []):
        opts.account_type_regexes.append(p)
    for p in options.get('start_name_regexes', []):
        opts.start_name_regexes.append(p)
    for p in options.get('trim_name_regexes', []):
        opts.trim_name_regexes.append(p)
    for p in options.get('show_name_regexes', []):
        opts.show_name_regexes.append(p)
    for p in options.get('hide_name_regexes', []):
        opts.hide_name_regexes.append(p)
    opts.account_displayed_op_only = options.get('account_displayed_op_only', False)
    for p in options.get('select', []):
        opts.select.append(p)
    opts.output = options.get('output', 'stdout')
    opts.dump_to_file = options.get('dump_to_file', '')
    return opts

def _build_advisor_options(options):
    if False:
        for i in range(10):
            print('nop')
    'Build tfprof.AdvisorOptionsProto.\n\n  Args:\n    options: A dictionary of options. See ALL_ADVICE example.\n\n  Returns:\n    tfprof.AdvisorOptionsProto.\n  '
    opts = tfprof_options_pb2.AdvisorOptionsProto()
    if options is None:
        return opts
    for (checker, checker_opts) in options.items():
        checker_ops_pb = tfprof_options_pb2.AdvisorOptionsProto.CheckerOption()
        for (k, v) in checker_opts.items():
            checker_ops_pb[k] = v
        opts.checkers[checker].MergeFrom(checker_ops_pb)
    return opts

@tf_export(v1=['profiler.Profiler'])
class Profiler:
    """TensorFlow multi-step profiler.


  ```python
  Typical use case:
    # Currently we are only allowed to create 1 profiler per process.
    profiler = Profiler(sess.graph)

    for i in range(total_steps):
      if i % 10000 == 0:
        run_meta = tf.compat.v1.RunMetadata()
        _ = sess.run(...,
                     options=tf.compat.v1.RunOptions(
                         trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_meta)
        profiler.add_step(i, run_meta)

        # Profile the parameters of your model.
        profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
            .trainable_variables_parameter()))

        # Or profile the timing of your model operations.
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = (option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(i)
                .with_timeline_output(filename).build())
        profiler.profile_graph(options=opts)
      else:
        _ = sess.run(...)
    # Auto detect problems and generate advice.
    profiler.advise()
  ```
  """

    def __init__(self, graph=None, op_log=None):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      graph: tf.Graph. If None and eager execution is not enabled, use default\n        graph.\n      op_log: optional. tensorflow::tfprof::OpLogProto proto. Used to define\n        extra op types.\n    '
        if not graph and (not context.executing_eagerly()):
            graph = ops.get_default_graph()
        self._coverage = 0.0
        self._graph = graph
        op_log = tfprof_logger.merge_default_with_oplog(self._graph, op_log=op_log)
        print_mdl.NewProfiler(_graph_string(self._graph), op_log.SerializeToString())

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        print_mdl.DeleteProfiler()

    def add_step(self, step, run_meta):
        if False:
            print('Hello World!')
        'Add statistics of a step.\n\n    Args:\n      step: int, An id used to group one or more different `run_meta` together.\n        When profiling with the profile_xxx APIs, user can use the `step` id in\n        the `options` to profile these `run_meta` together.\n      run_meta: RunMetadata proto that contains statistics of a session run.\n    '
        op_log = tfprof_logger.merge_default_with_oplog(self._graph, run_meta=run_meta)
        self._coverage = print_mdl.AddStep(step, _graph_string(self._graph), run_meta.SerializeToString(), op_log.SerializeToString())

    def profile_python(self, options):
        if False:
            for i in range(10):
                print('nop')
        "Profile the statistics of the Python codes.\n\n      By default, it shows the call stack from root. To avoid\n      redundant output, you may use options to filter as below\n        options['show_name_regexes'] = ['.*my_code.py.*']\n\n    Args:\n      options: A dict of options. See core/profiler/g3doc/options.md.\n\n    Returns:\n      a MultiGraphNodeProto that records the results.\n    "
        opts = _build_options(options)
        tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
        try:
            tfprof_node.ParseFromString(print_mdl.Profile('code'.encode('utf-8'), opts.SerializeToString()))
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
        return tfprof_node

    def profile_operations(self, options):
        if False:
            print('Hello World!')
        'Profile the statistics of the Operation types (e.g.\n\n    MatMul, Conv2D).\n\n    Args:\n      options: A dict of options. See core/profiler/g3doc/options.md.\n\n    Returns:\n      a MultiGraphNodeProto that records the results.\n    '
        opts = _build_options(options)
        tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
        try:
            tfprof_node.ParseFromString(print_mdl.Profile('op'.encode('utf-8'), opts.SerializeToString()))
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
        return tfprof_node

    def profile_name_scope(self, options):
        if False:
            i = 10
            return i + 15
        'Profile the statistics of graph nodes, organized by name scope.\n\n    Args:\n      options: A dict of options. See core/profiler/g3doc/options.md.\n\n    Returns:\n      a GraphNodeProto that records the results.\n    '
        opts = _build_options(options)
        tfprof_node = tfprof_output_pb2.GraphNodeProto()
        try:
            tfprof_node.ParseFromString(print_mdl.Profile('scope'.encode('utf-8'), opts.SerializeToString()))
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
        return tfprof_node

    def profile_graph(self, options):
        if False:
            print('Hello World!')
        'Profile the statistics of graph nodes, organized by dataflow graph.\n\n    Args:\n      options: A dict of options. See core/profiler/g3doc/options.md.\n\n    Returns:\n      a GraphNodeProto that records the results.\n    '
        opts = _build_options(options)
        tfprof_node = tfprof_output_pb2.GraphNodeProto()
        try:
            tfprof_node.ParseFromString(print_mdl.Profile('graph'.encode('utf-8'), opts.SerializeToString()))
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
        return tfprof_node

    def advise(self, options):
        if False:
            while True:
                i = 10
        'Automatically detect problems and generate reports.\n\n    Args:\n      options: A dict of options. See ALL_ADVICE example above.\n\n    Returns:\n      An Advise proto that contains the reports from all checkers.\n    '
        advise_pb = tfprof_output_pb2.AdviceProto()
        opts = _build_advisor_options(options)
        advise_pb.ParseFromString(print_mdl.Profile('advise'.encode('utf-8'), opts.SerializeToString()))
        return advise_pb

    def serialize_to_string(self):
        if False:
            while True:
                i = 10
        'Serialize the ProfileProto to a binary string.\n\n      Users can write it to file for offline analysis by tfprof commandline\n      or graphical interface.\n\n    Returns:\n      ProfileProto binary string.\n    '
        return print_mdl.SerializeToString()

    def _write_profile(self, filename):
        if False:
            while True:
                i = 10
        'Writes the profile to a file.'
        print_mdl.WriteProfile(filename)

@tf_export(v1=['profiler.profile'])
def profile(graph=None, run_meta=None, op_log=None, cmd='scope', options=_DEFAULT_PROFILE_OPTIONS):
    if False:
        while True:
            i = 10
    'Profile model.\n\n    Tutorials and examples can be found in:\n    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md\n\n  Args:\n    graph: tf.Graph. If None and eager execution is not enabled, use default\n      graph.\n    run_meta: optional tensorflow.RunMetadata proto. It is necessary to\n      support run time information profiling, such as time and memory.\n    op_log: tensorflow.tfprof.OpLogProto proto. User can assign "types" to graph\n      nodes with op_log. "types" allow user to flexibly group and account\n      profiles using options[\'accounted_type_regexes\'].\n    cmd: string. Either \'op\', \'scope\', \'graph\' or \'code\'. \'op\' view organizes\n      profile using operation type. (e.g. MatMul) \'scope\' view organizes profile\n      using graph node name scope. \'graph\' view organizes profile using graph\n      node inputs/outputs. \'code\' view organizes profile using Python call\n      stack.\n    options: A dict of options. See core/profiler/g3doc/options.md.\n\n  Returns:\n    If cmd is \'scope\' or \'graph\', returns GraphNodeProto proto.\n    If cmd is \'op\' or \'code\', returns MultiGraphNodeProto proto.\n    Side effect: stdout/file/timeline.json depending on options[\'output\']\n  '
    if not graph and (not context.executing_eagerly()):
        graph = ops.get_default_graph()
    if options == _DEFAULT_PROFILE_OPTIONS:
        options = option_builder.ProfileOptionBuilder.trainable_variables_parameter()
    op_log = tfprof_logger.merge_default_with_oplog(graph, op_log, run_meta, add_trace=cmd == 'code')
    opts = _build_options(options)
    run_meta_str = run_meta.SerializeToString() if run_meta else b''
    graph_str = _graph_string(graph)
    if cmd == 'code' or cmd == 'op':
        tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
        ret = print_mdl.PrintModelAnalysis(graph_str, run_meta_str, op_log.SerializeToString(), cmd.encode('utf-8'), opts.SerializeToString())
        try:
            tfprof_node.ParseFromString(ret)
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    elif cmd == 'graph' or cmd == 'scope':
        tfprof_node = tfprof_output_pb2.GraphNodeProto()
        ret = print_mdl.PrintModelAnalysis(graph_str, run_meta_str, op_log.SerializeToString(), cmd.encode('utf-8'), opts.SerializeToString())
        try:
            tfprof_node.ParseFromString(ret)
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    else:
        raise errors.InvalidArgumentError(None, None, 'unknown cmd: %s\n' % cmd)
    return tfprof_node

@tf_export(v1=['profiler.advise'])
def advise(graph=None, run_meta=None, options=_DEFAULT_ADVISE_OPTIONS):
    if False:
        while True:
            i = 10
    'Auto profile and advise.\n\n    Builds profiles and automatically check anomalies of various\n    aspects. For more details:\n    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/README.md\n\n  Args:\n    graph: tf.Graph. If None and eager execution is not enabled, use default\n      graph.\n    run_meta: optional tensorflow.RunMetadata proto. It is necessary to\n      support run time information profiling, such as time and memory.\n    options: see ALL_ADVICE example above. Default checks everything.\n\n  Returns:\n    Returns AdviceProto proto\n  '
    if not graph and (not context.executing_eagerly()):
        graph = ops.get_default_graph()
    if options == _DEFAULT_ADVISE_OPTIONS:
        options = ALL_ADVICE.copy()
    op_log = tfprof_logger.merge_default_with_oplog(graph, None, run_meta, add_trace=True)
    run_meta_str = run_meta.SerializeToString() if run_meta else b''
    opts = _build_advisor_options(options)
    ret = tfprof_output_pb2.AdviceProto()
    ret.ParseFromString(print_mdl.PrintModelAnalysis(_graph_string(graph), run_meta_str, op_log.SerializeToString(), 'advise'.encode('utf-8'), opts.SerializeToString()))
    return ret