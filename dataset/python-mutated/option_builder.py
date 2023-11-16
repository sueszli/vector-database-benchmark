"""Utilities for building profiler options."""
import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['profiler.ProfileOptionBuilder'])
class ProfileOptionBuilder(object):
    """Option Builder for Profiling API.

  For tutorial on the options, see
  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md

  ```python
  # Users can use pre-built options:
  opts = (
      tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

  # Or, build your own options:
  opts = (tf.compat.v1.profiler.ProfileOptionBuilder()
      .with_max_depth(10)
      .with_min_micros(1000)
      .select(['accelerator_micros'])
      .with_stdout_output()
      .build()

  # Or customize the pre-built options:
  opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
      tf.profiler.ProfileOptionBuilder.time_and_memory())
      .with_displaying_options(show_name_regexes=['.*rnn.*'])
      .build())

  # Finally, profiling with the options:
  _ = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(),
                          run_meta=run_meta,
                          cmd='scope',
                          options=opts)
  ```
  """

    def __init__(self, options=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n    Args:\n      options: Optional initial option dict to start with.\n    '
        if options is not None:
            self._options = copy.deepcopy(options)
        else:
            self._options = {'max_depth': 100, 'min_bytes': 0, 'min_micros': 0, 'min_params': 0, 'min_float_ops': 0, 'min_occurrence': 0, 'order_by': 'name', 'account_type_regexes': ['.*'], 'start_name_regexes': ['.*'], 'trim_name_regexes': [], 'show_name_regexes': ['.*'], 'hide_name_regexes': [], 'account_displayed_op_only': False, 'select': ['micros'], 'step': -1, 'output': 'stdout'}

    @staticmethod
    def trainable_variables_parameter():
        if False:
            i = 10
            return i + 15
        "Options used to profile trainable variable parameters.\n\n    Normally used together with 'scope' view.\n\n    Returns:\n      A dict of profiling options.\n    "
        return {'max_depth': 10000, 'min_bytes': 0, 'min_micros': 0, 'min_params': 0, 'min_float_ops': 0, 'min_occurrence': 0, 'order_by': 'name', 'account_type_regexes': [tfprof_logger.TRAINABLE_VARIABLES], 'start_name_regexes': ['.*'], 'trim_name_regexes': [], 'show_name_regexes': ['.*'], 'hide_name_regexes': [], 'account_displayed_op_only': True, 'select': ['params'], 'step': -1, 'output': 'stdout'}

    @staticmethod
    def float_operation():
        if False:
            for i in range(10):
                print('nop')
        'Options used to profile float operations.\n\n    Please see https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md\n    on the caveats of calculating float operations.\n\n    Returns:\n      A dict of profiling options.\n    '
        return {'max_depth': 10000, 'min_bytes': 0, 'min_micros': 0, 'min_params': 0, 'min_float_ops': 1, 'min_occurrence': 0, 'order_by': 'float_ops', 'account_type_regexes': ['.*'], 'start_name_regexes': ['.*'], 'trim_name_regexes': [], 'show_name_regexes': ['.*'], 'hide_name_regexes': [], 'account_displayed_op_only': True, 'select': ['float_ops'], 'step': -1, 'output': 'stdout'}

    @staticmethod
    def time_and_memory(min_micros=1, min_bytes=1, min_accelerator_micros=0, min_cpu_micros=0, min_peak_bytes=0, min_residual_bytes=0, min_output_bytes=0):
        if False:
            i = 10
            return i + 15
        "Show operation time and memory consumptions.\n\n    Args:\n      min_micros: Only show profiler nodes with execution time\n          no less than this. It sums accelerator and cpu times.\n      min_bytes: Only show profiler nodes requested to allocate no less bytes\n          than this.\n      min_accelerator_micros: Only show profiler nodes spend no less than\n          this time on accelerator (e.g. GPU).\n      min_cpu_micros: Only show profiler nodes spend no less than\n          this time on cpu.\n      min_peak_bytes: Only show profiler nodes using no less than this bytes\n          at peak (high watermark). For profiler nodes consist of multiple\n          graph nodes, it sums the graph nodes' peak_bytes.\n      min_residual_bytes: Only show profiler nodes have no less than\n          this bytes not being de-allocated after Compute() ends. For\n          profiler nodes consist of multiple graph nodes, it sums the\n          graph nodes' residual_bytes.\n      min_output_bytes: Only show profiler nodes have no less than this bytes\n          output. The output are not necessarily allocated by this profiler\n          nodes.\n    Returns:\n      A dict of profiling options.\n    "
        return {'max_depth': 10000, 'min_bytes': min_bytes, 'min_peak_bytes': min_peak_bytes, 'min_residual_bytes': min_residual_bytes, 'min_output_bytes': min_output_bytes, 'min_micros': min_micros, 'min_accelerator_micros': min_accelerator_micros, 'min_cpu_micros': min_cpu_micros, 'min_params': 0, 'min_float_ops': 0, 'min_occurrence': 0, 'order_by': 'micros', 'account_type_regexes': ['.*'], 'start_name_regexes': ['.*'], 'trim_name_regexes': [], 'show_name_regexes': ['.*'], 'hide_name_regexes': [], 'account_displayed_op_only': True, 'select': ['micros', 'bytes'], 'step': -1, 'output': 'stdout'}

    def build(self):
        if False:
            i = 10
            return i + 15
        'Build a profiling option.\n\n    Returns:\n      A dict of profiling options.\n    '
        return copy.deepcopy(self._options)

    def with_max_depth(self, max_depth):
        if False:
            print('Hello World!')
        "Set the maximum depth of display.\n\n    The depth depends on profiling view. For 'scope' view, it's the\n    depth of name scope hierarchy (tree), for 'op' view, it's the number\n    of operation types (list), etc.\n\n    Args:\n      max_depth: Maximum depth of the data structure to display.\n    Returns:\n      self\n    "
        self._options['max_depth'] = max_depth
        return self

    def with_min_memory(self, min_bytes=0, min_peak_bytes=0, min_residual_bytes=0, min_output_bytes=0):
        if False:
            return 10
        "Only show profiler nodes consuming no less than 'min_bytes'.\n\n    Args:\n      min_bytes: Only show profiler nodes requested to allocate no less bytes\n          than this.\n      min_peak_bytes: Only show profiler nodes using no less than this bytes\n          at peak (high watermark). For profiler nodes consist of multiple\n          graph nodes, it sums the graph nodes' peak_bytes.\n      min_residual_bytes: Only show profiler nodes have no less than\n          this bytes not being de-allocated after Compute() ends. For\n          profiler nodes consist of multiple graph nodes, it sums the\n          graph nodes' residual_bytes.\n      min_output_bytes: Only show profiler nodes have no less than this bytes\n          output. The output are not necessarily allocated by this profiler\n          nodes.\n    Returns:\n      self\n    "
        self._options['min_bytes'] = min_bytes
        self._options['min_peak_bytes'] = min_peak_bytes
        self._options['min_residual_bytes'] = min_residual_bytes
        self._options['min_output_bytes'] = min_output_bytes
        return self

    def with_min_execution_time(self, min_micros=0, min_accelerator_micros=0, min_cpu_micros=0):
        if False:
            return 10
        "Only show profiler nodes consuming no less than 'min_micros'.\n\n    Args:\n      min_micros: Only show profiler nodes with execution time\n          no less than this. It sums accelerator and cpu times.\n      min_accelerator_micros: Only show profiler nodes spend no less than\n          this time on accelerator (e.g. GPU).\n      min_cpu_micros: Only show profiler nodes spend no less than\n          this time on cpu.\n    Returns:\n      self\n    "
        self._options['min_micros'] = min_micros
        self._options['min_accelerator_micros'] = min_accelerator_micros
        self._options['min_cpu_micros'] = min_cpu_micros
        return self

    def with_min_parameters(self, min_params):
        if False:
            for i in range(10):
                print('nop')
        "Only show profiler nodes holding no less than 'min_params' parameters.\n\n    'Parameters' normally refers the weights of in TensorFlow variables.\n    It reflects the 'capacity' of models.\n\n    Args:\n      min_params: Only show profiler nodes holding number parameters\n          no less than this.\n    Returns:\n      self\n    "
        self._options['min_params'] = min_params
        return self

    def with_min_occurrence(self, min_occurrence):
        if False:
            print('Hello World!')
        'Only show profiler nodes including no less than \'min_occurrence\' graph nodes.\n\n    A "node" means a profiler output node, which can be a python line\n    (code view), an operation type (op view), or a graph node\n    (graph/scope view). A python line includes all graph nodes created by that\n    line, while an operation type includes all graph nodes of that type.\n\n    Args:\n      min_occurrence: Only show nodes including no less than this.\n    Returns:\n      self\n    '
        self._options['min_occurrence'] = min_occurrence
        return self

    def with_min_float_operations(self, min_float_ops):
        if False:
            while True:
                i = 10
        "Only show profiler nodes consuming no less than 'min_float_ops'.\n\n    Please see https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md\n    on the caveats of calculating float operations.\n\n    Args:\n      min_float_ops: Only show profiler nodes with float operations\n          no less than this.\n    Returns:\n      self\n    "
        self._options['min_float_ops'] = min_float_ops
        return self

    def with_accounted_types(self, account_type_regexes):
        if False:
            i = 10
            return i + 15
        "Selectively counting statistics based on node types.\n\n    Here, 'types' means the profiler nodes' properties. Profiler by default\n    consider device name (e.g. /job:xx/.../device:GPU:0) and operation type\n    (e.g. MatMul) as profiler nodes' properties. User can also associate\n    customized 'types' to profiler nodes through OpLogProto proto.\n\n    For example, user can select profiler nodes placed on gpu:0 with:\n    `account_type_regexes=['.*gpu:0.*']`\n\n    If none of a node's properties match the specified regexes, the node is\n    not displayed nor accounted.\n\n    Args:\n      account_type_regexes: A list of regexes specifying the types.\n    Returns:\n      self.\n    "
        self._options['account_type_regexes'] = copy.copy(account_type_regexes)
        return self

    def with_node_names(self, start_name_regexes=None, show_name_regexes=None, hide_name_regexes=None, trim_name_regexes=None):
        if False:
            for i in range(10):
                print('nop')
        "Regular expressions used to select profiler nodes to display.\n\n    After 'with_accounted_types' is evaluated, 'with_node_names' are\n    evaluated as follows:\n\n      For a profile data structure, profiler first finds the profiler\n      nodes matching 'start_name_regexes', and starts displaying profiler\n      nodes from there. Then, if a node matches 'show_name_regexes' and\n      doesn't match 'hide_name_regexes', it's displayed. If a node matches\n      'trim_name_regexes', profiler stops further searching that branch.\n\n    Args:\n      start_name_regexes: list of node name regexes to start displaying.\n      show_name_regexes: list of node names regexes to display.\n      hide_name_regexes: list of node_names regexes that should be hidden.\n      trim_name_regexes: list of node name regexes from where to stop.\n    Returns:\n      self\n    "
        if start_name_regexes is not None:
            self._options['start_name_regexes'] = copy.copy(start_name_regexes)
        if show_name_regexes is not None:
            self._options['show_name_regexes'] = copy.copy(show_name_regexes)
        if hide_name_regexes is not None:
            self._options['hide_name_regexes'] = copy.copy(hide_name_regexes)
        if trim_name_regexes is not None:
            self._options['trim_name_regexes'] = copy.copy(trim_name_regexes)
        return self

    def account_displayed_op_only(self, is_true):
        if False:
            return 10
        "Whether only account the statistics of displayed profiler nodes.\n\n    Args:\n      is_true: If true, only account statistics of nodes eventually\n          displayed by the outputs.\n          Otherwise, a node's statistics are accounted by its parents\n          as long as it's types match 'account_type_regexes', even if\n          it is hidden from the output, say, by hide_name_regexes.\n    Returns:\n      self\n    "
        self._options['account_displayed_op_only'] = is_true
        return self

    def with_empty_output(self):
        if False:
            while True:
                i = 10
        'Do not generate side-effect outputs.'
        self._options['output'] = 'none'
        return self

    def with_stdout_output(self):
        if False:
            return 10
        'Print the result to stdout.'
        self._options['output'] = 'stdout'
        return self

    def with_file_output(self, outfile):
        if False:
            while True:
                i = 10
        'Print the result to a file.'
        self._options['output'] = 'file:outfile=%s' % outfile
        return self

    def with_timeline_output(self, timeline_file):
        if False:
            print('Hello World!')
        'Generate a timeline json file.'
        self._options['output'] = 'timeline:outfile=%s' % timeline_file
        return self

    def with_pprof_output(self, pprof_file):
        if False:
            return 10
        'Generate a pprof profile gzip file.\n\n    To use the pprof file:\n      pprof -png --nodecount=100 --sample_index=1 <pprof_file>\n\n    Args:\n      pprof_file: filename for output, usually suffixed with .pb.gz.\n    Returns:\n      self.\n    '
        self._options['output'] = 'pprof:outfile=%s' % pprof_file
        return self

    def order_by(self, attribute):
        if False:
            for i in range(10):
                print('nop')
        'Order the displayed profiler nodes based on a attribute.\n\n    Supported attribute includes micros, bytes, occurrence, params, etc.\n    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md\n\n    Args:\n      attribute: An attribute the profiler node has.\n    Returns:\n      self\n    '
        self._options['order_by'] = attribute
        return self

    def select(self, attributes):
        if False:
            while True:
                i = 10
        'Select the attributes to display.\n\n    See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md\n    for supported attributes.\n\n    Args:\n      attributes: A list of attribute the profiler node has.\n    Returns:\n      self\n    '
        self._options['select'] = copy.copy(attributes)
        return self

    def with_step(self, step):
        if False:
            while True:
                i = 10
        "Which profile step to use for profiling.\n\n    The 'step' here refers to the step defined by `Profiler.add_step()` API.\n\n    Args:\n      step: When multiple steps of profiles are available, select which step's\n         profile to use. If -1, use average of all available steps.\n    Returns:\n      self\n    "
        self._options['step'] = step
        return self