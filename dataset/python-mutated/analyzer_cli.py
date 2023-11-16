"""CLI Backend for the Analyzer Part of the Debugger.

The analyzer performs post hoc analysis of dumped intermediate tensors and
graph structure information from debugged Session.run() calls.
"""
import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
RL = debugger_cli_common.RichLine
HANG_UNFINISHED = '|  '
HANG_FINISHED = '   '
HANG_SUFFIX = '|- '
DEPTH_TEMPLATE = '(%d) '
OP_TYPE_TEMPLATE = '[%s] '
CTRL_LABEL = '(Ctrl) '
ELLIPSIS = '...'
SORT_TENSORS_BY_TIMESTAMP = 'timestamp'
SORT_TENSORS_BY_DUMP_SIZE = 'dump_size'
SORT_TENSORS_BY_OP_TYPE = 'op_type'
SORT_TENSORS_BY_TENSOR_NAME = 'tensor_name'

def _add_main_menu(output, node_name=None, enable_list_tensors=True, enable_node_info=True, enable_print_tensor=True, enable_list_inputs=True, enable_list_outputs=True):
    if False:
        print('Hello World!')
    'Generate main menu for the screen output from a command.\n\n  Args:\n    output: (debugger_cli_common.RichTextLines) the output object to modify.\n    node_name: (str or None) name of the node involved (if any). If None,\n      the menu items node_info, list_inputs and list_outputs will be\n      automatically disabled, overriding the values of arguments\n      enable_node_info, enable_list_inputs and enable_list_outputs.\n    enable_list_tensors: (bool) whether the list_tensor menu item will be\n      enabled.\n    enable_node_info: (bool) whether the node_info item will be enabled.\n    enable_print_tensor: (bool) whether the print_tensor item will be enabled.\n    enable_list_inputs: (bool) whether the item list_inputs will be enabled.\n    enable_list_outputs: (bool) whether the item list_outputs will be enabled.\n  '
    menu = debugger_cli_common.Menu()
    menu.append(debugger_cli_common.MenuItem('list_tensors', 'list_tensors', enabled=enable_list_tensors))
    if node_name:
        menu.append(debugger_cli_common.MenuItem('node_info', 'node_info -a -d -t %s' % node_name, enabled=enable_node_info))
        menu.append(debugger_cli_common.MenuItem('print_tensor', 'print_tensor %s' % node_name, enabled=enable_print_tensor))
        menu.append(debugger_cli_common.MenuItem('list_inputs', 'list_inputs -c -r %s' % node_name, enabled=enable_list_inputs))
        menu.append(debugger_cli_common.MenuItem('list_outputs', 'list_outputs -c -r %s' % node_name, enabled=enable_list_outputs))
    else:
        menu.append(debugger_cli_common.MenuItem('node_info', None, enabled=False))
        menu.append(debugger_cli_common.MenuItem('print_tensor', None, enabled=False))
        menu.append(debugger_cli_common.MenuItem('list_inputs', None, enabled=False))
        menu.append(debugger_cli_common.MenuItem('list_outputs', None, enabled=False))
    menu.append(debugger_cli_common.MenuItem('run_info', 'run_info'))
    menu.append(debugger_cli_common.MenuItem('help', 'help'))
    output.annotations[debugger_cli_common.MAIN_MENU_KEY] = menu

class DebugAnalyzer(object):
    """Analyzer for debug data from dump directories."""
    _TIMESTAMP_COLUMN_HEAD = 't (ms)'
    _DUMP_SIZE_COLUMN_HEAD = 'Size (B)'
    _OP_TYPE_COLUMN_HEAD = 'Op type'
    _TENSOR_NAME_COLUMN_HEAD = 'Tensor name'
    _GRAPH_STRUCT_OP_TYPE_DENYLIST = ('_Send', '_Recv', '_HostSend', '_HostRecv', '_Retval')

    def __init__(self, debug_dump, config):
        if False:
            return 10
        'DebugAnalyzer constructor.\n\n    Args:\n      debug_dump: A DebugDumpDir object.\n      config: A `cli_config.CLIConfig` object that carries user-facing\n        configurations.\n    '
        self._debug_dump = debug_dump
        self._evaluator = evaluator.ExpressionEvaluator(self._debug_dump)
        self._tensor_filters = {}
        self._build_argument_parsers(config)
        config.set_callback('graph_recursion_depth', self._build_argument_parsers)

    def _build_argument_parsers(self, config):
        if False:
            while True:
                i = 10
        'Build argument parsers for DebugAnalayzer.\n\n    Args:\n      config: A `cli_config.CLIConfig` object.\n\n    Returns:\n      A dict mapping command handler name to `ArgumentParser` instance.\n    '
        self._arg_parsers = {}
        ap = argparse.ArgumentParser(description='List dumped intermediate tensors.', usage=argparse.SUPPRESS)
        ap.add_argument('-f', '--tensor_filter', dest='tensor_filter', type=str, default='', help='List only Tensors passing the filter of the specified name')
        ap.add_argument('-fenn', '--filter_exclude_node_names', dest='filter_exclude_node_names', type=str, default='', help='When applying the tensor filter, exclude node with names matching the regular expression. Applicable only if --tensor_filter or -f is used.')
        ap.add_argument('-n', '--node_name_filter', dest='node_name_filter', type=str, default='', help='filter node name by regex.')
        ap.add_argument('-t', '--op_type_filter', dest='op_type_filter', type=str, default='', help='filter op type by regex.')
        ap.add_argument('-s', '--sort_by', dest='sort_by', type=str, default=SORT_TENSORS_BY_TIMESTAMP, help='the field to sort the data by: (%s | %s | %s | %s)' % (SORT_TENSORS_BY_TIMESTAMP, SORT_TENSORS_BY_DUMP_SIZE, SORT_TENSORS_BY_OP_TYPE, SORT_TENSORS_BY_TENSOR_NAME))
        ap.add_argument('-r', '--reverse', dest='reverse', action='store_true', help='sort the data in reverse (descending) order')
        self._arg_parsers['list_tensors'] = ap
        ap = argparse.ArgumentParser(description='Show information about a node.', usage=argparse.SUPPRESS)
        ap.add_argument('node_name', type=str, help='Name of the node or an associated tensor, e.g., hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0')
        ap.add_argument('-a', '--attributes', dest='attributes', action='store_true', help='Also list attributes of the node.')
        ap.add_argument('-d', '--dumps', dest='dumps', action='store_true', help='Also list dumps available from the node.')
        ap.add_argument('-t', '--traceback', dest='traceback', action='store_true', help="Also include the traceback of the node's creation (if available in Python).")
        self._arg_parsers['node_info'] = ap
        ap = argparse.ArgumentParser(description='Show inputs to a node.', usage=argparse.SUPPRESS)
        ap.add_argument('node_name', type=str, help='Name of the node or an output tensor from the node, e.g., hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0')
        ap.add_argument('-c', '--control', action='store_true', help='Include control inputs.')
        ap.add_argument('-d', '--depth', dest='depth', type=int, default=config.get('graph_recursion_depth'), help='Maximum depth of recursion used when showing the input tree.')
        ap.add_argument('-r', '--recursive', dest='recursive', action='store_true', help='Show inputs to the node recursively, i.e., the input tree.')
        ap.add_argument('-t', '--op_type', action='store_true', help='Show op types of input nodes.')
        self._arg_parsers['list_inputs'] = ap
        ap = argparse.ArgumentParser(description='Show the nodes that receive the outputs of given node.', usage=argparse.SUPPRESS)
        ap.add_argument('node_name', type=str, help='Name of the node or an output tensor from the node, e.g., hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0')
        ap.add_argument('-c', '--control', action='store_true', help='Include control inputs.')
        ap.add_argument('-d', '--depth', dest='depth', type=int, default=config.get('graph_recursion_depth'), help='Maximum depth of recursion used when showing the output tree.')
        ap.add_argument('-r', '--recursive', dest='recursive', action='store_true', help='Show recipients of the node recursively, i.e., the output tree.')
        ap.add_argument('-t', '--op_type', action='store_true', help='Show op types of recipient nodes.')
        self._arg_parsers['list_outputs'] = ap
        self._arg_parsers['print_tensor'] = command_parser.get_print_tensor_argparser('Print the value of a dumped tensor.')
        ap = argparse.ArgumentParser(description='Print a Python source file with overlaid debug information, including the nodes (ops) or Tensors created at the source lines.', usage=argparse.SUPPRESS)
        ap.add_argument('source_file_path', type=str, help='Path to the source file.')
        ap.add_argument('-t', '--tensors', dest='tensors', action='store_true', help='Label lines with dumped Tensors, instead of ops.')
        ap.add_argument('-m', '--max_elements_per_line', type=int, default=10, help='Maximum number of elements (ops or Tensors) to show per source line.')
        ap.add_argument('-b', '--line_begin', type=int, default=1, help='Print source beginning at line number (1-based.)')
        self._arg_parsers['print_source'] = ap
        ap = argparse.ArgumentParser(description='List source files responsible for constructing nodes and tensors present in the run().', usage=argparse.SUPPRESS)
        ap.add_argument('-p', '--path_filter', type=str, default='', help='Regular expression filter for file path.')
        ap.add_argument('-n', '--node_name_filter', type=str, default='', help='Regular expression filter for node name.')
        self._arg_parsers['list_source'] = ap
        ap = argparse.ArgumentParser(description='Evaluate an arbitrary expression. Can use tensor values\n        from the current debug dump. The debug tensor names should be enclosed\n        in pairs of backticks. Expressions with spaces should be enclosed in\n        a pair of double quotes or a pair of single quotes. By default, numpy\n        is imported as np and can be used in the expressions. E.g.,\n          1) eval np.argmax(`Softmax:0`),\n          2) eval \'np.sum(`Softmax:0`, axis=1)\',\n          3) eval "np.matmul((`output/Identity:0`/`Softmax:0`).T, `Softmax:0`)".\n        ', usage=argparse.SUPPRESS)
        ap.add_argument('expression', type=str, help='Expression to be evaluated.\n        1) in the simplest case, use <node_name>:<output_slot>, e.g.,\n          hidden_0/MatMul:0.\n\n        2) if the default debug op "DebugIdentity" is to be overridden, use\n          <node_name>:<output_slot>:<debug_op>, e.g.,\n          hidden_0/MatMul:0:DebugNumericSummary.\n\n        3) if the tensor of the same name exists on more than one device, use\n          <device_name>:<node_name>:<output_slot>[:<debug_op>], e.g.,\n          /job:worker/replica:0/task:0/gpu:0:hidden_0/MatMul:0\n          /job:worker/replica:0/task:2/cpu:0:hidden_0/MatMul:0:DebugNanCount.\n\n        4) if the tensor is executed multiple times in a given `Session.run`\n        call, specify the execution index with a 0-based integer enclose in a\n        pair of brackets at the end, e.g.,\n          RNN/tanh:0[0]\n          /job:worker/replica:0/task:0/gpu:0:RNN/tanh:0[0].')
        ap.add_argument('-a', '--all', dest='print_all', action='store_true', help='Print the tensor in its entirety, i.e., do not use ellipses (may be slow for large results).')
        ap.add_argument('-w', '--write_path', default='', help='Path of the numpy file to write the evaluation result to, using numpy.save()')
        self._arg_parsers['eval'] = ap

    def add_tensor_filter(self, filter_name, filter_callable):
        if False:
            while True:
                i = 10
        'Add a tensor filter.\n\n    A tensor filter is a named callable of the signature:\n      filter_callable(dump_datum, tensor),\n\n    wherein dump_datum is an instance of debug_data.DebugTensorDatum carrying\n    metadata about the dumped tensor, including tensor name, timestamps, etc.\n    tensor is the value of the dumped tensor as an numpy.ndarray object.\n    The return value of the function is a bool.\n    This is the same signature as the input argument to\n    debug_data.DebugDumpDir.find().\n\n    Args:\n      filter_name: (str) name of the filter. Cannot be empty.\n      filter_callable: (callable) a filter function of the signature described\n        as above.\n\n    Raises:\n      ValueError: If filter_name is an empty str.\n      TypeError: If filter_name is not a str.\n                 Or if filter_callable is not callable.\n    '
        if not isinstance(filter_name, str):
            raise TypeError('Input argument filter_name is expected to be str, but is not.')
        if not filter_name:
            raise ValueError('Input argument filter_name cannot be empty.')
        if not callable(filter_callable):
            raise TypeError('Input argument filter_callable is expected to be callable, but is not.')
        self._tensor_filters[filter_name] = filter_callable

    def get_tensor_filter(self, filter_name):
        if False:
            while True:
                i = 10
        'Retrieve filter function by name.\n\n    Args:\n      filter_name: Name of the filter set during add_tensor_filter() call.\n\n    Returns:\n      The callable associated with the filter name.\n\n    Raises:\n      ValueError: If there is no tensor filter of the specified filter name.\n    '
        if filter_name not in self._tensor_filters:
            raise ValueError('There is no tensor filter named "%s"' % filter_name)
        return self._tensor_filters[filter_name]

    def get_help(self, handler_name):
        if False:
            return 10
        return self._arg_parsers[handler_name].format_help()

    def list_tensors(self, args, screen_info=None):
        if False:
            for i in range(10):
                print('nop')
        'Command handler for list_tensors.\n\n    List tensors dumped during debugged Session.run() call.\n\n    Args:\n      args: Command-line arguments, excluding the command prefix, as a list of\n        str.\n      screen_info: Optional dict input containing screen information such as\n        cols.\n\n    Returns:\n      Output text lines as a RichTextLines object.\n\n    Raises:\n      ValueError: If `--filter_exclude_node_names` is used without `-f` or\n        `--tensor_filter` being used.\n    '
        _ = screen_info
        parsed = self._arg_parsers['list_tensors'].parse_args(args)
        output = []
        filter_strs = []
        if parsed.op_type_filter:
            op_type_regex = re.compile(parsed.op_type_filter)
            filter_strs.append('Op type regex filter: "%s"' % parsed.op_type_filter)
        else:
            op_type_regex = None
        if parsed.node_name_filter:
            node_name_regex = re.compile(parsed.node_name_filter)
            filter_strs.append('Node name regex filter: "%s"' % parsed.node_name_filter)
        else:
            node_name_regex = None
        output = debugger_cli_common.RichTextLines(filter_strs)
        output.append('')
        if parsed.tensor_filter:
            try:
                filter_callable = self.get_tensor_filter(parsed.tensor_filter)
            except ValueError:
                output = cli_shared.error('There is no tensor filter named "%s".' % parsed.tensor_filter)
                _add_main_menu(output, node_name=None, enable_list_tensors=False)
                return output
            data_to_show = self._debug_dump.find(filter_callable, exclude_node_names=parsed.filter_exclude_node_names)
        else:
            if parsed.filter_exclude_node_names:
                raise ValueError('The flag --filter_exclude_node_names is valid only when the flag -f or --tensor_filter is used.')
            data_to_show = self._debug_dump.dumped_tensor_data
        (max_timestamp_width, max_dump_size_width, max_op_type_width) = self._measure_tensor_list_column_widths(data_to_show)
        data_to_show = self._sort_dump_data_by(data_to_show, parsed.sort_by, parsed.reverse)
        output.extend(self._tensor_list_column_heads(parsed, max_timestamp_width, max_dump_size_width, max_op_type_width))
        dump_count = 0
        for dump in data_to_show:
            if node_name_regex and (not node_name_regex.match(dump.node_name)):
                continue
            if op_type_regex:
                op_type = self._debug_dump.node_op_type(dump.node_name)
                if not op_type_regex.match(op_type):
                    continue
            rel_time = (dump.timestamp - self._debug_dump.t0) / 1000.0
            dump_size_str = cli_shared.bytes_to_readable_str(dump.dump_size_bytes)
            dumped_tensor_name = '%s:%d' % (dump.node_name, dump.output_slot)
            op_type = self._debug_dump.node_op_type(dump.node_name)
            line = '[%.3f]' % rel_time
            line += ' ' * (max_timestamp_width - len(line))
            line += dump_size_str
            line += ' ' * (max_timestamp_width + max_dump_size_width - len(line))
            line += op_type
            line += ' ' * (max_timestamp_width + max_dump_size_width + max_op_type_width - len(line))
            line += dumped_tensor_name
            output.append(line, font_attr_segs=[(len(line) - len(dumped_tensor_name), len(line), debugger_cli_common.MenuItem('', 'pt %s' % dumped_tensor_name))])
            dump_count += 1
        if parsed.tensor_filter:
            output.prepend(['%d dumped tensor(s) passing filter "%s":' % (dump_count, parsed.tensor_filter)])
        else:
            output.prepend(['%d dumped tensor(s):' % dump_count])
        _add_main_menu(output, node_name=None, enable_list_tensors=False)
        return output

    def _measure_tensor_list_column_widths(self, data):
        if False:
            while True:
                i = 10
        'Determine the maximum widths of the timestamp and op-type column.\n\n    This method assumes that data is sorted in the default order, i.e.,\n    by ascending timestamps.\n\n    Args:\n      data: (list of DebugTensorDaum) the data based on which the maximum\n        column widths will be determined.\n\n    Returns:\n      (int) maximum width of the timestamp column. 0 if data is empty.\n      (int) maximum width of the dump size column. 0 if data is empty.\n      (int) maximum width of the op type column. 0 if data is empty.\n    '
        max_timestamp_width = 0
        if data:
            max_rel_time_ms = (data[-1].timestamp - self._debug_dump.t0) / 1000.0
            max_timestamp_width = len('[%.3f] ' % max_rel_time_ms) + 1
        max_timestamp_width = max(max_timestamp_width, len(self._TIMESTAMP_COLUMN_HEAD) + 1)
        max_dump_size_width = 0
        for dump in data:
            dump_size_str = cli_shared.bytes_to_readable_str(dump.dump_size_bytes)
            if len(dump_size_str) + 1 > max_dump_size_width:
                max_dump_size_width = len(dump_size_str) + 1
        max_dump_size_width = max(max_dump_size_width, len(self._DUMP_SIZE_COLUMN_HEAD) + 1)
        max_op_type_width = 0
        for dump in data:
            op_type = self._debug_dump.node_op_type(dump.node_name)
            if len(op_type) + 1 > max_op_type_width:
                max_op_type_width = len(op_type) + 1
        max_op_type_width = max(max_op_type_width, len(self._OP_TYPE_COLUMN_HEAD) + 1)
        return (max_timestamp_width, max_dump_size_width, max_op_type_width)

    def _sort_dump_data_by(self, data, sort_by, reverse):
        if False:
            return 10
        'Sort a list of DebugTensorDatum in specified order.\n\n    Args:\n      data: (list of DebugTensorDatum) the data to be sorted.\n      sort_by: The field to sort data by.\n      reverse: (bool) Whether to use reversed (descending) order.\n\n    Returns:\n      (list of DebugTensorDatum) in sorted order.\n\n    Raises:\n      ValueError: given an invalid value of sort_by.\n    '
        if sort_by == SORT_TENSORS_BY_TIMESTAMP:
            return sorted(data, reverse=reverse, key=lambda x: x.timestamp)
        elif sort_by == SORT_TENSORS_BY_DUMP_SIZE:
            return sorted(data, reverse=reverse, key=lambda x: x.dump_size_bytes)
        elif sort_by == SORT_TENSORS_BY_OP_TYPE:
            return sorted(data, reverse=reverse, key=lambda x: self._debug_dump.node_op_type(x.node_name))
        elif sort_by == SORT_TENSORS_BY_TENSOR_NAME:
            return sorted(data, reverse=reverse, key=lambda x: '%s:%d' % (x.node_name, x.output_slot))
        else:
            raise ValueError('Unsupported key to sort tensors by: %s' % sort_by)

    def _tensor_list_column_heads(self, parsed, max_timestamp_width, max_dump_size_width, max_op_type_width):
        if False:
            print('Hello World!')
        'Generate a line containing the column heads of the tensor list.\n\n    Args:\n      parsed: Parsed arguments (by argparse) of the list_tensors command.\n      max_timestamp_width: (int) maximum width of the timestamp column.\n      max_dump_size_width: (int) maximum width of the dump size column.\n      max_op_type_width: (int) maximum width of the op type column.\n\n    Returns:\n      A RichTextLines object.\n    '
        base_command = 'list_tensors'
        if parsed.tensor_filter:
            base_command += ' -f %s' % parsed.tensor_filter
        if parsed.op_type_filter:
            base_command += ' -t %s' % parsed.op_type_filter
        if parsed.node_name_filter:
            base_command += ' -n %s' % parsed.node_name_filter
        attr_segs = {0: []}
        row = self._TIMESTAMP_COLUMN_HEAD
        command = '%s -s %s' % (base_command, SORT_TENSORS_BY_TIMESTAMP)
        if parsed.sort_by == SORT_TENSORS_BY_TIMESTAMP and (not parsed.reverse):
            command += ' -r'
        attr_segs[0].append((0, len(row), [debugger_cli_common.MenuItem(None, command), 'bold']))
        row += ' ' * (max_timestamp_width - len(row))
        prev_len = len(row)
        row += self._DUMP_SIZE_COLUMN_HEAD
        command = '%s -s %s' % (base_command, SORT_TENSORS_BY_DUMP_SIZE)
        if parsed.sort_by == SORT_TENSORS_BY_DUMP_SIZE and (not parsed.reverse):
            command += ' -r'
        attr_segs[0].append((prev_len, len(row), [debugger_cli_common.MenuItem(None, command), 'bold']))
        row += ' ' * (max_dump_size_width + max_timestamp_width - len(row))
        prev_len = len(row)
        row += self._OP_TYPE_COLUMN_HEAD
        command = '%s -s %s' % (base_command, SORT_TENSORS_BY_OP_TYPE)
        if parsed.sort_by == SORT_TENSORS_BY_OP_TYPE and (not parsed.reverse):
            command += ' -r'
        attr_segs[0].append((prev_len, len(row), [debugger_cli_common.MenuItem(None, command), 'bold']))
        row += ' ' * (max_op_type_width + max_dump_size_width + max_timestamp_width - len(row))
        prev_len = len(row)
        row += self._TENSOR_NAME_COLUMN_HEAD
        command = '%s -s %s' % (base_command, SORT_TENSORS_BY_TENSOR_NAME)
        if parsed.sort_by == SORT_TENSORS_BY_TENSOR_NAME and (not parsed.reverse):
            command += ' -r'
        attr_segs[0].append((prev_len, len(row), [debugger_cli_common.MenuItem('', command), 'bold']))
        row += ' ' * (max_op_type_width + max_dump_size_width + max_timestamp_width - len(row))
        return debugger_cli_common.RichTextLines([row], font_attr_segs=attr_segs)

    def node_info(self, args, screen_info=None):
        if False:
            while True:
                i = 10
        'Command handler for node_info.\n\n    Query information about a given node.\n\n    Args:\n      args: Command-line arguments, excluding the command prefix, as a list of\n        str.\n      screen_info: Optional dict input containing screen information such as\n        cols.\n\n    Returns:\n      Output text lines as a RichTextLines object.\n    '
        _ = screen_info
        parsed = self._arg_parsers['node_info'].parse_args(args)
        (node_name, unused_slot) = debug_graphs.parse_node_or_tensor_name(parsed.node_name)
        if not self._debug_dump.node_exists(node_name):
            output = cli_shared.error('There is no node named "%s" in the partition graphs' % node_name)
            _add_main_menu(output, node_name=None, enable_list_tensors=True, enable_node_info=False, enable_list_inputs=False, enable_list_outputs=False)
            return output
        lines = ['Node %s' % node_name]
        font_attr_segs = {0: [(len(lines[-1]) - len(node_name), len(lines[-1]), 'bold')]}
        lines.append('')
        lines.append('  Op: %s' % self._debug_dump.node_op_type(node_name))
        lines.append('  Device: %s' % self._debug_dump.node_device(node_name))
        output = debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)
        inputs = self._exclude_denylisted_ops(self._debug_dump.node_inputs(node_name))
        ctrl_inputs = self._exclude_denylisted_ops(self._debug_dump.node_inputs(node_name, is_control=True))
        output.extend(self._format_neighbors('input', inputs, ctrl_inputs))
        recs = self._exclude_denylisted_ops(self._debug_dump.node_recipients(node_name))
        ctrl_recs = self._exclude_denylisted_ops(self._debug_dump.node_recipients(node_name, is_control=True))
        output.extend(self._format_neighbors('recipient', recs, ctrl_recs))
        if parsed.attributes:
            output.extend(self._list_node_attributes(node_name))
        if parsed.dumps:
            output.extend(self._list_node_dumps(node_name))
        if parsed.traceback:
            output.extend(self._render_node_traceback(node_name))
        _add_main_menu(output, node_name=node_name, enable_node_info=False)
        return output

    def _exclude_denylisted_ops(self, node_names):
        if False:
            for i in range(10):
                print('nop')
        'Exclude all nodes whose op types are in _GRAPH_STRUCT_OP_TYPE_DENYLIST.\n\n    Args:\n      node_names: An iterable of node or graph element names.\n\n    Returns:\n      A list of node names that are not denylisted.\n    '
        return [node_name for node_name in node_names if self._debug_dump.node_op_type(debug_graphs.get_node_name(node_name)) not in self._GRAPH_STRUCT_OP_TYPE_DENYLIST]

    def _render_node_traceback(self, node_name):
        if False:
            print('Hello World!')
        "Render traceback of a node's creation in Python, if available.\n\n    Args:\n      node_name: (str) name of the node.\n\n    Returns:\n      A RichTextLines object containing the stack trace of the node's\n      construction.\n    "
        lines = [RL(''), RL(''), RL('Traceback of node construction:', 'bold')]
        try:
            node_stack = self._debug_dump.node_traceback(node_name)
            for (depth, (file_path, line, function_name, text)) in enumerate(node_stack):
                lines.append('%d: %s' % (depth, file_path))
                attribute = debugger_cli_common.MenuItem('', 'ps %s -b %d' % (file_path, line)) if text else None
                line_number_line = RL('  ')
                line_number_line += RL('Line:     %d' % line, attribute)
                lines.append(line_number_line)
                lines.append('  Function: %s' % function_name)
                lines.append('  Text:     ' + ('"%s"' % text if text else 'None'))
                lines.append('')
        except KeyError:
            lines.append('(Node unavailable in the loaded Python graph)')
        except LookupError:
            lines.append('(Unavailable because no Python graph has been loaded)')
        return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

    def list_inputs(self, args, screen_info=None):
        if False:
            while True:
                i = 10
        'Command handler for inputs.\n\n    Show inputs to a given node.\n\n    Args:\n      args: Command-line arguments, excluding the command prefix, as a list of\n        str.\n      screen_info: Optional dict input containing screen information such as\n        cols.\n\n    Returns:\n      Output text lines as a RichTextLines object.\n    '
        _ = screen_info
        parsed = self._arg_parsers['list_inputs'].parse_args(args)
        output = self._list_inputs_or_outputs(parsed.recursive, parsed.node_name, parsed.depth, parsed.control, parsed.op_type, do_outputs=False)
        node_name = debug_graphs.get_node_name(parsed.node_name)
        _add_main_menu(output, node_name=node_name, enable_list_inputs=False)
        return output

    def print_tensor(self, args, screen_info=None):
        if False:
            i = 10
            return i + 15
        'Command handler for print_tensor.\n\n    Print value of a given dumped tensor.\n\n    Args:\n      args: Command-line arguments, excluding the command prefix, as a list of\n        str.\n      screen_info: Optional dict input containing screen information such as\n        cols.\n\n    Returns:\n      Output text lines as a RichTextLines object.\n    '
        parsed = self._arg_parsers['print_tensor'].parse_args(args)
        np_printoptions = cli_shared.numpy_printoptions_from_screen_info(screen_info)
        highlight_options = cli_shared.parse_ranges_highlight(parsed.ranges)
        (tensor_name, tensor_slicing) = command_parser.parse_tensor_name_with_slicing(parsed.tensor_name)
        (node_name, output_slot) = debug_graphs.parse_node_or_tensor_name(tensor_name)
        if self._debug_dump.loaded_partition_graphs() and (not self._debug_dump.node_exists(node_name)):
            output = cli_shared.error('Node "%s" does not exist in partition graphs' % node_name)
            _add_main_menu(output, node_name=None, enable_list_tensors=True, enable_print_tensor=False)
            return output
        watch_keys = self._debug_dump.debug_watch_keys(node_name)
        if output_slot is None:
            output_slots = set()
            for watch_key in watch_keys:
                output_slots.add(int(watch_key.split(':')[1]))
            if len(output_slots) == 1:
                output_slot = list(output_slots)[0]
            else:
                lines = ['Node "%s" generated debug dumps from %s output slots:' % (node_name, len(output_slots)), 'Please specify the output slot: %s:x.' % node_name]
                output = debugger_cli_common.RichTextLines(lines)
                _add_main_menu(output, node_name=node_name, enable_list_tensors=True, enable_print_tensor=False)
                return output
        matching_data = []
        for watch_key in watch_keys:
            debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
            for datum in debug_tensor_data:
                if datum.output_slot == output_slot:
                    matching_data.append(datum)
        if not matching_data:
            output = cli_shared.error('Tensor "%s" did not generate any dumps.' % parsed.tensor_name)
        elif len(matching_data) == 1:
            if parsed.number <= 0:
                output = cli_shared.format_tensor(matching_data[0].get_tensor(), matching_data[0].watch_key, np_printoptions, print_all=parsed.print_all, tensor_slicing=tensor_slicing, highlight_options=highlight_options, include_numeric_summary=parsed.numeric_summary, write_path=parsed.write_path)
            else:
                output = cli_shared.error('Invalid number (%d) for tensor %s, which generated one dump.' % (parsed.number, parsed.tensor_name))
            _add_main_menu(output, node_name=node_name, enable_print_tensor=False)
        else:
            if parsed.number < 0:
                lines = ['Tensor "%s" generated %d dumps:' % (parsed.tensor_name, len(matching_data))]
                font_attr_segs = {}
                for (i, datum) in enumerate(matching_data):
                    rel_time = (datum.timestamp - self._debug_dump.t0) / 1000.0
                    lines.append('#%d [%.3f ms] %s' % (i, rel_time, datum.watch_key))
                    command = 'print_tensor %s -n %d' % (parsed.tensor_name, i)
                    font_attr_segs[len(lines) - 1] = [(len(lines[-1]) - len(datum.watch_key), len(lines[-1]), debugger_cli_common.MenuItem(None, command))]
                lines.append('')
                lines.append('You can use the -n (--number) flag to specify which dump to print.')
                lines.append('For example:')
                lines.append('  print_tensor %s -n 0' % parsed.tensor_name)
                output = debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)
            elif parsed.number >= len(matching_data):
                output = cli_shared.error('Specified number (%d) exceeds the number of available dumps (%d) for tensor %s' % (parsed.number, len(matching_data), parsed.tensor_name))
            else:
                output = cli_shared.format_tensor(matching_data[parsed.number].get_tensor(), matching_data[parsed.number].watch_key + ' (dump #%d)' % parsed.number, np_printoptions, print_all=parsed.print_all, tensor_slicing=tensor_slicing, highlight_options=highlight_options, write_path=parsed.write_path)
            _add_main_menu(output, node_name=node_name, enable_print_tensor=False)
        return output

    def list_outputs(self, args, screen_info=None):
        if False:
            print('Hello World!')
        'Command handler for inputs.\n\n    Show inputs to a given node.\n\n    Args:\n      args: Command-line arguments, excluding the command prefix, as a list of\n        str.\n      screen_info: Optional dict input containing screen information such as\n        cols.\n\n    Returns:\n      Output text lines as a RichTextLines object.\n    '
        _ = screen_info
        parsed = self._arg_parsers['list_outputs'].parse_args(args)
        output = self._list_inputs_or_outputs(parsed.recursive, parsed.node_name, parsed.depth, parsed.control, parsed.op_type, do_outputs=True)
        node_name = debug_graphs.get_node_name(parsed.node_name)
        _add_main_menu(output, node_name=node_name, enable_list_outputs=False)
        return output

    def evaluate_expression(self, args, screen_info=None):
        if False:
            return 10
        parsed = self._arg_parsers['eval'].parse_args(args)
        eval_res = self._evaluator.evaluate(parsed.expression)
        np_printoptions = cli_shared.numpy_printoptions_from_screen_info(screen_info)
        return cli_shared.format_tensor(eval_res, "from eval of expression '%s'" % parsed.expression, np_printoptions, print_all=parsed.print_all, include_numeric_summary=True, write_path=parsed.write_path)

    def _reconstruct_print_source_command(self, parsed, line_begin, max_elements_per_line_increase=0):
        if False:
            while True:
                i = 10
        return 'ps %s %s -b %d -m %d' % (parsed.source_file_path, '-t' if parsed.tensors else '', line_begin, parsed.max_elements_per_line + max_elements_per_line_increase)

    def print_source(self, args, screen_info=None):
        if False:
            return 10
        'Print the content of a source file.'
        del screen_info
        parsed = self._arg_parsers['print_source'].parse_args(args)
        source_annotation = source_utils.annotate_source(self._debug_dump, parsed.source_file_path, do_dumped_tensors=parsed.tensors)
        (source_lines, line_num_width) = source_utils.load_source(parsed.source_file_path)
        labeled_source_lines = []
        actual_initial_scroll_target = 0
        for (i, line) in enumerate(source_lines):
            annotated_line = RL('L%d' % (i + 1), cli_shared.COLOR_YELLOW)
            annotated_line += ' ' * (line_num_width - len(annotated_line))
            annotated_line += line
            labeled_source_lines.append(annotated_line)
            if i + 1 == parsed.line_begin:
                actual_initial_scroll_target = len(labeled_source_lines) - 1
            if i + 1 in source_annotation:
                sorted_elements = sorted(source_annotation[i + 1])
                for (k, element) in enumerate(sorted_elements):
                    if k >= parsed.max_elements_per_line:
                        omitted_info_line = RL('    (... Omitted %d of %d %s ...) ' % (len(sorted_elements) - parsed.max_elements_per_line, len(sorted_elements), 'tensor(s)' if parsed.tensors else 'op(s)'))
                        omitted_info_line += RL('+5', debugger_cli_common.MenuItem(None, self._reconstruct_print_source_command(parsed, i + 1, max_elements_per_line_increase=5)))
                        labeled_source_lines.append(omitted_info_line)
                        break
                    label = RL(' ' * 4)
                    if self._debug_dump.debug_watch_keys(debug_graphs.get_node_name(element)):
                        attribute = debugger_cli_common.MenuItem('', 'pt %s' % element)
                    else:
                        attribute = cli_shared.COLOR_BLUE
                    label += RL(element, attribute)
                    labeled_source_lines.append(label)
        output = debugger_cli_common.rich_text_lines_from_rich_line_list(labeled_source_lines, annotations={debugger_cli_common.INIT_SCROLL_POS_KEY: actual_initial_scroll_target})
        _add_main_menu(output, node_name=None)
        return output

    def _make_source_table(self, source_list, is_tf_py_library):
        if False:
            return 10
        'Make a table summarizing the source files that create nodes and tensors.\n\n    Args:\n      source_list: List of source files and related information as a list of\n        tuples (file_path, is_tf_library, num_nodes, num_tensors, num_dumps,\n        first_line).\n      is_tf_py_library: (`bool`) whether this table is for files that belong\n        to the TensorFlow Python library.\n\n    Returns:\n      The table as a `debugger_cli_common.RichTextLines` object.\n    '
        path_head = 'Source file path'
        num_nodes_head = '#(nodes)'
        num_tensors_head = '#(tensors)'
        num_dumps_head = '#(tensor dumps)'
        if is_tf_py_library:
            color = cli_shared.COLOR_GRAY
            lines = [RL('TensorFlow Python library file(s):', color)]
        else:
            color = cli_shared.COLOR_WHITE
            lines = [RL('File(s) outside TensorFlow Python library:', color)]
        if not source_list:
            lines.append(RL('[No files.]'))
            lines.append(RL())
            return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)
        path_column_width = max(max((len(item[0]) for item in source_list)), len(path_head)) + 1
        num_nodes_column_width = max(max((len(str(item[2])) for item in source_list)), len(num_nodes_head)) + 1
        num_tensors_column_width = max(max((len(str(item[3])) for item in source_list)), len(num_tensors_head)) + 1
        head = RL(path_head + ' ' * (path_column_width - len(path_head)), color)
        head += RL(num_nodes_head + ' ' * (num_nodes_column_width - len(num_nodes_head)), color)
        head += RL(num_tensors_head + ' ' * (num_tensors_column_width - len(num_tensors_head)), color)
        head += RL(num_dumps_head, color)
        lines.append(head)
        for (file_path, _, num_nodes, num_tensors, num_dumps, first_line_num) in source_list:
            path_attributes = [color]
            if source_utils.is_extension_uncompiled_python_source(file_path):
                path_attributes.append(debugger_cli_common.MenuItem(None, 'ps %s -b %d' % (file_path, first_line_num)))
            line = RL(file_path, path_attributes)
            line += ' ' * (path_column_width - len(line))
            line += RL(str(num_nodes) + ' ' * (num_nodes_column_width - len(str(num_nodes))), color)
            line += RL(str(num_tensors) + ' ' * (num_tensors_column_width - len(str(num_tensors))), color)
            line += RL(str(num_dumps), color)
            lines.append(line)
        lines.append(RL())
        return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

    def list_source(self, args, screen_info=None):
        if False:
            for i in range(10):
                print('nop')
        'List Python source files that constructed nodes and tensors.'
        del screen_info
        parsed = self._arg_parsers['list_source'].parse_args(args)
        source_list = source_utils.list_source_files_against_dump(self._debug_dump, path_regex_allowlist=parsed.path_filter, node_name_regex_allowlist=parsed.node_name_filter)
        top_lines = [RL('List of source files that created nodes in this run', 'bold')]
        if parsed.path_filter:
            top_lines.append(RL('File path regex filter: "%s"' % parsed.path_filter))
        if parsed.node_name_filter:
            top_lines.append(RL('Node name regex filter: "%s"' % parsed.node_name_filter))
        top_lines.append(RL())
        output = debugger_cli_common.rich_text_lines_from_rich_line_list(top_lines)
        if not source_list:
            output.append('[No source file information.]')
            return output
        output.extend(self._make_source_table([item for item in source_list if not item[1]], False))
        output.extend(self._make_source_table([item for item in source_list if item[1]], True))
        _add_main_menu(output, node_name=None)
        return output

    def _list_inputs_or_outputs(self, recursive, node_name, depth, control, op_type, do_outputs=False):
        if False:
            print('Hello World!')
        'Helper function used by list_inputs and list_outputs.\n\n    Format a list of lines to display the inputs or output recipients of a\n    given node.\n\n    Args:\n      recursive: Whether the listing is to be done recursively, as a boolean.\n      node_name: The name of the node in question, as a str.\n      depth: Maximum recursion depth, applies only if recursive == True, as an\n        int.\n      control: Whether control inputs or control recipients are included, as a\n        boolean.\n      op_type: Whether the op types of the nodes are to be included, as a\n        boolean.\n      do_outputs: Whether recipients, instead of input nodes are to be\n        listed, as a boolean.\n\n    Returns:\n      Input or recipient tree formatted as a RichTextLines object.\n    '
        if do_outputs:
            tracker = self._debug_dump.node_recipients
            type_str = 'Recipients of'
            short_type_str = 'recipients'
        else:
            tracker = self._debug_dump.node_inputs
            type_str = 'Inputs to'
            short_type_str = 'inputs'
        lines = []
        font_attr_segs = {}
        (node_name, _) = debug_graphs.parse_node_or_tensor_name(node_name)
        if not self._debug_dump.node_exists(node_name):
            return cli_shared.error('There is no node named "%s" in the partition graphs' % node_name)
        if recursive:
            max_depth = depth
        else:
            max_depth = 1
        if control:
            include_ctrls_str = ', control %s included' % short_type_str
        else:
            include_ctrls_str = ''
        line = '%s node "%s"' % (type_str, node_name)
        font_attr_segs[0] = [(len(line) - 1 - len(node_name), len(line) - 1, 'bold')]
        lines.append(line + ' (Depth limit = %d%s):' % (max_depth, include_ctrls_str))
        command_template = 'lo -c -r %s' if do_outputs else 'li -c -r %s'
        self._dfs_from_node(lines, font_attr_segs, node_name, tracker, max_depth, 1, [], control, op_type, command_template=command_template)
        lines.append('')
        lines.append('Legend:')
        lines.append('  (d): recursion depth = d.')
        if control:
            lines.append('  (Ctrl): Control input.')
        if op_type:
            lines.append('  [Op]: Input node has op type Op.')
        return debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)

    def _dfs_from_node(self, lines, attr_segs, node_name, tracker, max_depth, depth, unfinished, include_control=False, show_op_type=False, command_template=None):
        if False:
            print('Hello World!')
        "Perform depth-first search (DFS) traversal of a node's input tree.\n\n    It recursively tracks the inputs (or output recipients) of the node called\n    node_name, and append these inputs (or output recipients) to a list of text\n    lines (lines) with proper indentation that reflects the recursion depth,\n    together with some formatting attributes (to attr_segs). The formatting\n    attributes can include command shortcuts, for example.\n\n    Args:\n      lines: Text lines to append to, as a list of str.\n      attr_segs: (dict) Attribute segments dictionary to append to.\n      node_name: Name of the node, as a str. This arg is updated during the\n        recursion.\n      tracker: A callable that takes one str as the node name input and\n        returns a list of str as the inputs/outputs.\n        This makes it this function general enough to be used with both\n        node-input and node-output tracking.\n      max_depth: Maximum recursion depth, as an int.\n      depth: Current recursion depth. This arg is updated during the\n        recursion.\n      unfinished: A stack of unfinished recursion depths, as a list of int.\n      include_control: Whether control dependencies are to be included as\n        inputs (and marked as such).\n      show_op_type: Whether op type of the input nodes are to be displayed\n        alongside the nodes' names.\n      command_template: (str) Template for command shortcut of the node names.\n    "
        all_inputs = self._exclude_denylisted_ops(copy.copy(tracker(node_name, is_control=False)))
        is_ctrl = [False] * len(all_inputs)
        if include_control:
            ctrl_inputs = self._exclude_denylisted_ops(sorted(tracker(node_name, is_control=True)))
            all_inputs.extend(ctrl_inputs)
            is_ctrl.extend([True] * len(ctrl_inputs))
        if not all_inputs:
            if depth == 1:
                lines.append('  [None]')
            return
        unfinished.append(depth)
        hang = ''
        for k in range(depth):
            if k < depth - 1:
                if k + 1 in unfinished:
                    hang += HANG_UNFINISHED
                else:
                    hang += HANG_FINISHED
            else:
                hang += HANG_SUFFIX
        if all_inputs and depth > max_depth:
            lines.append(hang + ELLIPSIS)
            unfinished.pop()
            return
        hang += DEPTH_TEMPLATE % depth
        for (i, inp) in enumerate(all_inputs):
            op_type = self._debug_dump.node_op_type(debug_graphs.get_node_name(inp))
            if op_type in self._GRAPH_STRUCT_OP_TYPE_DENYLIST:
                continue
            if is_ctrl[i]:
                ctrl_str = CTRL_LABEL
            else:
                ctrl_str = ''
            op_type_str = ''
            if show_op_type:
                op_type_str = OP_TYPE_TEMPLATE % op_type
            if i == len(all_inputs) - 1:
                unfinished.pop()
            line = hang + ctrl_str + op_type_str + inp
            lines.append(line)
            if command_template:
                attr_segs[len(lines) - 1] = [(len(line) - len(inp), len(line), debugger_cli_common.MenuItem(None, command_template % inp))]
            (inp_node_name, _) = debug_graphs.parse_node_or_tensor_name(inp)
            self._dfs_from_node(lines, attr_segs, inp_node_name, tracker, max_depth, depth + 1, unfinished, include_control=include_control, show_op_type=show_op_type, command_template=command_template)

    def _format_neighbors(self, neighbor_type, non_ctrls, ctrls):
        if False:
            print('Hello World!')
        'List neighbors (inputs or recipients) of a node.\n\n    Args:\n      neighbor_type: ("input" | "recipient")\n      non_ctrls: Non-control neighbor node names, as a list of str.\n      ctrls: Control neighbor node names, as a list of str.\n\n    Returns:\n      A RichTextLines object.\n    '
        lines = []
        font_attr_segs = {}
        lines.append('')
        lines.append('  %d %s(s) + %d control %s(s):' % (len(non_ctrls), neighbor_type, len(ctrls), neighbor_type))
        lines.append('    %d %s(s):' % (len(non_ctrls), neighbor_type))
        for non_ctrl in non_ctrls:
            line = '      [%s] %s' % (self._debug_dump.node_op_type(non_ctrl), non_ctrl)
            lines.append(line)
            font_attr_segs[len(lines) - 1] = [(len(line) - len(non_ctrl), len(line), debugger_cli_common.MenuItem(None, 'ni -a -d -t %s' % non_ctrl))]
        if ctrls:
            lines.append('')
            lines.append('    %d control %s(s):' % (len(ctrls), neighbor_type))
            for ctrl in ctrls:
                line = '      [%s] %s' % (self._debug_dump.node_op_type(ctrl), ctrl)
                lines.append(line)
                font_attr_segs[len(lines) - 1] = [(len(line) - len(ctrl), len(line), debugger_cli_common.MenuItem(None, 'ni -a -d -t %s' % ctrl))]
        return debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)

    def _list_node_attributes(self, node_name):
        if False:
            i = 10
            return i + 15
        'List neighbors (inputs or recipients) of a node.\n\n    Args:\n      node_name: Name of the node of which the attributes are to be listed.\n\n    Returns:\n      A RichTextLines object.\n    '
        lines = []
        lines.append('')
        lines.append('Node attributes:')
        attrs = self._debug_dump.node_attributes(node_name)
        for attr_key in attrs:
            lines.append('  %s:' % attr_key)
            attr_val_str = repr(attrs[attr_key]).strip().replace('\n', ' ')
            lines.append('    %s' % attr_val_str)
            lines.append('')
        return debugger_cli_common.RichTextLines(lines)

    def _list_node_dumps(self, node_name):
        if False:
            for i in range(10):
                print('nop')
        'List dumped tensor data from a node.\n\n    Args:\n      node_name: Name of the node of which the attributes are to be listed.\n\n    Returns:\n      A RichTextLines object.\n    '
        lines = []
        font_attr_segs = {}
        watch_keys = self._debug_dump.debug_watch_keys(node_name)
        dump_count = 0
        for watch_key in watch_keys:
            debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
            for datum in debug_tensor_data:
                line = '  Slot %d @ %s @ %.3f ms' % (datum.output_slot, datum.debug_op, (datum.timestamp - self._debug_dump.t0) / 1000.0)
                lines.append(line)
                command = 'pt %s:%d -n %d' % (node_name, datum.output_slot, dump_count)
                font_attr_segs[len(lines) - 1] = [(2, len(line), debugger_cli_common.MenuItem(None, command))]
                dump_count += 1
        output = debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)
        output_with_header = debugger_cli_common.RichTextLines(['%d dumped tensor(s):' % dump_count, ''])
        output_with_header.extend(output)
        return output_with_header

def create_analyzer_ui(debug_dump, tensor_filters=None, ui_type='readline', on_ui_exit=None, config=None):
    if False:
        i = 10
        return i + 15
    'Create an instance of ReadlineUI based on a DebugDumpDir object.\n\n  Args:\n    debug_dump: (debug_data.DebugDumpDir) The debug dump to use.\n    tensor_filters: (dict) A dict mapping tensor filter name (str) to tensor\n      filter (Callable).\n    ui_type: (str) requested UI type, only "readline" is supported.\n    on_ui_exit: (`Callable`) the callback to be called when the UI exits.\n    config: A `cli_config.CLIConfig` object.\n\n  Returns:\n    (base_ui.BaseUI) A BaseUI subtype object with a set of standard analyzer\n      commands and tab-completions registered.\n  '
    if config is None:
        config = cli_config.CLIConfig()
    analyzer = DebugAnalyzer(debug_dump, config=config)
    if tensor_filters:
        for tensor_filter_name in tensor_filters:
            analyzer.add_tensor_filter(tensor_filter_name, tensor_filters[tensor_filter_name])
    cli = ui_factory.get_ui(ui_type, on_ui_exit=on_ui_exit, config=config)
    cli.register_command_handler('list_tensors', analyzer.list_tensors, analyzer.get_help('list_tensors'), prefix_aliases=['lt'])
    cli.register_command_handler('node_info', analyzer.node_info, analyzer.get_help('node_info'), prefix_aliases=['ni'])
    cli.register_command_handler('list_inputs', analyzer.list_inputs, analyzer.get_help('list_inputs'), prefix_aliases=['li'])
    cli.register_command_handler('list_outputs', analyzer.list_outputs, analyzer.get_help('list_outputs'), prefix_aliases=['lo'])
    cli.register_command_handler('print_tensor', analyzer.print_tensor, analyzer.get_help('print_tensor'), prefix_aliases=['pt'])
    cli.register_command_handler('print_source', analyzer.print_source, analyzer.get_help('print_source'), prefix_aliases=['ps'])
    cli.register_command_handler('list_source', analyzer.list_source, analyzer.get_help('list_source'), prefix_aliases=['ls'])
    cli.register_command_handler('eval', analyzer.evaluate_expression, analyzer.get_help('eval'), prefix_aliases=['ev'])
    dumped_tensor_names = []
    for datum in debug_dump.dumped_tensor_data:
        dumped_tensor_names.append('%s:%d' % (datum.node_name, datum.output_slot))
    cli.register_tab_comp_context(['print_tensor', 'pt'], dumped_tensor_names)
    return cli