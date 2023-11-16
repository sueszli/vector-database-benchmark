"""Shared functions and classes for tfdbg command-line interface."""
import math
import numpy as np
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
RL = debugger_cli_common.RichLine
DEFAULT_NDARRAY_DISPLAY_THRESHOLD = 2000
COLOR_BLACK = 'black'
COLOR_BLUE = 'blue'
COLOR_CYAN = 'cyan'
COLOR_GRAY = 'gray'
COLOR_GREEN = 'green'
COLOR_MAGENTA = 'magenta'
COLOR_RED = 'red'
COLOR_WHITE = 'white'
COLOR_YELLOW = 'yellow'
TIME_UNIT_US = 'us'
TIME_UNIT_MS = 'ms'
TIME_UNIT_S = 's'
TIME_UNITS = [TIME_UNIT_US, TIME_UNIT_MS, TIME_UNIT_S]

def bytes_to_readable_str(num_bytes, include_b=False):
    if False:
        for i in range(10):
            print('nop')
    'Generate a human-readable string representing number of bytes.\n\n  The units B, kB, MB and GB are used.\n\n  Args:\n    num_bytes: (`int` or None) Number of bytes.\n    include_b: (`bool`) Include the letter B at the end of the unit.\n\n  Returns:\n    (`str`) A string representing the number of bytes in a human-readable way,\n      including a unit at the end.\n  '
    if num_bytes is None:
        return str(num_bytes)
    if num_bytes < 1024:
        result = '%d' % num_bytes
    elif num_bytes < 1048576:
        result = '%.2fk' % (num_bytes / 1024.0)
    elif num_bytes < 1073741824:
        result = '%.2fM' % (num_bytes / 1048576.0)
    else:
        result = '%.2fG' % (num_bytes / 1073741824.0)
    if include_b:
        result += 'B'
    return result

def time_to_readable_str(value_us, force_time_unit=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert time value to human-readable string.\n\n  Args:\n    value_us: time value in microseconds.\n    force_time_unit: force the output to use the specified time unit. Must be\n      in TIME_UNITS.\n\n  Returns:\n    Human-readable string representation of the time value.\n\n  Raises:\n    ValueError: if force_time_unit value is not in TIME_UNITS.\n  '
    if not value_us:
        return '0'
    if force_time_unit:
        if force_time_unit not in TIME_UNITS:
            raise ValueError('Invalid time unit: %s' % force_time_unit)
        order = TIME_UNITS.index(force_time_unit)
        time_unit = force_time_unit
        return '{:.10g}{}'.format(value_us / math.pow(10.0, 3 * order), time_unit)
    else:
        order = min(len(TIME_UNITS) - 1, int(math.log(value_us, 10) / 3))
        time_unit = TIME_UNITS[order]
        return '{:.3g}{}'.format(value_us / math.pow(10.0, 3 * order), time_unit)

def parse_ranges_highlight(ranges_string):
    if False:
        for i in range(10):
            print('nop')
    'Process ranges highlight string.\n\n  Args:\n    ranges_string: (str) A string representing a numerical range of a list of\n      numerical ranges. See the help info of the -r flag of the print_tensor\n      command for more details.\n\n  Returns:\n    An instance of tensor_format.HighlightOptions, if range_string is a valid\n      representation of a range or a list of ranges.\n  '
    ranges = None

    def ranges_filter(x):
        if False:
            while True:
                i = 10
        r = np.zeros(x.shape, dtype=bool)
        for (range_start, range_end) in ranges:
            r = np.logical_or(r, np.logical_and(x >= range_start, x <= range_end))
        return r
    if ranges_string:
        ranges = command_parser.parse_ranges(ranges_string)
        return tensor_format.HighlightOptions(ranges_filter, description=ranges_string)
    else:
        return None

def numpy_printoptions_from_screen_info(screen_info):
    if False:
        for i in range(10):
            print('nop')
    if screen_info and 'cols' in screen_info:
        return {'linewidth': screen_info['cols']}
    else:
        return {}

def format_tensor(tensor, tensor_name, np_printoptions, print_all=False, tensor_slicing=None, highlight_options=None, include_numeric_summary=False, write_path=None):
    if False:
        while True:
            i = 10
    'Generate formatted str to represent a tensor or its slices.\n\n  Args:\n    tensor: (numpy ndarray) The tensor value.\n    tensor_name: (str) Name of the tensor, e.g., the tensor\'s debug watch key.\n    np_printoptions: (dict) Numpy tensor formatting options.\n    print_all: (bool) Whether the tensor is to be displayed in its entirety,\n      instead of printing ellipses, even if its number of elements exceeds\n      the default numpy display threshold.\n      (Note: Even if this is set to true, the screen output can still be cut\n       off by the UI frontend if it consist of more lines than the frontend\n       can handle.)\n    tensor_slicing: (str or None) Slicing of the tensor, e.g., "[:, 1]". If\n      None, no slicing will be performed on the tensor.\n    highlight_options: (tensor_format.HighlightOptions) options to highlight\n      elements of the tensor. See the doc of tensor_format.format_tensor()\n      for more details.\n    include_numeric_summary: Whether a text summary of the numeric values (if\n      applicable) will be included.\n    write_path: A path to save the tensor value (after any slicing) to\n      (optional). `numpy.save()` is used to save the value.\n\n  Returns:\n    An instance of `debugger_cli_common.RichTextLines` representing the\n    (potentially sliced) tensor.\n  '
    if tensor_slicing:
        value = command_parser.evaluate_tensor_slice(tensor, tensor_slicing)
        sliced_name = tensor_name + tensor_slicing
    else:
        value = tensor
        sliced_name = tensor_name
    auxiliary_message = None
    if write_path:
        with gfile.Open(write_path, 'wb') as output_file:
            np.save(output_file, value)
        line = debugger_cli_common.RichLine('Saved value to: ')
        line += debugger_cli_common.RichLine(write_path, font_attr='bold')
        line += ' (%sB)' % bytes_to_readable_str(gfile.Stat(write_path).length)
        auxiliary_message = debugger_cli_common.rich_text_lines_from_rich_line_list([line, debugger_cli_common.RichLine('')])
    if print_all:
        np_printoptions['threshold'] = value.size
    else:
        np_printoptions['threshold'] = DEFAULT_NDARRAY_DISPLAY_THRESHOLD
    return tensor_format.format_tensor(value, sliced_name, include_metadata=True, include_numeric_summary=include_numeric_summary, auxiliary_message=auxiliary_message, np_printoptions=np_printoptions, highlight_options=highlight_options)

def error(msg):
    if False:
        i = 10
        return i + 15
    'Generate a RichTextLines output for error.\n\n  Args:\n    msg: (str) The error message.\n\n  Returns:\n    (debugger_cli_common.RichTextLines) A representation of the error message\n      for screen output.\n  '
    return debugger_cli_common.rich_text_lines_from_rich_line_list([RL('ERROR: ' + msg, COLOR_RED)])

def _recommend_command(command, description, indent=2, create_link=False):
    if False:
        for i in range(10):
            print('nop')
    'Generate a RichTextLines object that describes a recommended command.\n\n  Args:\n    command: (str) The command to recommend.\n    description: (str) A description of what the command does.\n    indent: (int) How many spaces to indent in the beginning.\n    create_link: (bool) Whether a command link is to be applied to the command\n      string.\n\n  Returns:\n    (RichTextLines) Formatted text (with font attributes) for recommending the\n      command.\n  '
    indent_str = ' ' * indent
    if create_link:
        font_attr = [debugger_cli_common.MenuItem('', command), 'bold']
    else:
        font_attr = 'bold'
    lines = [RL(indent_str) + RL(command, font_attr) + ':', indent_str + '  ' + description]
    return debugger_cli_common.rich_text_lines_from_rich_line_list(lines)

def get_tfdbg_logo():
    if False:
        return 10
    'Make an ASCII representation of the tfdbg logo.'
    lines = ['', 'TTTTTT FFFF DDD  BBBB   GGG ', '  TT   F    D  D B   B G    ', '  TT   FFF  D  D BBBB  G  GG', '  TT   F    D  D B   B G   G', '  TT   F    DDD  BBBB   GGG ', '']
    return debugger_cli_common.RichTextLines(lines)
_HORIZONTAL_BAR = '======================================'

def get_run_start_intro(run_call_count, fetches, feed_dict, tensor_filters, is_callable_runner=False):
    if False:
        i = 10
        return i + 15
    'Generate formatted intro for run-start UI.\n\n  Args:\n    run_call_count: (int) Run call counter.\n    fetches: Fetches of the `Session.run()` call. See doc of `Session.run()`\n      for more details.\n    feed_dict: Feeds to the `Session.run()` call. See doc of `Session.run()`\n      for more details.\n    tensor_filters: (dict) A dict from tensor-filter name to tensor-filter\n      callable.\n    is_callable_runner: (bool) whether a runner returned by\n        Session.make_callable is being run.\n\n  Returns:\n    (RichTextLines) Formatted intro message about the `Session.run()` call.\n  '
    fetch_lines = common.get_flattened_names(fetches)
    if not feed_dict:
        feed_dict_lines = [debugger_cli_common.RichLine('  (Empty)')]
    else:
        feed_dict_lines = []
        for feed_key in feed_dict:
            feed_key_name = common.get_graph_element_name(feed_key)
            feed_dict_line = debugger_cli_common.RichLine('  ')
            feed_dict_line += debugger_cli_common.RichLine(feed_key_name, debugger_cli_common.MenuItem(None, "pf '%s'" % feed_key_name))
            feed_dict_lines.append(feed_dict_line)
    feed_dict_lines = debugger_cli_common.rich_text_lines_from_rich_line_list(feed_dict_lines)
    out = debugger_cli_common.RichTextLines(_HORIZONTAL_BAR)
    if is_callable_runner:
        out.append('Running a runner returned by Session.make_callable()')
    else:
        out.append('Session.run() call #%d:' % run_call_count)
        out.append('')
        out.append('Fetch(es):')
        out.extend(debugger_cli_common.RichTextLines(['  ' + line for line in fetch_lines]))
        out.append('')
        out.append('Feed dict:')
        out.extend(feed_dict_lines)
    out.append(_HORIZONTAL_BAR)
    out.append('')
    out.append('Select one of the following commands to proceed ---->')
    out.extend(_recommend_command('run', 'Execute the run() call with debug tensor-watching', create_link=True))
    out.extend(_recommend_command('run -n', 'Execute the run() call without debug tensor-watching', create_link=True))
    out.extend(_recommend_command('run -t <T>', 'Execute run() calls (T - 1) times without debugging, then execute run() once more with debugging and drop back to the CLI'))
    out.extend(_recommend_command('run -f <filter_name>', 'Keep executing run() calls until a dumped tensor passes a given, registered filter (conditional breakpoint mode)'))
    more_lines = ['    Registered filter(s):']
    if tensor_filters:
        filter_names = []
        for filter_name in tensor_filters:
            filter_names.append(filter_name)
            command_menu_node = debugger_cli_common.MenuItem('', 'run -f %s' % filter_name)
            more_lines.append(RL('        * ') + RL(filter_name, command_menu_node))
    else:
        more_lines.append('        (None)')
    out.extend(debugger_cli_common.rich_text_lines_from_rich_line_list(more_lines))
    out.append('')
    out.append_rich_line(RL('For more details, see ') + RL('help.', debugger_cli_common.MenuItem('', 'help')) + '.')
    out.append('')
    menu = debugger_cli_common.Menu()
    menu.append(debugger_cli_common.MenuItem('run', 'run'))
    menu.append(debugger_cli_common.MenuItem('exit', 'exit'))
    out.annotations[debugger_cli_common.MAIN_MENU_KEY] = menu
    return out

def get_run_short_description(run_call_count, fetches, feed_dict, is_callable_runner=False):
    if False:
        return 10
    'Get a short description of the run() call.\n\n  Args:\n    run_call_count: (int) Run call counter.\n    fetches: Fetches of the `Session.run()` call. See doc of `Session.run()`\n      for more details.\n    feed_dict: Feeds to the `Session.run()` call. See doc of `Session.run()`\n      for more details.\n    is_callable_runner: (bool) whether a runner returned by\n        Session.make_callable is being run.\n\n  Returns:\n    (str) A short description of the run() call, including information about\n      the fetche(s) and feed(s).\n  '
    if is_callable_runner:
        return 'runner from make_callable()'
    description = 'run #%d: ' % run_call_count
    if isinstance(fetches, (tensor_lib.Tensor, ops.Operation, variables.Variable)):
        description += '1 fetch (%s); ' % common.get_graph_element_name(fetches)
    else:
        num_fetches = len(common.get_flattened_names(fetches))
        if num_fetches > 1:
            description += '%d fetches; ' % num_fetches
        else:
            description += '%d fetch; ' % num_fetches
    if not feed_dict:
        description += '0 feeds'
    elif len(feed_dict) == 1:
        for key in feed_dict:
            description += '1 feed (%s)' % (key if isinstance(key, str) or not hasattr(key, 'name') else key.name)
    else:
        description += '%d feeds' % len(feed_dict)
    return description

def get_error_intro(tf_error):
    if False:
        while True:
            i = 10
    'Generate formatted intro for TensorFlow run-time error.\n\n  Args:\n    tf_error: (errors.OpError) TensorFlow run-time error object.\n\n  Returns:\n    (RichTextLines) Formatted intro message about the run-time OpError, with\n      sample commands for debugging.\n  '
    if hasattr(tf_error, 'op') and hasattr(tf_error.op, 'name'):
        op_name = tf_error.op.name
    else:
        op_name = None
    intro_lines = ['--------------------------------------', RL('!!! An error occurred during the run !!!', 'blink'), '']
    out = debugger_cli_common.rich_text_lines_from_rich_line_list(intro_lines)
    if op_name is not None:
        out.extend(debugger_cli_common.RichTextLines(['You may use the following commands to debug:']))
        out.extend(_recommend_command('ni -a -d -t %s' % op_name, 'Inspect information about the failing op.', create_link=True))
        out.extend(_recommend_command('li -r %s' % op_name, 'List inputs to the failing op, recursively.', create_link=True))
        out.extend(_recommend_command('lt', 'List all tensors dumped during the failing run() call.', create_link=True))
    else:
        out.extend(debugger_cli_common.RichTextLines(['WARNING: Cannot determine the name of the op that caused the error.']))
    more_lines = ['', 'Op name:    %s' % op_name, 'Error type: ' + str(type(tf_error)), '', 'Details:', str(tf_error), '', '--------------------------------------', '']
    out.extend(debugger_cli_common.RichTextLines(more_lines))
    return out