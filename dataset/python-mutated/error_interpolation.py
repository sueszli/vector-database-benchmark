"""Function for interpolating formatted errors from the TensorFlow runtime.

Exposes the function `interpolate` to interpolate messages with tags of the form
{{type name}}.
"""
import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
_NAME_REGEX = '[A-Za-z0-9_.][A-Za-z0-9_.\\-/]*?'
_TAG_REGEX = f'{{{{(?P<type>{_NAME_REGEX}) (?P<name>{_NAME_REGEX})}}}}'
_INTERPOLATION_REGEX = f'(?P<sep>.*?)(?P<tag>{_TAG_REGEX})'
_INTERPOLATION_PATTERN = re.compile(_INTERPOLATION_REGEX, re.DOTALL)
_ParseTag = collections.namedtuple('_ParseTag', ['type', 'name'])
_FRAMEWORK_COMMON_PREFIX = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_FRAMEWORK_PATH_PREFIXES = [os.path.join(_FRAMEWORK_COMMON_PREFIX, 'python') + os.sep, os.path.join(_FRAMEWORK_COMMON_PREFIX, 'contrib') + os.sep, os.path.join(os.path.dirname(_FRAMEWORK_COMMON_PREFIX), 'py', 'keras') + os.sep]
_FRAMEWORK_FILENAME_PATTERNS = [re.compile('<embedded')]
try:
    _FRAMEWORK_PATH_PREFIXES.extend([os.path.join(package_path, 'keras') + os.sep for package_path in site.getsitepackages() + [site.getusersitepackages()]])
except AttributeError:
    _FRAMEWORK_FILENAME_PATTERNS.append(re.compile('keras'))
_EXTERNAL_FILENAME_PATTERNS = [re.compile('_test\\.py$')]

def parse_message(message):
    if False:
        while True:
            i = 10
    'Extract function tags and node tags from a message.\n\n  Tags are named tuples representing the string {{type name}}. For example,\n  in "123{{node Foo}}456{{function_node Bar}}789", there are two tags: a node\n  tag and a function tag.\n\n  Args:\n    message: An error message, possibly from an OpError.\n\n  Returns:\n    A tuple containing the original message with function nodes stripped,\n    function tags, and node tags.\n\n    For example, if message is "123{{node Foo}}456{{function_node Bar}}789"\n    then this function returns ("123{{node Foo}}456789",\n    [_ParseTag("function_node", "Bar")], [_ParseTag("node", "Foo")]).\n  '
    error_message = []
    func_tags = []
    node_tags = []
    pos = 0
    for match in re.finditer(_INTERPOLATION_PATTERN, message):
        parsed_tag = _ParseTag(match.group('type'), match.group('name'))
        if parsed_tag.type == 'function_node':
            error_message.append(match.group('sep'))
            func_tags.append(parsed_tag)
        else:
            error_message.append(match.group())
            node_tags.append(parsed_tag)
        pos = match.end()
    error_message.append(message[pos:])
    return (''.join(error_message), func_tags, node_tags)

def _compute_device_summary_from_list(name, device_assignment_list, prefix=''):
    if False:
        print('Hello World!')
    "Return a summary of an op's device function stack.\n\n  Args:\n    name: The name of the op.\n    device_assignment_list: The op._device_assignments list.\n    prefix:  An optional string prefix used before each line of the multi-\n        line string returned by this function.\n\n  Returns:\n    A multi-line string similar to:\n        Device assignments active during op 'foo' creation:\n          with tf.device(/cpu:0): <test_1.py:27>\n          with tf.device(some_func<foo.py, 123>): <test_2.py:38>\n    The first line will have no padding to its left by default.  Subsequent\n    lines will have two spaces of left-padding.  Use the prefix argument\n    to increase indentation.\n  "
    if not device_assignment_list:
        message = "No device assignments were active during op '%s' creation."
        message %= name
        return prefix + message
    str_list = []
    str_list.append("%sDevice assignments active during op '%s' creation:" % (prefix, name))
    for traceable_obj in device_assignment_list:
        location_summary = '<{file}:{line}>'.format(file=traceable_obj.filename, line=traceable_obj.lineno)
        subs = {'prefix': prefix, 'indent': '  ', 'dev_name': traceable_obj.obj, 'loc': location_summary}
        str_list.append('{prefix}{indent}with tf.device({dev_name}): {loc}'.format(**subs))
    return '\n'.join(str_list)

def _compute_device_assignment_summary_from_op(op, prefix=''):
    if False:
        while True:
            i = 10
    return _compute_device_summary_from_list(op.name, op._device_assignments, prefix)

def _compute_colocation_summary_from_dict(name, colocation_dict, prefix=''):
    if False:
        for i in range(10):
            print('nop')
    "Return a summary of an op's colocation stack.\n\n  Args:\n    name: The op name.\n    colocation_dict: The op._colocation_dict.\n    prefix:  An optional string prefix used before each line of the multi-\n        line string returned by this function.\n\n  Returns:\n    A multi-line string similar to:\n        Node-device colocations active during op creation:\n          with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>\n          with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>\n    The first line will have no padding to its left by default.  Subsequent\n    lines will have two spaces of left-padding.  Use the prefix argument\n    to increase indentation.\n  "
    if not colocation_dict:
        message = "No node-device colocations were active during op '%s' creation."
        message %= name
        return prefix + message
    str_list = []
    str_list.append("%sNode-device colocations active during op '%s' creation:" % (prefix, name))
    for (coloc_name, location) in colocation_dict.items():
        location_summary = '<{file}:{line}>'.format(file=location.filename, line=location.lineno)
        subs = {'prefix': prefix, 'indent': '  ', 'name': coloc_name, 'loc': location_summary}
        str_list.append('{prefix}{indent}with tf.colocate_with({name}): {loc}'.format(**subs))
    return '\n'.join(str_list)

def _compute_colocation_summary_from_op(op, prefix=''):
    if False:
        for i in range(10):
            print('nop')
    'Fetch colocation file, line, and nesting and return a summary string.'
    return _compute_colocation_summary_from_dict(op.name, op._colocation_dict, prefix)

def _is_framework_filename(filename):
    if False:
        for i in range(10):
            print('nop')
    'Returns whether a filename should be considered a part of the framework.\n\n  A file is part of the framework if it does not match a pattern in\n  _EXTERNAL_FILENAME_PATTERNS and it either matches a pattern in\n  _FRAMEWORK_FILENAME_PATTERNS or starts with a _FRAMEWORK_PATH_PREFIXES prefix.\n\n  Args:\n    filename: A filename string.\n\n  Returns:\n    Whether the filename should be considered to be internal to the\n    TensorFlow framework for the purposes of reporting errors.\n  '
    for pattern in _EXTERNAL_FILENAME_PATTERNS:
        if pattern.search(filename):
            return False
    for pattern in _FRAMEWORK_FILENAME_PATTERNS:
        if pattern.search(filename):
            return True
    for prefix in _FRAMEWORK_PATH_PREFIXES:
        if filename.startswith(prefix):
            return True
    return False

def _find_index_of_defining_frame(tb):
    if False:
        i = 10
        return i + 15
    "Return index in op.traceback with first 'useful' frame.\n\n  This method reads through the stack stored in op.traceback looking for the\n  innermost frame which (hopefully) belongs to the caller.  It accomplishes this\n  by rejecting frames deemed to be part of the TensorFlow framework (by\n  pattern matching the filename).\n\n  Args:\n    tb: A list of traceback frames (as from Operation.traceback).\n\n  Returns:\n    Integer index into op.traceback where the first non-TF file was found\n    (innermost to outermost), or 0 (for the outermost stack frame) if all files\n    came from TensorFlow.\n  "
    size = len(tb)
    filenames = [frame.filename for frame in tb]
    for (idx, filename) in enumerate(reversed(filenames)):
        is_framework = _is_framework_filename(filename)
        if not is_framework:
            return size - idx - 1
    return 0

def _compute_useful_frames(tb, num):
    if False:
        for i in range(10):
            print('nop')
    "Return a list of frames, which form a 'useful' stack.\n\n  Starting from the defining frame to the outermost one, this method computes\n  the contiguous portion of the 'useful' stack trace and returns the selected\n  frames.\n\n  Args:\n    tb: A list of traceback frames (as from Operation.traceback).\n    num: total number of frames to return.\n\n  Returns:\n    A list of frames.\n  "
    defining_frame_index = _find_index_of_defining_frame(tb)
    innermost_excluded = min(defining_frame_index + 2 + 1, len(tb))
    outermost_included = max(innermost_excluded - num, 0)
    return tb[outermost_included:innermost_excluded]

def create_graph_debug_info_def(func_named_operations):
    if False:
        return 10
    'Construct and returns a `GraphDebugInfo` protocol buffer.\n\n  Args:\n    func_named_operations: An iterable of (func_name, op.Operation) tuples\n      where the Operation instances have a _traceback members. The func_name\n      should be the empty string for operations in the top-level Graph.\n\n  Returns:\n    GraphDebugInfo protocol buffer.\n\n  Raises:\n    TypeError: If the arguments are not of the correct proto buffer type.\n  '
    builder = tf_stack.GraphDebugInfoBuilder()
    for (func_name, op) in func_named_operations:
        if op.traceback is None:
            continue
        builder.AccumulateStackTrace(func_name, op.name, _compute_useful_frames(op.traceback, 10))
    return builder.Build()

def _compute_field_dict(op):
    if False:
        return 10
    'Return a dictionary mapping interpolation tokens to values.\n\n  Args:\n    op: op.Operation object.\n\n  Returns:\n    A dictionary mapping string tokens to string values.  The keys are shown\n    below along with example values.\n    {\n      "file": "tool_utils.py",\n      "lineno": "124",\n      "line": "  source code line",\n      "defined_at": " (defined at tool_utils.py:124)",\n      "colocations":\n          \'\'\'Node-device colocations active during op creation:\n               with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>\n               with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>\'\'\'\n      "devices":\n          \'\'\'Device assignments active during op \'foo\' creation:\n               with tf.device(/cpu:0): <test_1.py:27>\n               with tf.device(some_func<foo.py, 123>): <test_2.py:38>\'\'\'\n      "devs_and_colocs": A concatenation of colocations and devices, e.g.\n          \'\'\'Node-device colocations active during op creation:\n               with tf.compat.v1.colocate_with(test_node_1): <test_1.py:27>\n               with tf.compat.v1.colocate_with(test_node_2): <test_2.py:38>\'\'\'\n             Device assignments active during op \'foo\' creation:\n               with tf.device(/cpu:0): <test_1.py:27>\n               with tf.device(some_func<foo.py, 123>): <test_2.py:38>\'\'\'\n    }\n  '
    colocation_summary = _compute_colocation_summary_from_op(op)
    device_summary = _compute_device_assignment_summary_from_op(op)
    combined_summary = '\n'.join([colocation_summary, device_summary])
    if op.traceback is None:
        filename = '<unknown>'
        definition_traceback = ''
        lineno = 0
        line = ''
        defined_at = '<unknown>'
    else:
        frame = op.traceback.last_user_frame()
        filename = frame.filename
        definition_traceback = traceback.format_list(op.traceback.get_user_frames())
        lineno = frame.lineno
        line = frame.line
        defined_at = f'{filename}:{lineno:d}'
    field_dict = {'colocations': colocation_summary, 'devices': device_summary, 'devs_and_colocs': combined_summary, 'defined_at': defined_at, 'file': filename, 'lineno': lineno, 'line': line, 'definition_traceback': definition_traceback}
    return field_dict

def _build_node_error_message(op):
    if False:
        for i in range(10):
            print('nop')
    'Returns the formatted error message for the given op.\n\n  Args:\n    op: The node.\n\n  Returns:\n    The formatted error message for the given op with traceback.\n  '
    node_error_message = [f'Detected at node {op.name!r} defined at (most recent call last):']
    field_dict = _compute_field_dict(op)
    for frame in field_dict['definition_traceback']:
        if '<embedded' not in frame:
            node_error_message.extend([f'  {line}' for line in frame.split('\n') if line.strip()])
    node_error_message.append(f'Node: {op.name!r}')
    return '\n'.join(node_error_message)

def interpolate_graph(message, graph):
    if False:
        i = 10
        return i + 15
    'Interpolates an error message.\n\n  The error message can contain tags of form `{{node_type node_name}}`\n  which will be parsed to identify the tf.Graph and op. If the op contains\n  traceback, the traceback will be attached to the error message.\n\n  Args:\n    message: A string to interpolate.\n    graph: ops.Graph object containing all nodes referenced in the error\n        message.\n\n  Returns:\n    The error message string with node definition traceback.\n  '
    (parsed_messaged, _, node_tags) = parse_message(message)
    error_message = ['Graph execution error:', '']
    for tag in node_tags:
        try:
            op = graph.get_operation_by_name(tag.name)
        except KeyError:
            continue
        else:
            error_message.append(_build_node_error_message(op))
    error_message.append(parsed_messaged.strip())
    return '\n'.join(error_message)