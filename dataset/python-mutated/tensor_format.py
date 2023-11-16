"""Format tensors (ndarrays) for screen display and navigation."""
import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
_NUMPY_OMISSION = '...,'
_NUMPY_DEFAULT_EDGE_ITEMS = 3
_NUMBER_REGEX = re.compile('[-+]?([0-9][-+0-9eE\\.]+|nan|inf)(\\s|,|\\])')
BEGIN_INDICES_KEY = 'i0'
OMITTED_INDICES_KEY = 'omitted'
DEFAULT_TENSOR_ELEMENT_HIGHLIGHT_FONT_ATTR = 'bold'

class HighlightOptions(object):
    """Options for highlighting elements of a tensor."""

    def __init__(self, criterion, description=None, font_attr=DEFAULT_TENSOR_ELEMENT_HIGHLIGHT_FONT_ATTR):
        if False:
            for i in range(10):
                print('nop')
        'Constructor of HighlightOptions.\n\n    Args:\n      criterion: (callable) A callable of the following signature:\n        def to_highlight(X):\n          # Args:\n          #   X: The tensor to highlight elements in.\n          #\n          # Returns:\n          #   (boolean ndarray) A boolean ndarray of the same shape as X\n          #   indicating which elements are to be highlighted (iff True).\n        This callable will be used as the argument of np.argwhere() to\n        determine which elements of the tensor are to be highlighted.\n      description: (str) Description of the highlight criterion embodied by\n        criterion.\n      font_attr: (str) Font attribute to be applied to the\n        highlighted elements.\n\n    '
        self.criterion = criterion
        self.description = description
        self.font_attr = font_attr

def format_tensor(tensor, tensor_label, include_metadata=False, auxiliary_message=None, include_numeric_summary=False, np_printoptions=None, highlight_options=None):
    if False:
        for i in range(10):
            print('nop')
    'Generate a RichTextLines object showing a tensor in formatted style.\n\n  Args:\n    tensor: The tensor to be displayed, as a numpy ndarray or other\n      appropriate format (e.g., None representing uninitialized tensors).\n    tensor_label: A label for the tensor, as a string. If set to None, will\n      suppress the tensor name line in the return value.\n    include_metadata: Whether metadata such as dtype and shape are to be\n      included in the formatted text.\n    auxiliary_message: An auxiliary message to display under the tensor label,\n      dtype and shape information lines.\n    include_numeric_summary: Whether a text summary of the numeric values (if\n      applicable) will be included.\n    np_printoptions: A dictionary of keyword arguments that are passed to a\n      call of np.set_printoptions() to set the text format for display numpy\n      ndarrays.\n    highlight_options: (HighlightOptions) options for highlighting elements\n      of the tensor.\n\n  Returns:\n    A RichTextLines object. Its annotation field has line-by-line markups to\n    indicate which indices in the array the first element of each line\n    corresponds to.\n  '
    lines = []
    font_attr_segs = {}
    if tensor_label is not None:
        lines.append('Tensor "%s":' % tensor_label)
        suffix = tensor_label.split(':')[-1]
        if suffix.isdigit():
            font_attr_segs[0] = [(8, 8 + len(tensor_label), 'bold')]
        else:
            debug_op_len = len(suffix)
            proper_len = len(tensor_label) - debug_op_len - 1
            font_attr_segs[0] = [(8, 8 + proper_len, 'bold'), (8 + proper_len + 1, 8 + proper_len + 1 + debug_op_len, 'yellow')]
    if isinstance(tensor, debug_data.InconvertibleTensorProto):
        if lines:
            lines.append('')
        lines.extend(str(tensor).split('\n'))
        return debugger_cli_common.RichTextLines(lines)
    elif not isinstance(tensor, np.ndarray):
        if lines:
            lines.append('')
        lines.extend(repr(tensor).split('\n'))
        return debugger_cli_common.RichTextLines(lines)
    if include_metadata:
        lines.append('  dtype: %s' % str(tensor.dtype))
        lines.append('  shape: %s' % str(tensor.shape).replace('L', ''))
    if lines:
        lines.append('')
    formatted = debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)
    if auxiliary_message:
        formatted.extend(auxiliary_message)
    if include_numeric_summary:
        formatted.append('Numeric summary:')
        formatted.extend(numeric_summary(tensor))
        formatted.append('')
    if np_printoptions is not None:
        np.set_printoptions(**np_printoptions)
    array_lines = repr(tensor).split('\n')
    if tensor.dtype.type is not np.string_:
        annotations = _annotate_ndarray_lines(array_lines, tensor, np_printoptions=np_printoptions)
    else:
        annotations = None
    formatted_array = debugger_cli_common.RichTextLines(array_lines, annotations=annotations)
    formatted.extend(formatted_array)
    if highlight_options is not None:
        indices_list = list(np.argwhere(highlight_options.criterion(tensor)))
        total_elements = np.size(tensor)
        highlight_summary = 'Highlighted%s: %d of %d element(s) (%.2f%%)' % ('(%s)' % highlight_options.description if highlight_options.description else '', len(indices_list), total_elements, len(indices_list) / float(total_elements) * 100.0)
        formatted.lines[0] += ' ' + highlight_summary
        if indices_list:
            indices_list = [list(indices) for indices in indices_list]
            (are_omitted, rows, start_cols, end_cols) = locate_tensor_element(formatted, indices_list)
            for (is_omitted, row, start_col, end_col) in zip(are_omitted, rows, start_cols, end_cols):
                if is_omitted or start_col is None or end_col is None:
                    continue
                if row in formatted.font_attr_segs:
                    formatted.font_attr_segs[row].append((start_col, end_col, highlight_options.font_attr))
                else:
                    formatted.font_attr_segs[row] = [(start_col, end_col, highlight_options.font_attr)]
    return formatted

def _annotate_ndarray_lines(array_lines, tensor, np_printoptions=None, offset=0):
    if False:
        print('Hello World!')
    'Generate annotations for line-by-line begin indices of tensor text.\n\n  Parse the numpy-generated text representation of a numpy ndarray to\n  determine the indices of the first element of each text line (if any\n  element is present in the line).\n\n  For example, given the following multi-line ndarray text representation:\n      ["array([[ 0.    ,  0.0625,  0.125 ,  0.1875],",\n       "       [ 0.25  ,  0.3125,  0.375 ,  0.4375],",\n       "       [ 0.5   ,  0.5625,  0.625 ,  0.6875],",\n       "       [ 0.75  ,  0.8125,  0.875 ,  0.9375]])"]\n  the generate annotation will be:\n      {0: {BEGIN_INDICES_KEY: [0, 0]},\n       1: {BEGIN_INDICES_KEY: [1, 0]},\n       2: {BEGIN_INDICES_KEY: [2, 0]},\n       3: {BEGIN_INDICES_KEY: [3, 0]}}\n\n  Args:\n    array_lines: Text lines representing the tensor, as a list of str.\n    tensor: The tensor being formatted as string.\n    np_printoptions: A dictionary of keyword arguments that are passed to a\n      call of np.set_printoptions().\n    offset: Line number offset applied to the line indices in the returned\n      annotation.\n\n  Returns:\n    An annotation as a dict.\n  '
    if np_printoptions and 'edgeitems' in np_printoptions:
        edge_items = np_printoptions['edgeitems']
    else:
        edge_items = _NUMPY_DEFAULT_EDGE_ITEMS
    annotations = {}
    annotations['tensor_metadata'] = {'dtype': tensor.dtype, 'shape': tensor.shape}
    dims = np.shape(tensor)
    ndims = len(dims)
    if ndims == 0:
        return annotations
    curr_indices = [0] * len(dims)
    curr_dim = 0
    for (i, raw_line) in enumerate(array_lines):
        line = raw_line.strip()
        if not line:
            continue
        if line == _NUMPY_OMISSION:
            annotations[offset + i] = {OMITTED_INDICES_KEY: copy.copy(curr_indices)}
            curr_indices[curr_dim - 1] = dims[curr_dim - 1] - edge_items
        else:
            num_lbrackets = line.count('[')
            num_rbrackets = line.count(']')
            curr_dim += num_lbrackets - num_rbrackets
            annotations[offset + i] = {BEGIN_INDICES_KEY: copy.copy(curr_indices)}
            if num_rbrackets == 0:
                line_content = line[line.rfind('[') + 1:]
                num_elements = line_content.count(',')
                curr_indices[curr_dim - 1] += num_elements
            elif curr_dim > 0:
                curr_indices[curr_dim - 1] += 1
                for k in range(curr_dim, ndims):
                    curr_indices[k] = 0
    return annotations

def locate_tensor_element(formatted, indices):
    if False:
        i = 10
        return i + 15
    'Locate a tensor element in formatted text lines, given element indices.\n\n  Given a RichTextLines object representing a tensor and indices of the sought\n  element, return the row number at which the element is located (if exists).\n\n  Args:\n    formatted: A RichTextLines object containing formatted text lines\n      representing the tensor.\n    indices: Indices of the sought element, as a list of int or a list of list\n      of int. The former case is for a single set of indices to look up,\n      whereas the latter case is for looking up a batch of indices sets at once.\n      In the latter case, the indices must be in ascending order, or a\n      ValueError will be raised.\n\n  Returns:\n    1) A boolean indicating whether the element falls into an omitted line.\n    2) Row index.\n    3) Column start index, i.e., the first column in which the representation\n       of the specified tensor starts, if it can be determined. If it cannot\n       be determined (e.g., due to ellipsis), None.\n    4) Column end index, i.e., the column right after the last column that\n       represents the specified tensor. Iff it cannot be determined, None.\n\n  For return values described above are based on a single set of indices to\n    look up. In the case of batch mode (multiple sets of indices), the return\n    values will be lists of the types described above.\n\n  Raises:\n    AttributeError: If:\n      Input argument "formatted" does not have the required annotations.\n    ValueError: If:\n      1) Indices do not match the dimensions of the tensor, or\n      2) Indices exceed sizes of the tensor, or\n      3) Indices contain negative value(s).\n      4) If in batch mode, and if not all sets of indices are in ascending\n         order.\n  '
    if isinstance(indices[0], list):
        indices_list = indices
        input_batch = True
    else:
        indices_list = [indices]
        input_batch = False
    if 'tensor_metadata' not in formatted.annotations:
        raise AttributeError('tensor_metadata is not available in annotations.')
    _validate_indices_list(indices_list, formatted)
    dims = formatted.annotations['tensor_metadata']['shape']
    batch_size = len(indices_list)
    lines = formatted.lines
    annot = formatted.annotations
    prev_r = 0
    prev_line = ''
    prev_indices = [0] * len(dims)
    are_omitted = [None] * batch_size
    row_indices = [None] * batch_size
    start_columns = [None] * batch_size
    end_columns = [None] * batch_size
    batch_pos = 0
    for r in range(len(lines)):
        if r not in annot:
            continue
        if BEGIN_INDICES_KEY in annot[r]:
            indices_key = BEGIN_INDICES_KEY
        elif OMITTED_INDICES_KEY in annot[r]:
            indices_key = OMITTED_INDICES_KEY
        matching_indices_list = [ind for ind in indices_list[batch_pos:] if prev_indices <= ind < annot[r][indices_key]]
        if matching_indices_list:
            num_matches = len(matching_indices_list)
            (match_start_columns, match_end_columns) = _locate_elements_in_line(prev_line, matching_indices_list, prev_indices)
            start_columns[batch_pos:batch_pos + num_matches] = match_start_columns
            end_columns[batch_pos:batch_pos + num_matches] = match_end_columns
            are_omitted[batch_pos:batch_pos + num_matches] = [OMITTED_INDICES_KEY in annot[prev_r]] * num_matches
            row_indices[batch_pos:batch_pos + num_matches] = [prev_r] * num_matches
            batch_pos += num_matches
            if batch_pos >= batch_size:
                break
        prev_r = r
        prev_line = lines[r]
        prev_indices = annot[r][indices_key]
    if batch_pos < batch_size:
        matching_indices_list = indices_list[batch_pos:]
        num_matches = len(matching_indices_list)
        (match_start_columns, match_end_columns) = _locate_elements_in_line(prev_line, matching_indices_list, prev_indices)
        start_columns[batch_pos:batch_pos + num_matches] = match_start_columns
        end_columns[batch_pos:batch_pos + num_matches] = match_end_columns
        are_omitted[batch_pos:batch_pos + num_matches] = [OMITTED_INDICES_KEY in annot[prev_r]] * num_matches
        row_indices[batch_pos:batch_pos + num_matches] = [prev_r] * num_matches
    if input_batch:
        return (are_omitted, row_indices, start_columns, end_columns)
    else:
        return (are_omitted[0], row_indices[0], start_columns[0], end_columns[0])

def _validate_indices_list(indices_list, formatted):
    if False:
        print('Hello World!')
    prev_ind = None
    for ind in indices_list:
        dims = formatted.annotations['tensor_metadata']['shape']
        if len(ind) != len(dims):
            raise ValueError('Dimensions mismatch: requested: %d; actual: %d' % (len(ind), len(dims)))
        for (req_idx, siz) in zip(ind, dims):
            if req_idx >= siz:
                raise ValueError('Indices exceed tensor dimensions.')
            if req_idx < 0:
                raise ValueError('Indices contain negative value(s).')
        if prev_ind and ind < prev_ind:
            raise ValueError('Input indices sets are not in ascending order.')
        prev_ind = ind

def _locate_elements_in_line(line, indices_list, ref_indices):
    if False:
        for i in range(10):
            print('nop')
    'Determine the start and end indices of an element in a line.\n\n  Args:\n    line: (str) the line in which the element is to be sought.\n    indices_list: (list of list of int) list of indices of the element to\n       search for. Assumes that the indices in the batch are unique and sorted\n       in ascending order.\n    ref_indices: (list of int) reference indices, i.e., the indices of the\n      first element represented in the line.\n\n  Returns:\n    start_columns: (list of int) start column indices, if found. If not found,\n      None.\n    end_columns: (list of int) end column indices, if found. If not found,\n      None.\n    If found, the element is represented in the left-closed-right-open interval\n      [start_column, end_column].\n  '
    batch_size = len(indices_list)
    offsets = [indices[-1] - ref_indices[-1] for indices in indices_list]
    start_columns = [None] * batch_size
    end_columns = [None] * batch_size
    if _NUMPY_OMISSION in line:
        ellipsis_index = line.find(_NUMPY_OMISSION)
    else:
        ellipsis_index = len(line)
    matches_iter = re.finditer(_NUMBER_REGEX, line)
    batch_pos = 0
    offset_counter = 0
    for match in matches_iter:
        if match.start() > ellipsis_index:
            break
        if offset_counter == offsets[batch_pos]:
            start_columns[batch_pos] = match.start()
            end_columns[batch_pos] = match.end() - 1
            batch_pos += 1
            if batch_pos >= batch_size:
                break
        offset_counter += 1
    return (start_columns, end_columns)

def _pad_string_to_length(string, length):
    if False:
        while True:
            i = 10
    return ' ' * (length - len(string)) + string

def numeric_summary(tensor):
    if False:
        i = 10
        return i + 15
    'Get a text summary of a numeric tensor.\n\n  This summary is only available for numeric (int*, float*, complex*) and\n  Boolean tensors.\n\n  Args:\n    tensor: (`numpy.ndarray`) the tensor value object to be summarized.\n\n  Returns:\n    The summary text as a `RichTextLines` object. If the type of `tensor` is not\n    numeric or Boolean, a single-line `RichTextLines` object containing a\n    warning message will reflect that.\n  '

    def _counts_summary(counts, skip_zeros=True, total_count=None):
        if False:
            for i in range(10):
                print('nop')
        'Format values as a two-row table.'
        if skip_zeros:
            counts = [(count_key, count_val) for (count_key, count_val) in counts if count_val]
        max_common_len = 0
        for (count_key, count_val) in counts:
            count_val_str = str(count_val)
            common_len = max(len(count_key) + 1, len(count_val_str) + 1)
            max_common_len = max(common_len, max_common_len)
        key_line = debugger_cli_common.RichLine('|')
        val_line = debugger_cli_common.RichLine('|')
        for (count_key, count_val) in counts:
            count_val_str = str(count_val)
            key_line += _pad_string_to_length(count_key, max_common_len)
            val_line += _pad_string_to_length(count_val_str, max_common_len)
        key_line += ' |'
        val_line += ' |'
        if total_count is not None:
            total_key_str = 'total'
            total_val_str = str(total_count)
            max_common_len = max(len(total_key_str) + 1, len(total_val_str))
            total_key_str = _pad_string_to_length(total_key_str, max_common_len)
            total_val_str = _pad_string_to_length(total_val_str, max_common_len)
            key_line += total_key_str + ' |'
            val_line += total_val_str + ' |'
        return debugger_cli_common.rich_text_lines_from_rich_line_list([key_line, val_line])
    if not isinstance(tensor, np.ndarray) or not np.size(tensor):
        return debugger_cli_common.RichTextLines(['No numeric summary available due to empty tensor.'])
    elif np.issubdtype(tensor.dtype, np.floating) or np.issubdtype(tensor.dtype, np.complexfloating) or np.issubdtype(tensor.dtype, np.integer):
        counts = [('nan', np.sum(np.isnan(tensor))), ('-inf', np.sum(np.isneginf(tensor))), ('-', np.sum(np.logical_and(tensor < 0.0, np.logical_not(np.isneginf(tensor))))), ('0', np.sum(tensor == 0.0)), ('+', np.sum(np.logical_and(tensor > 0.0, np.logical_not(np.isposinf(tensor))))), ('+inf', np.sum(np.isposinf(tensor)))]
        output = _counts_summary(counts, total_count=np.size(tensor))
        valid_array = tensor[np.logical_not(np.logical_or(np.isinf(tensor), np.isnan(tensor)))]
        if np.size(valid_array):
            stats = [('min', np.min(valid_array)), ('max', np.max(valid_array)), ('mean', np.mean(valid_array)), ('std', np.std(valid_array))]
            output.extend(_counts_summary(stats, skip_zeros=False))
        return output
    elif tensor.dtype == np.bool_:
        counts = [('False', np.sum(tensor == 0)), ('True', np.sum(tensor > 0))]
        return _counts_summary(counts, total_count=np.size(tensor))
    else:
        return debugger_cli_common.RichTextLines(['No numeric summary available due to tensor dtype: %s.' % tensor.dtype])