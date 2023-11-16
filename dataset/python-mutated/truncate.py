"""Utilities for truncating assertion output.

Current default behaviour is to truncate assertion explanations at
~8 terminal lines, unless running in "-vv" mode or running on CI.
"""
from typing import List
from typing import Optional
from _pytest.assertion import util
from _pytest.nodes import Item
DEFAULT_MAX_LINES = 8
DEFAULT_MAX_CHARS = 8 * 80
USAGE_MSG = "use '-vv' to show"

def truncate_if_required(explanation: List[str], item: Item, max_length: Optional[int]=None) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Truncate this assertion explanation if the given test item is eligible.'
    if _should_truncate_item(item):
        return _truncate_explanation(explanation)
    return explanation

def _should_truncate_item(item: Item) -> bool:
    if False:
        return 10
    'Whether or not this test item is eligible for truncation.'
    verbose = item.config.option.verbose
    return verbose < 2 and (not util.running_on_ci())

def _truncate_explanation(input_lines: List[str], max_lines: Optional[int]=None, max_chars: Optional[int]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Truncate given list of strings that makes up the assertion explanation.\n\n    Truncates to either 8 lines, or 640 characters - whichever the input reaches\n    first, taking the truncation explanation into account. The remaining lines\n    will be replaced by a usage message.\n    '
    if max_lines is None:
        max_lines = DEFAULT_MAX_LINES
    if max_chars is None:
        max_chars = DEFAULT_MAX_CHARS
    input_char_count = len(''.join(input_lines))
    tolerable_max_chars = max_chars + 70
    tolerable_max_lines = max_lines + 2
    if len(input_lines) <= tolerable_max_lines and input_char_count <= tolerable_max_chars:
        return input_lines
    truncated_explanation = input_lines[:max_lines]
    truncated_char = True
    if len(''.join(truncated_explanation)) > tolerable_max_chars:
        truncated_explanation = _truncate_by_char_count(truncated_explanation, max_chars)
    else:
        truncated_char = False
    truncated_line_count = len(input_lines) - len(truncated_explanation)
    if truncated_explanation[-1]:
        truncated_explanation[-1] = truncated_explanation[-1] + '...'
        if truncated_char:
            truncated_line_count += 1
    else:
        truncated_explanation[-1] = '...'
    return truncated_explanation + ['', f"...Full output truncated ({truncated_line_count} line{('' if truncated_line_count == 1 else 's')} hidden), {USAGE_MSG}"]

def _truncate_by_char_count(input_lines: List[str], max_chars: int) -> List[str]:
    if False:
        while True:
            i = 10
    iterated_char_count = 0
    for (iterated_index, input_line) in enumerate(input_lines):
        if iterated_char_count + len(input_line) > max_chars:
            break
        iterated_char_count += len(input_line)
    truncated_result = input_lines[:iterated_index]
    final_line = input_lines[iterated_index]
    if final_line:
        final_line_truncate_point = max_chars - iterated_char_count
        final_line = final_line[:final_line_truncate_point]
    truncated_result.append(final_line)
    return truncated_result