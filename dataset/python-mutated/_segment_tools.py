"""
Tools for processing Segments, or lists of Segments.
"""
from __future__ import annotations
from typing import Iterable
from rich.segment import Segment
from rich.style import Style
from ._cells import cell_len
from .css.types import AlignHorizontal, AlignVertical
from .geometry import Size

class NoCellPositionForIndex(Exception):
    pass

def index_to_cell_position(segments: Iterable[Segment], index: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    "Given a character index, return the cell position of that character within\n    an Iterable of Segments. This is the sum of the cell lengths of all the characters\n    *before* the character at `index`.\n\n    Args:\n        segments: The segments to find the cell position within.\n        index: The index to convert into a cell position.\n\n    Returns:\n        The cell position of the character at `index`.\n\n    Raises:\n        NoCellPositionForIndex: If the supplied index doesn't fall within the given segments.\n    "
    if not segments:
        raise NoCellPositionForIndex
    if index == 0:
        return 0
    cell_position_end = 0
    segment_length = 0
    segment_end_index = 0
    segment_cell_length = 0
    text = ''
    iter_segments = iter(segments)
    try:
        while segment_end_index < index:
            segment = next(iter_segments)
            text = segment.text
            segment_length = len(text)
            segment_cell_length = cell_len(text)
            cell_position_end += segment_cell_length
            segment_end_index += segment_length
    except StopIteration:
        raise NoCellPositionForIndex
    segment_index_start = segment_end_index - segment_length
    index_within_segment = index - segment_index_start
    segment_cell_start = cell_position_end - segment_cell_length
    return segment_cell_start + cell_len(text[:index_within_segment])

def line_crop(segments: list[Segment], start: int, end: int, total: int) -> list[Segment]:
    if False:
        for i in range(10):
            print('nop')
    'Crops a list of segments between two cell offsets.\n\n    Args:\n        segments: A list of Segments for a line.\n        start: Start offset (cells)\n        end: End offset (cells, exclusive)\n        total: Total cell length of segments.\n    Returns:\n        A new shorter list of segments\n    '
    _cell_len = cell_len
    pos = 0
    output_segments: list[Segment] = []
    add_segment = output_segments.append
    iter_segments = iter(segments)
    segment: Segment | None = None
    for segment in iter_segments:
        end_pos = pos + _cell_len(segment.text)
        if end_pos > start:
            segment = segment.split_cells(start - pos)[1]
            break
        pos = end_pos
    else:
        return []
    if end >= total:
        if segment:
            add_segment(segment)
        output_segments.extend(iter_segments)
        return output_segments
    pos = start
    while segment is not None:
        end_pos = pos + _cell_len(segment.text)
        if end_pos < end:
            add_segment(segment)
        else:
            add_segment(segment.split_cells(end - pos)[0])
            break
        pos = end_pos
        segment = next(iter_segments, None)
    return output_segments

def line_trim(segments: list[Segment], start: bool, end: bool) -> list[Segment]:
    if False:
        while True:
            i = 10
    'Optionally remove a cell from the start and / or end of a list of segments.\n\n    Args:\n        segments: A line (list of Segments)\n        start: Remove cell from start.\n        end: Remove cell from end.\n\n    Returns:\n        A new list of segments.\n    '
    segments = segments.copy()
    if segments and start:
        (_, first_segment) = segments[0].split_cells(1)
        if first_segment.text:
            segments[0] = first_segment
        else:
            segments.pop(0)
    if segments and end:
        last_segment = segments[-1]
        (last_segment, _) = last_segment.split_cells(len(last_segment.text) - 1)
        if last_segment.text:
            segments[-1] = last_segment
        else:
            segments.pop()
    return segments

def line_pad(segments: Iterable[Segment], pad_left: int, pad_right: int, style: Style) -> list[Segment]:
    if False:
        i = 10
        return i + 15
    'Adds padding to the left and / or right of a list of segments.\n\n    Args:\n        segments: A line of segments.\n        pad_left: Cells to pad on the left.\n        pad_right: Cells to pad on the right.\n        style: Style of padded cells.\n\n    Returns:\n        A new line with padding.\n    '
    if pad_left and pad_right:
        return [Segment(' ' * pad_left, style), *segments, Segment(' ' * pad_right, style)]
    elif pad_left:
        return [Segment(' ' * pad_left, style), *segments]
    elif pad_right:
        return [*segments, Segment(' ' * pad_right, style)]
    return list(segments)

def align_lines(lines: list[list[Segment]], style: Style, size: Size, horizontal: AlignHorizontal, vertical: AlignVertical) -> Iterable[list[Segment]]:
    if False:
        print('Hello World!')
    'Align lines.\n\n    Args:\n        lines: A list of lines.\n        style: Background style.\n        size: Size of container.\n        horizontal: Horizontal alignment.\n        vertical: Vertical alignment.\n\n    Returns:\n        Aligned lines.\n    '
    (width, height) = size
    (shape_width, shape_height) = Segment.get_shape(lines)

    def blank_lines(count: int) -> list[list[Segment]]:
        if False:
            i = 10
            return i + 15
        return [[Segment(' ' * width, style)]] * count
    top_blank_lines = bottom_blank_lines = 0
    vertical_excess_space = max(0, height - shape_height)
    if vertical == 'top':
        bottom_blank_lines = vertical_excess_space
    elif vertical == 'middle':
        top_blank_lines = vertical_excess_space // 2
        bottom_blank_lines = vertical_excess_space - top_blank_lines
    elif vertical == 'bottom':
        top_blank_lines = vertical_excess_space
    yield from blank_lines(top_blank_lines)
    horizontal_excess_space = max(0, width - shape_width)
    adjust_line_length = Segment.adjust_line_length
    if horizontal == 'left':
        for line in lines:
            yield adjust_line_length(line, width, style, pad=True)
    elif horizontal == 'center':
        left_space = horizontal_excess_space // 2
        for line in lines:
            left_padding = [Segment(' ' * left_space, style)] if left_space else []
            yield [*left_padding, *adjust_line_length(line, width - left_space, style, pad=True)]
    elif horizontal == 'right':
        get_line_length = Segment.get_line_length
        for line in lines:
            left_space = width - get_line_length(line)
            left_padding = [Segment(' ' * left_space, style)] if left_space else []
            yield [*left_padding, *line]
    yield from blank_lines(bottom_blank_lines)