import functools
from typing import List, Mapping, Optional
import dpath.exceptions
import dpath.util
import icdiff
import py
from pprintpp import pformat
MAX_COLS = py.io.TerminalWriter().fullwidth
MARGIN_LEFT = 20
GUTTER = 3
MARGINS = MARGIN_LEFT + GUTTER + 1

def diff_dicts(left, right, use_markup) -> Optional[List[str]]:
    if False:
        while True:
            i = 10
    half_cols = MAX_COLS / 2 - MARGINS
    pretty_left = pformat(left, indent=1, width=half_cols).splitlines()
    pretty_right = pformat(right, indent=1, width=half_cols).splitlines()
    diff_cols = MAX_COLS - MARGINS
    if len(pretty_left) < 3 or len(pretty_right) < 3:
        smallest_left = pformat(left, indent=2, width=1).splitlines()
        smallest_right = pformat(right, indent=2, width=1).splitlines()
        max_side = max((len(line) + 1 for line in smallest_left + smallest_right))
        if max_side * 2 + MARGIN_LEFT < MAX_COLS:
            diff_cols = max_side * 2 + GUTTER
            pretty_left = pformat(left, indent=2, width=max_side).splitlines()
            pretty_right = pformat(right, indent=2, width=max_side).splitlines()
    differ = icdiff.ConsoleDiff(cols=diff_cols, tabsize=2)
    if not use_markup:
        differ.colorize = lambda string: string
        color_off = ''
    else:
        color_off = icdiff.color_codes['none']
    icdiff_lines = list(differ.make_table(pretty_left, pretty_right, context=True))
    return ['equals failed'] + [color_off + line for line in icdiff_lines]

@functools.total_ordering
class HashMixin:

    @staticmethod
    def get_hash(obj):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, Mapping):
            return hash(str({k: HashMixin.get_hash(v) for (k, v) in sorted(obj.items())}))
        if isinstance(obj, List):
            return hash(str(sorted([HashMixin.get_hash(v) for v in obj])))
        return hash(obj)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return HashMixin.get_hash(self)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return hash(self) < hash(other)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return hash(self) == hash(other)

class DictWithHashMixin(HashMixin, dict):
    pass

class ListWithHashMixin(HashMixin, list):
    pass

def delete_fields(obj: Mapping, path_list: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    for path in path_list:
        try:
            dpath.util.delete(obj, path)
        except dpath.exceptions.PathNotFound:
            pass

def make_hashable(obj, exclude_fields: List[str]=None) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Simplify comparison of nested dicts/lists\n    :param obj value for comparison\n    :param exclude_fields if value is Mapping, some fields can be excluded\n    '
    if isinstance(obj, Mapping):
        if exclude_fields:
            delete_fields(obj, exclude_fields)
        return DictWithHashMixin(obj)
    if isinstance(obj, List):
        return ListWithHashMixin(obj)
    return obj