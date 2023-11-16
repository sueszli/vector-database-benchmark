import re
from ctypes import POINTER, c_uint, c_void_p, cast
from typing import Callable, Generator, Iterable, Pattern, Sequence, Tuple, Union
from .utils import resolve_custom_file
pointer_to_uint = POINTER(c_uint)
MarkerFunc = Callable[[str, int, int, int], Generator[None, None, None]]

def get_output_variables(left_address: int, right_address: int, color_address: int) -> Tuple[c_uint, c_uint, c_uint]:
    if False:
        print('Hello World!')
    return (cast(c_void_p(left_address), pointer_to_uint).contents, cast(c_void_p(right_address), pointer_to_uint).contents, cast(c_void_p(color_address), pointer_to_uint).contents)

def marker_from_regex(expression: Union[str, 'Pattern[str]'], color: int, flags: int=re.UNICODE) -> MarkerFunc:
    if False:
        print('Hello World!')
    color = max(1, min(color, 3))
    if isinstance(expression, str):
        pat = re.compile(expression, flags=flags)
    else:
        pat = expression

    def marker(text: str, left_address: int, right_address: int, color_address: int) -> Generator[None, None, None]:
        if False:
            while True:
                i = 10
        (left, right, colorv) = get_output_variables(left_address, right_address, color_address)
        colorv.value = color
        for match in pat.finditer(text):
            left.value = match.start()
            right.value = match.end() - 1
            yield
    return marker

def marker_from_multiple_regex(regexes: Iterable[Tuple[int, str]], flags: int=re.UNICODE) -> MarkerFunc:
    if False:
        print('Hello World!')
    expr = ''
    color_map = {}
    for (i, (color, spec)) in enumerate(regexes):
        grp = f'mcg{i}'
        expr += f'|(?P<{grp}>{spec})'
        color_map[grp] = color
    expr = expr[1:]
    pat = re.compile(expr, flags=flags)

    def marker(text: str, left_address: int, right_address: int, color_address: int) -> Generator[None, None, None]:
        if False:
            for i in range(10):
                print('nop')
        (left, right, color) = get_output_variables(left_address, right_address, color_address)
        for match in pat.finditer(text):
            left.value = match.start()
            right.value = match.end() - 1
            grp = match.lastgroup
            color.value = color_map[grp] if grp is not None else 0
            yield
    return marker

def marker_from_text(expression: str, color: int) -> MarkerFunc:
    if False:
        i = 10
        return i + 15
    return marker_from_regex(re.escape(expression), color)

def marker_from_function(func: Callable[[str], Iterable[Tuple[int, int, int]]]) -> MarkerFunc:
    if False:
        while True:
            i = 10

    def marker(text: str, left_address: int, right_address: int, color_address: int) -> Generator[None, None, None]:
        if False:
            print('Hello World!')
        (left, right, colorv) = get_output_variables(left_address, right_address, color_address)
        for (ll, r, c) in func(text):
            left.value = ll
            right.value = r
            colorv.value = c
            yield
    return marker

def marker_from_spec(ftype: str, spec: Union[str, Sequence[Tuple[int, str]]], flags: int) -> MarkerFunc:
    if False:
        while True:
            i = 10
    if ftype == 'regex':
        assert not isinstance(spec, str)
        if len(spec) == 1:
            return marker_from_regex(spec[0][1], spec[0][0], flags=flags)
        return marker_from_multiple_regex(spec, flags=flags)
    if ftype == 'function':
        import runpy
        assert isinstance(spec, str)
        path = resolve_custom_file(spec)
        return marker_from_function(runpy.run_path(path, run_name='__marker__')['marker'])
    raise ValueError(f'Unknown marker type: {ftype}')