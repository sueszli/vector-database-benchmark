from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
FloatRGBColorTuple: TypeAlias = Tuple[float, float, float]
FloatRGBAColorTuple: TypeAlias = Tuple[float, float, float, float]
IntRGBColorTuple: TypeAlias = Tuple[int, int, int]
IntRGBAColorTuple: TypeAlias = Tuple[int, int, int, int]
MixedRGBAColorTuple: TypeAlias = Tuple[int, int, int, float]
Color4Tuple: TypeAlias = Union[FloatRGBAColorTuple, IntRGBAColorTuple, MixedRGBAColorTuple]
Color3Tuple: TypeAlias = Union[FloatRGBColorTuple, IntRGBColorTuple]
ColorTuple: TypeAlias = Union[Color4Tuple, Color3Tuple]
IntColorTuple = Union[IntRGBColorTuple, IntRGBAColorTuple]
CSSColorStr = Union[IntRGBAColorTuple, MixedRGBAColorTuple]
ColorStr: TypeAlias = str
Color: TypeAlias = Union[ColorTuple, ColorStr]
MaybeColor: TypeAlias = Union[str, Collection[Any]]

def to_int_color_tuple(color: MaybeColor) -> IntColorTuple:
    if False:
        return 10
    'Convert input into color tuple of type (int, int, int, int).'
    color_tuple = _to_color_tuple(color, rgb_formatter=_int_formatter, alpha_formatter=_int_formatter)
    return cast(IntColorTuple, color_tuple)

def to_css_color(color: MaybeColor) -> Color:
    if False:
        return 10
    'Convert input into a CSS-compatible color that Vega can use.\n\n    Inputs must be a hex string, rgb()/rgba() string, or a color tuple. Inputs may not be a CSS\n    color name, other CSS color function (like "hsl(...)"), etc.\n\n    See tests for more info.\n    '
    if is_css_color_like(color):
        return cast(Color, color)
    if is_color_tuple_like(color):
        ctuple = cast(ColorTuple, color)
        ctuple = _normalize_tuple(ctuple, _int_formatter, _float_formatter)
        if len(ctuple) == 3:
            return f'rgb({ctuple[0]}, {ctuple[1]}, {ctuple[2]})'
        elif len(ctuple) == 4:
            c4tuple = cast(MixedRGBAColorTuple, ctuple)
            return f'rgba({c4tuple[0]}, {c4tuple[1]}, {c4tuple[2]}, {c4tuple[3]})'
    raise InvalidColorException(color)

def is_css_color_like(color: MaybeColor) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check whether the input looks like something Vega can use.\n\n    This is meant to be lightweight, and not a definitive answer. The definitive solution is to try\n    to convert and see if an error is thrown.\n\n    NOTE: We only accept hex colors and color tuples as user input. So do not use this function to\n    validate user input! Instead use is_hex_color_like and is_color_tuple_like.\n    '
    return is_hex_color_like(color) or _is_cssrgb_color_like(color)

def is_hex_color_like(color: MaybeColor) -> bool:
    if False:
        return 10
    'Check whether the input looks like a hex color.\n\n    This is meant to be lightweight, and not a definitive answer. The definitive solution is to try\n    to convert and see if an error is thrown.\n    '
    return isinstance(color, str) and color.startswith('#') and color[1:].isalnum() and (len(color) in {4, 5, 7, 9})

def _is_cssrgb_color_like(color: MaybeColor) -> bool:
    if False:
        while True:
            i = 10
    'Check whether the input looks like a CSS rgb() or rgba() color string.\n\n    This is meant to be lightweight, and not a definitive answer. The definitive solution is to try\n    to convert and see if an error is thrown.\n\n    NOTE: We only accept hex colors and color tuples as user input. So do not use this function to\n    validate user input! Instead use is_hex_color_like and is_color_tuple_like.\n    '
    return isinstance(color, str) and (color.startswith('rgb(') or color.startswith('rgba('))

def is_color_tuple_like(color: MaybeColor) -> bool:
    if False:
        i = 10
        return i + 15
    'Check whether the input looks like a tuple color.\n\n    This is meant to be lightweight, and not a definitive answer. The definitive solution is to try\n    to convert and see if an error is thrown.\n    '
    return isinstance(color, (tuple, list)) and len(color) in {3, 4} and all((isinstance(c, (int, float)) for c in color))

def is_color_like(color: MaybeColor) -> bool:
    if False:
        return 10
    "A fairly lightweight check of whether the input is a color.\n\n    This isn't meant to be a definitive answer. The definitive solution is to\n    try to convert and see if an error is thrown.\n    "
    return is_css_color_like(color) or is_color_tuple_like(color)

def _to_color_tuple(color: MaybeColor, rgb_formatter: Callable[[float, MaybeColor], float], alpha_formatter: Callable[[float, MaybeColor], float]):
    if False:
        while True:
            i = 10
    'Convert a potential color to a color tuple.\n\n    The exact type of color tuple this outputs is dictated by the formatter parameters.\n\n    The R, G, B components are transformed by rgb_formatter, and the alpha component is transformed\n    by alpha_formatter.\n\n    For example, to output a (float, float, float, int) color tuple, set rgb_formatter\n    to _float_formatter and alpha_formatter to _int_formatter.\n    '
    if is_hex_color_like(color):
        hex_len = len(color)
        color_hex = cast(str, color)
        if hex_len == 4:
            r = 2 * color_hex[1]
            g = 2 * color_hex[2]
            b = 2 * color_hex[3]
            a = 'ff'
        elif hex_len == 5:
            r = 2 * color_hex[1]
            g = 2 * color_hex[2]
            b = 2 * color_hex[3]
            a = 2 * color_hex[4]
        elif hex_len == 7:
            r = color_hex[1:3]
            g = color_hex[3:5]
            b = color_hex[5:7]
            a = 'ff'
        elif hex_len == 9:
            r = color_hex[1:3]
            g = color_hex[3:5]
            b = color_hex[5:7]
            a = color_hex[7:9]
        else:
            raise InvalidColorException(color)
        try:
            color = (int(r, 16), int(g, 16), int(b, 16), int(a, 16))
        except:
            raise InvalidColorException(color)
    if is_color_tuple_like(color):
        color_tuple = cast(ColorTuple, color)
        return _normalize_tuple(color_tuple, rgb_formatter, alpha_formatter)
    raise InvalidColorException(color)

def _normalize_tuple(color: ColorTuple, rgb_formatter: Callable[[float, MaybeColor], float], alpha_formatter: Callable[[float, MaybeColor], float]) -> ColorTuple:
    if False:
        while True:
            i = 10
    'Parse color tuple using the specified color formatters.\n\n    The R, G, B components are transformed by rgb_formatter, and the alpha component is transformed\n    by alpha_formatter.\n\n    For example, to output a (float, float, float, int) color tuple, set rgb_formatter\n    to _float_formatter and alpha_formatter to _int_formatter.\n    '
    if len(color) == 3:
        r = rgb_formatter(color[0], color)
        g = rgb_formatter(color[1], color)
        b = rgb_formatter(color[2], color)
        return (r, g, b)
    elif len(color) == 4:
        color_4tuple = cast(Color4Tuple, color)
        r = rgb_formatter(color_4tuple[0], color_4tuple)
        g = rgb_formatter(color_4tuple[1], color_4tuple)
        b = rgb_formatter(color_4tuple[2], color_4tuple)
        alpha = alpha_formatter(color_4tuple[3], color_4tuple)
        return (r, g, b, alpha)
    raise InvalidColorException(color)

def _int_formatter(component: float, color: MaybeColor) -> int:
    if False:
        i = 10
        return i + 15
    'Convert a color component (float or int) to an int from 0 to 255.\n\n    Anything too small will become 0, and anything too large will become 255.\n    '
    if isinstance(component, float):
        component = int(component * 255)
    if isinstance(component, int):
        return min(255, max(component, 0))
    raise InvalidColorException(color)

def _float_formatter(component: float, color: MaybeColor) -> float:
    if False:
        print('Hello World!')
    'Convert a color component (float or int) to a float from 0.0 to 1.0.\n\n    Anything too small will become 0.0, and anything too large will become 1.0.\n    '
    if isinstance(component, int):
        component = component / 255.0
    if isinstance(component, float):
        return min(1.0, max(component, 0.0))
    raise InvalidColorException(color)

class InvalidColorException(StreamlitAPIException):

    def __init__(self, color, *args):
        if False:
            print('Hello World!')
        message = f"This does not look like a valid color: {repr(color)}.\n\nColors must be in one of the following formats:\n\n* Hex string with 3, 4, 6, or 8 digits. Example: `'#00ff00'`\n* List or tuple with 3 or 4 components. Example: `[1.0, 0.5, 0, 0.2]`\n            "
        super().__init__(message, *args)