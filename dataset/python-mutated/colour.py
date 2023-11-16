"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations
import colorsys
import random
import re
from typing import TYPE_CHECKING, Optional, Tuple, Union
if TYPE_CHECKING:
    from typing_extensions import Self
__all__ = ('Colour', 'Color')
RGB_REGEX = re.compile('rgb\\s*\\((?P<r>[0-9.]+%?)\\s*,\\s*(?P<g>[0-9.]+%?)\\s*,\\s*(?P<b>[0-9.]+%?)\\s*\\)')

def parse_hex_number(argument: str) -> Colour:
    if False:
        return 10
    arg = ''.join((i * 2 for i in argument)) if len(argument) == 3 else argument
    try:
        value = int(arg, base=16)
        if not 0 <= value <= 16777215:
            raise ValueError('hex number out of range for 24-bit colour')
    except ValueError:
        raise ValueError('invalid hex digit given') from None
    else:
        return Color(value=value)

def parse_rgb_number(number: str) -> int:
    if False:
        return 10
    if number[-1] == '%':
        value = float(number[:-1])
        if not 0 <= value <= 100:
            raise ValueError('rgb percentage can only be between 0 to 100')
        return round(255 * (value / 100))
    value = int(number)
    if not 0 <= value <= 255:
        raise ValueError('rgb number can only be between 0 to 255')
    return value

def parse_rgb(argument: str, *, regex: re.Pattern[str]=RGB_REGEX) -> Colour:
    if False:
        i = 10
        return i + 15
    match = regex.match(argument)
    if match is None:
        raise ValueError('invalid rgb syntax found')
    red = parse_rgb_number(match.group('r'))
    green = parse_rgb_number(match.group('g'))
    blue = parse_rgb_number(match.group('b'))
    return Color.from_rgb(red, green, blue)

class Colour:
    """Represents a Discord role colour. This class is similar
    to a (red, green, blue) :class:`tuple`.

    There is an alias for this called Color.

    .. container:: operations

        .. describe:: x == y

             Checks if two colours are equal.

        .. describe:: x != y

             Checks if two colours are not equal.

        .. describe:: hash(x)

             Return the colour's hash.

        .. describe:: str(x)

             Returns the hex format for the colour.

        .. describe:: int(x)

             Returns the raw colour value.

    .. note::

        The colour values in the classmethods are mostly provided as-is and can change between
        versions should the Discord client's representation of that colour also change.

    Attributes
    ------------
    value: :class:`int`
        The raw integer colour value.
    """
    __slots__ = ('value',)

    def __init__(self, value: int):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, int):
            raise TypeError(f'Expected int parameter, received {value.__class__.__name__} instead.')
        self.value: int = value

    def _get_byte(self, byte: int) -> int:
        if False:
            while True:
                i = 10
        return self.value >> 8 * byte & 255

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, Colour) and self.value == other.value

    def __ne__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return f'#{self.value:0>6x}'

    def __int__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.value

    def __repr__(self) -> str:
        if False:
            return 10
        return f'<Colour value={self.value}>'

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(self.value)

    @property
    def r(self) -> int:
        if False:
            return 10
        ':class:`int`: Returns the red component of the colour.'
        return self._get_byte(2)

    @property
    def g(self) -> int:
        if False:
            return 10
        ':class:`int`: Returns the green component of the colour.'
        return self._get_byte(1)

    @property
    def b(self) -> int:
        if False:
            while True:
                i = 10
        ':class:`int`: Returns the blue component of the colour.'
        return self._get_byte(0)

    def to_rgb(self) -> Tuple[int, int, int]:
        if False:
            return 10
        'Tuple[:class:`int`, :class:`int`, :class:`int`]: Returns an (r, g, b) tuple representing the colour.'
        return (self.r, self.g, self.b)

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> Self:
        if False:
            return 10
        'Constructs a :class:`Colour` from an RGB tuple.'
        return cls((r << 16) + (g << 8) + b)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float) -> Self:
        if False:
            i = 10
            return i + 15
        'Constructs a :class:`Colour` from an HSV tuple.'
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return cls.from_rgb(*(int(x * 255) for x in rgb))

    @classmethod
    def from_str(cls, value: str) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a :class:`Colour` from a string.\n\n        The following formats are accepted:\n\n        - ``0x<hex>``\n        - ``#<hex>``\n        - ``0x#<hex>``\n        - ``rgb(<number>, <number>, <number>)``\n\n        Like CSS, ``<number>`` can be either 0-255 or 0-100% and ``<hex>`` can be\n        either a 6 digit hex number or a 3 digit hex shortcut (e.g. #FFF).\n\n        .. versionadded:: 2.0\n\n        Raises\n        -------\n        ValueError\n            The string could not be converted into a colour.\n        '
        if not value:
            raise ValueError('unknown colour format given')
        if value[0] == '#':
            return parse_hex_number(value[1:])
        if value[0:2] == '0x':
            rest = value[2:]
            if rest.startswith('#'):
                return parse_hex_number(rest[1:])
            return parse_hex_number(rest)
        arg = value.lower()
        if arg[0:3] == 'rgb':
            return parse_rgb(arg)
        raise ValueError('unknown colour format given')

    @classmethod
    def default(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0``.\n\n        .. colour:: #000000\n        '
        return cls(0)

    @classmethod
    def random(cls, *, seed: Optional[Union[int, str, float, bytes, bytearray]]=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a random hue.\n\n        .. note::\n\n            The random algorithm works by choosing a colour with a random hue but\n            with maxed out saturation and value.\n\n        .. versionadded:: 1.6\n\n        Parameters\n        ------------\n        seed: Optional[Union[:class:`int`, :class:`str`, :class:`float`, :class:`bytes`, :class:`bytearray`]]\n            The seed to initialize the RNG with. If ``None`` is passed the default RNG is used.\n\n            .. versionadded:: 1.7\n        '
        rand = random if seed is None else random.Random(seed)
        return cls.from_hsv(rand.random(), 1, 1)

    @classmethod
    def teal(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x1ABC9C``.\n\n        .. colour:: #1ABC9C\n        '
        return cls(1752220)

    @classmethod
    def dark_teal(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x11806A``.\n\n        .. colour:: #11806A\n        '
        return cls(1146986)

    @classmethod
    def brand_green(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x57F287``.\n\n        .. colour:: #57F287\n\n\n        .. versionadded:: 2.0\n        '
        return cls(5763719)

    @classmethod
    def green(cls) -> Self:
        if False:
            while True:
                i = 10
        'A factory method that returns a :class:`Colour` with a value of ``0x2ECC71``.\n\n        .. colour:: #2ECC71\n        '
        return cls(3066993)

    @classmethod
    def dark_green(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x1F8B4C``.\n\n        .. colour:: #1F8B4C\n        '
        return cls(2067276)

    @classmethod
    def blue(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0x3498DB``.\n\n        .. colour:: #3498DB\n        '
        return cls(3447003)

    @classmethod
    def dark_blue(cls) -> Self:
        if False:
            return 10
        'A factory method that returns a :class:`Colour` with a value of ``0x206694``.\n\n        .. colour:: #206694\n        '
        return cls(2123412)

    @classmethod
    def purple(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x9B59B6``.\n\n        .. colour:: #9B59B6\n        '
        return cls(10181046)

    @classmethod
    def dark_purple(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x71368A``.\n\n        .. colour:: #71368A\n        '
        return cls(7419530)

    @classmethod
    def magenta(cls) -> Self:
        if False:
            while True:
                i = 10
        'A factory method that returns a :class:`Colour` with a value of ``0xE91E63``.\n\n        .. colour:: #E91E63\n        '
        return cls(15277667)

    @classmethod
    def dark_magenta(cls) -> Self:
        if False:
            print('Hello World!')
        'A factory method that returns a :class:`Colour` with a value of ``0xAD1457``.\n\n        .. colour:: #AD1457\n        '
        return cls(11342935)

    @classmethod
    def gold(cls) -> Self:
        if False:
            return 10
        'A factory method that returns a :class:`Colour` with a value of ``0xF1C40F``.\n\n        .. colour:: #F1C40F\n        '
        return cls(15844367)

    @classmethod
    def dark_gold(cls) -> Self:
        if False:
            print('Hello World!')
        'A factory method that returns a :class:`Colour` with a value of ``0xC27C0E``.\n\n        .. colour:: #C27C0E\n        '
        return cls(12745742)

    @classmethod
    def orange(cls) -> Self:
        if False:
            while True:
                i = 10
        'A factory method that returns a :class:`Colour` with a value of ``0xE67E22``.\n\n        .. colour:: #E67E22\n        '
        return cls(15105570)

    @classmethod
    def dark_orange(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0xA84300``.\n\n        .. colour:: #A84300\n        '
        return cls(11027200)

    @classmethod
    def brand_red(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0xED4245``.\n\n        .. colour:: #ED4245\n\n        .. versionadded:: 2.0\n        '
        return cls(15548997)

    @classmethod
    def red(cls) -> Self:
        if False:
            while True:
                i = 10
        'A factory method that returns a :class:`Colour` with a value of ``0xE74C3C``.\n\n        .. colour:: #E74C3C\n        '
        return cls(15158332)

    @classmethod
    def dark_red(cls) -> Self:
        if False:
            while True:
                i = 10
        'A factory method that returns a :class:`Colour` with a value of ``0x992D22``.\n\n        .. colour:: #992D22\n        '
        return cls(10038562)

    @classmethod
    def lighter_grey(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x95A5A6``.\n\n        .. colour:: #95A5A6\n        '
        return cls(9807270)
    lighter_gray = lighter_grey

    @classmethod
    def dark_grey(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0x607d8b``.\n\n        .. colour:: #607d8b\n        '
        return cls(6323595)
    dark_gray = dark_grey

    @classmethod
    def light_grey(cls) -> Self:
        if False:
            print('Hello World!')
        'A factory method that returns a :class:`Colour` with a value of ``0x979C9F``.\n\n        .. colour:: #979C9F\n        '
        return cls(9936031)
    light_gray = light_grey

    @classmethod
    def darker_grey(cls) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'A factory method that returns a :class:`Colour` with a value of ``0x546E7A``.\n\n        .. colour:: #546E7A\n        '
        return cls(5533306)
    darker_gray = darker_grey

    @classmethod
    def og_blurple(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0x7289DA``.\n\n        .. colour:: #7289DA\n        '
        return cls(7506394)

    @classmethod
    def blurple(cls) -> Self:
        if False:
            while True:
                i = 10
        'A factory method that returns a :class:`Colour` with a value of ``0x5865F2``.\n\n        .. colour:: #5865F2\n        '
        return cls(5793266)

    @classmethod
    def greyple(cls) -> Self:
        if False:
            print('Hello World!')
        'A factory method that returns a :class:`Colour` with a value of ``0x99AAB5``.\n\n        .. colour:: #99AAB5\n        '
        return cls(10070709)

    @classmethod
    def dark_theme(cls) -> Self:
        if False:
            return 10
        "A factory method that returns a :class:`Colour` with a value of ``0x313338``.\n\n        This will appear transparent on Discord's dark theme.\n\n        .. colour:: #313338\n\n        .. versionadded:: 1.5\n\n        .. versionchanged:: 2.2\n            Updated colour from previous ``0x36393F`` to reflect discord theme changes.\n        "
        return cls(3224376)

    @classmethod
    def fuchsia(cls) -> Self:
        if False:
            return 10
        'A factory method that returns a :class:`Colour` with a value of ``0xEB459E``.\n\n        .. colour:: #EB459E\n\n        .. versionadded:: 2.0\n        '
        return cls(15418782)

    @classmethod
    def yellow(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0xFEE75C``.\n\n        .. colour:: #FEE75C\n\n        .. versionadded:: 2.0\n        '
        return cls(16705372)

    @classmethod
    def dark_embed(cls) -> Self:
        if False:
            return 10
        'A factory method that returns a :class:`Colour` with a value of ``0x2B2D31``.\n\n        .. colour:: #2B2D31\n\n        .. versionadded:: 2.2\n        '
        return cls(2829617)

    @classmethod
    def light_embed(cls) -> Self:
        if False:
            i = 10
            return i + 15
        'A factory method that returns a :class:`Colour` with a value of ``0xEEEFF1``.\n\n        .. colour:: #EEEFF1\n\n        .. versionadded:: 2.2\n        '
        return cls(15658993)

    @classmethod
    def pink(cls) -> Self:
        if False:
            print('Hello World!')
        'A factory method that returns a :class:`Colour` with a value of ``0xEB459F``.\n\n        .. colour:: #EB459F\n\n        .. versionadded:: 2.3\n        '
        return cls(15418783)
Color = Colour