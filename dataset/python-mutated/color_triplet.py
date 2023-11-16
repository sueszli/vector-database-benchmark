from typing import NamedTuple, Tuple

class ColorTriplet(NamedTuple):
    """The red, green, and blue components of a color."""
    red: int
    'Red component in 0 to 255 range.'
    green: int
    'Green component in 0 to 255 range.'
    blue: int
    'Blue component in 0 to 255 range.'

    @property
    def hex(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'get the color triplet in CSS style.'
        (red, green, blue) = self
        return f'#{red:02x}{green:02x}{blue:02x}'

    @property
    def rgb(self) -> str:
        if False:
            return 10
        'The color in RGB format.\n\n        Returns:\n            str: An rgb color, e.g. ``"rgb(100,23,255)"``.\n        '
        (red, green, blue) = self
        return f'rgb({red},{green},{blue})'

    @property
    def normalized(self) -> Tuple[float, float, float]:
        if False:
            while True:
                i = 10
        'Convert components into floats between 0 and 1.\n\n        Returns:\n            Tuple[float, float, float]: A tuple of three normalized colour components.\n        '
        (red, green, blue) = self
        return (red / 255.0, green / 255.0, blue / 255.0)