from operator import itemgetter
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence
from . import errors
from .protocol import is_renderable, rich_cast
if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderableType

class Measurement(NamedTuple):
    """Stores the minimum and maximum widths (in characters) required to render an object."""
    minimum: int
    'Minimum number of cells required to render.'
    maximum: int
    'Maximum number of cells required to render.'

    @property
    def span(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Get difference between maximum and minimum.'
        return self.maximum - self.minimum

    def normalize(self) -> 'Measurement':
        if False:
            while True:
                i = 10
        'Get measurement that ensures that minimum <= maximum and minimum >= 0\n\n        Returns:\n            Measurement: A normalized measurement.\n        '
        (minimum, maximum) = self
        minimum = min(max(0, minimum), maximum)
        return Measurement(max(0, minimum), max(0, max(minimum, maximum)))

    def with_maximum(self, width: int) -> 'Measurement':
        if False:
            print('Hello World!')
        'Get a RenderableWith where the widths are <= width.\n\n        Args:\n            width (int): Maximum desired width.\n\n        Returns:\n            Measurement: New Measurement object.\n        '
        (minimum, maximum) = self
        return Measurement(min(minimum, width), min(maximum, width))

    def with_minimum(self, width: int) -> 'Measurement':
        if False:
            for i in range(10):
                print('nop')
        'Get a RenderableWith where the widths are >= width.\n\n        Args:\n            width (int): Minimum desired width.\n\n        Returns:\n            Measurement: New Measurement object.\n        '
        (minimum, maximum) = self
        width = max(0, width)
        return Measurement(max(minimum, width), max(maximum, width))

    def clamp(self, min_width: Optional[int]=None, max_width: Optional[int]=None) -> 'Measurement':
        if False:
            print('Hello World!')
        'Clamp a measurement within the specified range.\n\n        Args:\n            min_width (int): Minimum desired width, or ``None`` for no minimum. Defaults to None.\n            max_width (int): Maximum desired width, or ``None`` for no maximum. Defaults to None.\n\n        Returns:\n            Measurement: New Measurement object.\n        '
        measurement = self
        if min_width is not None:
            measurement = measurement.with_minimum(min_width)
        if max_width is not None:
            measurement = measurement.with_maximum(max_width)
        return measurement

    @classmethod
    def get(cls, console: 'Console', options: 'ConsoleOptions', renderable: 'RenderableType') -> 'Measurement':
        if False:
            while True:
                i = 10
        'Get a measurement for a renderable.\n\n        Args:\n            console (~rich.console.Console): Console instance.\n            options (~rich.console.ConsoleOptions): Console options.\n            renderable (RenderableType): An object that may be rendered with Rich.\n\n        Raises:\n            errors.NotRenderableError: If the object is not renderable.\n\n        Returns:\n            Measurement: Measurement object containing range of character widths required to render the object.\n        '
        _max_width = options.max_width
        if _max_width < 1:
            return Measurement(0, 0)
        if isinstance(renderable, str):
            renderable = console.render_str(renderable, markup=options.markup, highlight=False)
        renderable = rich_cast(renderable)
        if is_renderable(renderable):
            get_console_width: Optional[Callable[['Console', 'ConsoleOptions'], 'Measurement']] = getattr(renderable, '__rich_measure__', None)
            if get_console_width is not None:
                render_width = get_console_width(console, options).normalize().with_maximum(_max_width)
                if render_width.maximum < 1:
                    return Measurement(0, 0)
                return render_width.normalize()
            else:
                return Measurement(0, _max_width)
        else:
            raise errors.NotRenderableError(f'Unable to get render width for {renderable!r}; a str, Segment, or object with __rich_console__ method is required')

def measure_renderables(console: 'Console', options: 'ConsoleOptions', renderables: Sequence['RenderableType']) -> 'Measurement':
    if False:
        i = 10
        return i + 15
    'Get a measurement that would fit a number of renderables.\n\n    Args:\n        console (~rich.console.Console): Console instance.\n        options (~rich.console.ConsoleOptions): Console options.\n        renderables (Iterable[RenderableType]): One or more renderable objects.\n\n    Returns:\n        Measurement: Measurement object containing range of character widths required to\n            contain all given renderables.\n    '
    if not renderables:
        return Measurement(0, 0)
    get_measurement = Measurement.get
    measurements = [get_measurement(console, options, renderable) for renderable in renderables]
    measured_width = Measurement(max(measurements, key=itemgetter(0)).minimum, max(measurements, key=itemgetter(1)).maximum)
    return measured_width