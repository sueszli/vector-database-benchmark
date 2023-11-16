""" Various kinds of slider widgets.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import numbers
from datetime import date, datetime, timezone
from ...core.has_props import abstract
from ...core.properties import Bool, Color, Datetime, Either, Enum, Float, Instance, Int, Nullable, Override, Readonly, Required, Seq, String, Tuple
from ...core.property.descriptors import UnsetValueError
from ...core.property.singletons import Undefined
from ...core.validation import error
from ...core.validation.errors import EQUAL_SLIDER_START_END
from ..formatters import TickFormatter
from .widget import Widget
__all__ = ('AbstractSlider', 'CategoricalSlider', 'Slider', 'RangeSlider', 'DateSlider', 'DateRangeSlider', 'DatetimeRangeSlider')

@abstract
class AbstractSlider(Widget):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        try:
            self.lookup('value_throttled')._set(self, Undefined, self.value)
        except UnsetValueError:
            pass
        except AttributeError:
            pass
    orientation = Enum('horizontal', 'vertical', help='\n    Orient the slider either horizontally (default) or vertically.\n    ')
    title = Nullable(String, default='', help="\n    The slider's label (supports :ref:`math text <ug_styling_mathtext>`).\n    ")
    show_value = Bool(default=True, help="\n    Whether or not show slider's value.\n    ")
    direction = Enum('ltr', 'rtl', help='\n    ')
    tooltips = Bool(default=True, help="\n    Display the slider's current value in a tooltip.\n    ")
    bar_color = Color(default='#e6e6e6', help='\n    ')
    width = Override(default=300)

    @error(EQUAL_SLIDER_START_END)
    def _check_missing_dimension(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'start') and hasattr(self, 'end'):
            if self.start == self.end:
                return f'{self!s} with title {self.title!s}'

@abstract
class NumericalSlider(AbstractSlider):
    """ Base class for numerical sliders. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    format = Either(String, Instance(TickFormatter), help='\n    ')

class CategoricalSlider(AbstractSlider):
    """ Discrete slider allowing selection from a collection of values. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    categories = Required(Seq(String), help='\n    A collection of categories to choose from.\n    ')
    value = Required(String, help='\n    Initial or selected value.\n    ')
    value_throttled = Readonly(Required(String), help='\n    Initial or throttled selected value.\n    ')

class Slider(NumericalSlider):
    """ Slider-based number selection widget. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    start = Required(Float, help='\n    The minimum allowable value.\n    ')
    end = Required(Float, help='\n    The maximum allowable value.\n    ')
    value = Required(Float, help='\n    Initial or selected value.\n    ')
    value_throttled = Readonly(Required(Float), help='\n    Initial or selected value, throttled according to report only on mouseup.\n    ')
    step = Float(default=1, help='\n    The step between consecutive values.\n    ')
    format = Override(default='0[.]00')

class RangeSlider(NumericalSlider):
    """ Range-slider based number range selection widget. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    value = Required(Tuple(Float, Float), help='\n    Initial or selected range.\n    ')
    value_throttled = Readonly(Required(Tuple(Float, Float)), help='\n    Initial or selected value, throttled according to report only on mouseup.\n    ')
    start = Required(Float, help='\n    The minimum allowable value.\n    ')
    end = Required(Float, help='\n    The maximum allowable value.\n    ')
    step = Float(default=1, help='\n    The step between consecutive values.\n    ')
    format = Override(default='0[.]00')

class DateSlider(NumericalSlider):
    """ Slider-based date selection widget. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

    @property
    def value_as_datetime(self) -> datetime | None:
        if False:
            return 10
        ' Convenience property to retrieve the value as a datetime object.\n\n        Added in version 2.0\n        '
        if self.value is None:
            return None
        if isinstance(self.value, numbers.Number):
            return datetime.fromtimestamp(self.value / 1000, tz=timezone.utc)
        return self.value

    @property
    def value_as_date(self) -> date | None:
        if False:
            print('Hello World!')
        ' Convenience property to retrieve the value as a date object.\n\n        Added in version 2.0\n        '
        if self.value is None:
            return None
        if isinstance(self.value, numbers.Number):
            dt = datetime.fromtimestamp(self.value / 1000, tz=timezone.utc)
            return date(*dt.timetuple()[:3])
        return self.value
    value = Required(Datetime, help='\n    Initial or selected value.\n    ')
    value_throttled = Readonly(Required(Datetime), help='\n    Initial or selected value, throttled to report only on mouseup.\n    ')
    start = Required(Datetime, help='\n    The minimum allowable value.\n    ')
    end = Required(Datetime, help='\n    The maximum allowable value.\n    ')
    step = Int(default=1, help='\n    The step between consecutive values, in units of days.\n    ')
    format = Override(default='%d %b %Y')

class DateRangeSlider(NumericalSlider):
    """ Slider-based date range selection widget. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

    @property
    def value_as_datetime(self) -> tuple[datetime, datetime] | None:
        if False:
            while True:
                i = 10
        ' Convenience property to retrieve the value tuple as a tuple of\n        datetime objects.\n\n        Added in version 1.1\n        '
        if self.value is None:
            return None
        (v1, v2) = self.value
        if isinstance(v1, numbers.Number):
            d1 = datetime.fromtimestamp(v1 / 1000, tz=timezone.utc)
        else:
            d1 = v1
        if isinstance(v2, numbers.Number):
            d2 = datetime.fromtimestamp(v2 / 1000, tz=timezone.utc)
        else:
            d2 = v2
        return (d1, d2)

    @property
    def value_as_date(self) -> tuple[date, date] | None:
        if False:
            i = 10
            return i + 15
        ' Convenience property to retrieve the value tuple as a tuple of\n        date objects.\n\n        Added in version 1.1\n        '
        if self.value is None:
            return None
        (v1, v2) = self.value
        if isinstance(v1, numbers.Number):
            dt = datetime.fromtimestamp(v1 / 1000, tz=timezone.utc)
            d1 = date(*dt.timetuple()[:3])
        else:
            d1 = v1
        if isinstance(v2, numbers.Number):
            dt = datetime.fromtimestamp(v2 / 1000, tz=timezone.utc)
            d2 = date(*dt.timetuple()[:3])
        else:
            d2 = v2
        return (d1, d2)
    value = Required(Tuple(Datetime, Datetime), help='\n    Initial or selected range.\n    ')
    value_throttled = Readonly(Required(Tuple(Datetime, Datetime)), help='\n    Initial or selected value, throttled to report only on mouseup.\n    ')
    start = Required(Datetime, help='\n    The minimum allowable value.\n    ')
    end = Required(Datetime, help='\n    The maximum allowable value.\n    ')
    step = Int(default=1, help='\n    The step between consecutive values, in units of days.\n    ')
    format = Override(default='%d %b %Y')

class DatetimeRangeSlider(NumericalSlider):
    """ Slider-based datetime range selection widget. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

    @property
    def value_as_datetime(self) -> tuple[datetime, datetime] | None:
        if False:
            for i in range(10):
                print('nop')
        ' Convenience property to retrieve the value tuple as a tuple of\n        datetime objects.\n        '
        if self.value is None:
            return None
        (v1, v2) = self.value
        if isinstance(v1, numbers.Number):
            d1 = datetime.fromtimestamp(v1 / 1000, tz=timezone.utc)
        else:
            d1 = v1
        if isinstance(v2, numbers.Number):
            d2 = datetime.fromtimestamp(v2 / 1000, tz=timezone.utc)
        else:
            d2 = v2
        return (d1, d2)
    value = Required(Tuple(Datetime, Datetime), help='\n    Initial or selected range.\n    ')
    value_throttled = Readonly(Required(Tuple(Datetime, Datetime)), help='\n    Initial or selected value, throttled to report only on mouseup.\n    ')
    start = Required(Datetime, help='\n    The minimum allowable value.\n    ')
    end = Required(Datetime, help='\n    The maximum allowable value.\n    ')
    step = Int(default=3600000, help='\n    The step between consecutive values, in units of milliseconds.\n    Default is one hour.\n    ')
    format = Override(default='%d %b %Y %H:%M:%S')