from typing import Union, Tuple
import copy
import re
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from .scaling_tracker import ScalingTracker
from ..font import CTkFont

class CTkScalingBaseClass:
    """
    Super-class that manages the scaling values and callbacks.
    Works for widgets and windows, type must be set in init method with
    scaling_type attribute. Methods:

    - _set_scaling() abstractmethod, gets called when scaling changes, must be overridden
    - destroy() must be called when sub-class is destroyed
    - _apply_widget_scaling()
    - _reverse_widget_scaling()
    - _apply_window_scaling()
    - _reverse_window_scaling()
    - _apply_font_scaling()
    - _apply_argument_scaling()
    - _apply_geometry_scaling()
    - _reverse_geometry_scaling()
    - _parse_geometry_string()

    """

    def __init__(self, scaling_type: Literal['widget', 'window']='widget'):
        if False:
            while True:
                i = 10
        self.__scaling_type = scaling_type
        if self.__scaling_type == 'widget':
            ScalingTracker.add_widget(self._set_scaling, self)
            self.__widget_scaling = ScalingTracker.get_widget_scaling(self)
        elif self.__scaling_type == 'window':
            ScalingTracker.activate_high_dpi_awareness()
            ScalingTracker.add_window(self._set_scaling, self)
            self.__window_scaling = ScalingTracker.get_window_scaling(self)

    def destroy(self):
        if False:
            return 10
        if self.__scaling_type == 'widget':
            ScalingTracker.remove_widget(self._set_scaling, self)
        elif self.__scaling_type == 'window':
            ScalingTracker.remove_window(self._set_scaling, self)

    def _set_scaling(self, new_widget_scaling, new_window_scaling):
        if False:
            return 10
        ' can be overridden, but super method must be called at the beginning '
        self.__widget_scaling = new_widget_scaling
        self.__window_scaling = new_window_scaling

    def _get_widget_scaling(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return self.__widget_scaling

    def _get_window_scaling(self) -> float:
        if False:
            print('Hello World!')
        return self.__window_scaling

    def _apply_widget_scaling(self, value: Union[int, float]) -> Union[float]:
        if False:
            i = 10
            return i + 15
        assert self.__scaling_type == 'widget'
        return value * self.__widget_scaling

    def _reverse_widget_scaling(self, value: Union[int, float]) -> Union[float]:
        if False:
            print('Hello World!')
        assert self.__scaling_type == 'widget'
        return value / self.__widget_scaling

    def _apply_window_scaling(self, value: Union[int, float]) -> int:
        if False:
            for i in range(10):
                print('nop')
        assert self.__scaling_type == 'window'
        return int(value * self.__window_scaling)

    def _reverse_window_scaling(self, scaled_value: Union[int, float]) -> int:
        if False:
            for i in range(10):
                print('nop')
        assert self.__scaling_type == 'window'
        return int(scaled_value / self.__window_scaling)

    def _apply_font_scaling(self, font: Union[Tuple, CTkFont]) -> tuple:
        if False:
            return 10
        ' Takes CTkFont object and returns tuple font with scaled size, has to be called again for every change of font object '
        assert self.__scaling_type == 'widget'
        if type(font) == tuple:
            if len(font) == 1:
                return font
            elif len(font) == 2:
                return (font[0], -abs(round(font[1] * self.__widget_scaling)))
            elif 3 <= len(font) <= 6:
                return (font[0], -abs(round(font[1] * self.__widget_scaling)), font[2:])
            else:
                raise ValueError(f'Can not scale font {font}. font needs to be tuple of len 1, 2 or 3')
        elif isinstance(font, CTkFont):
            return font.create_scaled_tuple(self.__widget_scaling)
        else:
            raise ValueError(f"Can not scale font '{font}' of type {type(font)}. font needs to be tuple or instance of CTkFont")

    def _apply_argument_scaling(self, kwargs: dict) -> dict:
        if False:
            return 10
        assert self.__scaling_type == 'widget'
        scaled_kwargs = copy.copy(kwargs)
        if 'pady' in scaled_kwargs:
            if isinstance(scaled_kwargs['pady'], (int, float)):
                scaled_kwargs['pady'] = self._apply_widget_scaling(scaled_kwargs['pady'])
            elif isinstance(scaled_kwargs['pady'], tuple):
                scaled_kwargs['pady'] = tuple([self._apply_widget_scaling(v) for v in scaled_kwargs['pady']])
        if 'padx' in kwargs:
            if isinstance(scaled_kwargs['padx'], (int, float)):
                scaled_kwargs['padx'] = self._apply_widget_scaling(scaled_kwargs['padx'])
            elif isinstance(scaled_kwargs['padx'], tuple):
                scaled_kwargs['padx'] = tuple([self._apply_widget_scaling(v) for v in scaled_kwargs['padx']])
        if 'x' in scaled_kwargs:
            scaled_kwargs['x'] = self._apply_widget_scaling(scaled_kwargs['x'])
        if 'y' in scaled_kwargs:
            scaled_kwargs['y'] = self._apply_widget_scaling(scaled_kwargs['y'])
        return scaled_kwargs

    @staticmethod
    def _parse_geometry_string(geometry_string: str) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        result = re.search('((\\d+)x(\\d+)){0,1}(\\+{0,1}([+-]{0,1}\\d+)\\+{0,1}([+-]{0,1}\\d+)){0,1}', geometry_string)
        width = int(result.group(2)) if result.group(2) is not None else None
        height = int(result.group(3)) if result.group(3) is not None else None
        x = int(result.group(5)) if result.group(5) is not None else None
        y = int(result.group(6)) if result.group(6) is not None else None
        return (width, height, x, y)

    def _apply_geometry_scaling(self, geometry_string: str) -> str:
        if False:
            print('Hello World!')
        assert self.__scaling_type == 'window'
        (width, height, x, y) = self._parse_geometry_string(geometry_string)
        if x is None and y is None:
            return f'{round(width * self.__window_scaling)}x{round(height * self.__window_scaling)}'
        elif width is None and height is None:
            return f'+{x}+{y}'
        else:
            return f'{round(width * self.__window_scaling)}x{round(height * self.__window_scaling)}+{x}+{y}'

    def _reverse_geometry_scaling(self, scaled_geometry_string: str) -> str:
        if False:
            i = 10
            return i + 15
        assert self.__scaling_type == 'window'
        (width, height, x, y) = self._parse_geometry_string(scaled_geometry_string)
        if x is None and y is None:
            return f'{round(width / self.__window_scaling)}x{round(height / self.__window_scaling)}'
        elif width is None and height is None:
            return f'+{x}+{y}'
        else:
            return f'{round(width / self.__window_scaling)}x{round(height / self.__window_scaling)}+{x}+{y}'