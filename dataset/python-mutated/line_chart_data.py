from typing import Any, List, Optional, Union
from flet_core.charts.chart_point_line import ChartPointLine
from flet_core.charts.chart_point_shape import ChartPointShape
from flet_core.charts.line_chart_data_point import LineChartDataPoint
from flet_core.control import Control, OptionalNumber
from flet_core.gradients import Gradient
from flet_core.ref import Ref
from flet_core.shadow import BoxShadow

class LineChartData(Control):

    def __init__(self, data_points: Optional[List[LineChartDataPoint]]=None, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, curved: Optional[bool]=None, color: Optional[str]=None, gradient: Optional[Gradient]=None, stroke_width: OptionalNumber=None, stroke_cap_round: Optional[bool]=None, dash_pattern: Optional[List[int]]=None, shadow: Optional[BoxShadow]=None, above_line_bgcolor: Optional[str]=None, above_line_gradient: Optional[Gradient]=None, above_line_cutoff_y: OptionalNumber=None, above_line: Optional[ChartPointLine]=None, below_line_bgcolor: Optional[str]=None, below_line_gradient: Optional[Gradient]=None, below_line_cutoff_y: OptionalNumber=None, below_line: Optional[ChartPointLine]=None, selected_below_line: Union[None, bool, ChartPointLine]=None, point: Union[None, bool, ChartPointShape]=None, selected_point: Union[None, bool, ChartPointShape]=None):
        if False:
            return 10
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.data_points = data_points
        self.curved = curved
        self.color = color
        self.gradient = gradient
        self.stroke_width = stroke_width
        self.stroke_cap_round = stroke_cap_round
        self.shadow = shadow
        self.dash_pattern = dash_pattern
        self.above_line_bgcolor = above_line_bgcolor
        self.above_line_gradient = above_line_gradient
        self.above_line_cutoff_y = above_line_cutoff_y
        self.above_line = above_line
        self.below_line_bgcolor = below_line_bgcolor
        self.below_line_gradient = below_line_gradient
        self.below_line_cutoff_y = below_line_cutoff_y
        self.below_line = below_line
        self.selected_below_line = selected_below_line
        self.point = point
        self.selected_point = selected_point

    def _get_control_name(self):
        if False:
            while True:
                i = 10
        return 'data'

    def _before_build_command(self):
        if False:
            return 10
        super()._before_build_command()
        self._set_attr_json('gradient', self.__gradient)
        self._set_attr_json('shadow', self.__shadow)
        self._set_attr_json('point', self.__point)
        self._set_attr_json('selectedPoint', self.__selected_point)
        self._set_attr_json('dashPattern', self.__dash_pattern)
        self._set_attr_json('aboveLineGradient', self.__above_line_gradient)
        self._set_attr_json('belowLineGradient', self.__below_line_gradient)
        self._set_attr_json('aboveLine', self.__above_line)
        self._set_attr_json('belowLine', self.__below_line)
        self._set_attr_json('selectedBelowLine', self.__selected_below_line)

    def _get_children(self):
        if False:
            return 10
        return self.__data_points

    @property
    def data_points(self):
        if False:
            return 10
        return self.__data_points

    @data_points.setter
    def data_points(self, value):
        if False:
            i = 10
            return i + 15
        self.__data_points = value if value is not None else []

    @property
    def stroke_width(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('strokeWidth', data_type='float', def_value=1.0)

    @stroke_width.setter
    def stroke_width(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('strokeWidth', value)

    @property
    def curved(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self._get_attr('curved', data_type='bool', def_value=False)

    @curved.setter
    def curved(self, value: Optional[bool]):
        if False:
            print('Hello World!')
        self._set_attr('curved', value)

    @property
    def color(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._get_attr('color')

    @color.setter
    def color(self, value: Optional[str]):
        if False:
            print('Hello World!')
        self._set_attr('color', value)

    @property
    def gradient(self) -> Optional[Gradient]:
        if False:
            return 10
        return self.__gradient

    @gradient.setter
    def gradient(self, value: Optional[Gradient]):
        if False:
            for i in range(10):
                print('nop')
        self.__gradient = value

    @property
    def stroke_cap_round(self) -> Optional[bool]:
        if False:
            print('Hello World!')
        return self._get_attr('strokeCapRound', data_type='bool', def_value=False)

    @stroke_cap_round.setter
    def stroke_cap_round(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('strokeCapRound', value)

    @property
    def dash_pattern(self):
        if False:
            while True:
                i = 10
        return self.__dash_pattern

    @dash_pattern.setter
    def dash_pattern(self, value: Optional[List[int]]):
        if False:
            print('Hello World!')
        self.__dash_pattern = value

    @property
    def shadow(self):
        if False:
            print('Hello World!')
        return self.__shadow

    @shadow.setter
    def shadow(self, value: Optional[BoxShadow]):
        if False:
            print('Hello World!')
        self.__shadow = value

    @property
    def point(self):
        if False:
            print('Hello World!')
        return self.__point

    @point.setter
    def point(self, value: Union[None, bool, ChartPointShape]):
        if False:
            print('Hello World!')
        self.__point = value

    @property
    def selected_point(self):
        if False:
            return 10
        return self.__selected_point

    @selected_point.setter
    def selected_point(self, value: Union[None, bool, ChartPointShape]):
        if False:
            for i in range(10):
                print('nop')
        self.__selected_point = value

    @property
    def above_line_bgcolor(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._get_attr('aboveLineBgcolor')

    @above_line_bgcolor.setter
    def above_line_bgcolor(self, value: Optional[str]):
        if False:
            while True:
                i = 10
        self._set_attr('aboveLineBgcolor', value)

    @property
    def above_line_gradient(self) -> Optional[Gradient]:
        if False:
            i = 10
            return i + 15
        return self.__above_line_gradient

    @above_line_gradient.setter
    def above_line_gradient(self, value: Optional[Gradient]):
        if False:
            return 10
        self.__above_line_gradient = value

    @property
    def above_line_cutoff_y(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('aboveLineCutoffY', data_type='float')

    @above_line_cutoff_y.setter
    def above_line_cutoff_y(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('aboveLineCutoffY', value)

    @property
    def above_line(self) -> Optional[ChartPointLine]:
        if False:
            print('Hello World!')
        return self.__above_line

    @above_line.setter
    def above_line(self, value: Optional[ChartPointLine]):
        if False:
            while True:
                i = 10
        self.__above_line = value

    @property
    def below_line_bgcolor(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('belowLineBgcolor')

    @below_line_bgcolor.setter
    def below_line_bgcolor(self, value: Optional[str]):
        if False:
            i = 10
            return i + 15
        self._set_attr('belowLineBgcolor', value)

    @property
    def below_line_gradient(self) -> Optional[Gradient]:
        if False:
            i = 10
            return i + 15
        return self.__below_line_gradient

    @below_line_gradient.setter
    def below_line_gradient(self, value: Optional[Gradient]):
        if False:
            i = 10
            return i + 15
        self.__below_line_gradient = value

    @property
    def below_line_cutoff_y(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('belowLineCutoffY', data_type='float')

    @below_line_cutoff_y.setter
    def below_line_cutoff_y(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('belowLineCutoffY', value)

    @property
    def below_line(self) -> Optional[ChartPointLine]:
        if False:
            for i in range(10):
                print('nop')
        return self.__below_line

    @below_line.setter
    def below_line(self, value: Optional[ChartPointLine]):
        if False:
            return 10
        self.__below_line = value

    @property
    def selected_below_line(self) -> Union[None, bool, ChartPointLine]:
        if False:
            for i in range(10):
                print('nop')
        return self.__selected_below_line

    @selected_below_line.setter
    def selected_below_line(self, value: Union[None, bool, ChartPointLine]):
        if False:
            while True:
                i = 10
        self.__selected_below_line = value