import dataclasses
from typing import Any, List, Optional
from flet_core.canvas.path import Path
from flet_core.canvas.shape import Shape
from flet_core.control import OptionalNumber

class Shadow(Shape):

    def __init__(self, path: Optional[List[Path.PathElement]]=None, color: Optional[str]=None, elevation: OptionalNumber=None, transparent_occluder: Optional[bool]=None, ref=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None):
        if False:
            return 10
        Shape.__init__(self, ref=ref, visible=visible, disabled=disabled, data=data)
        self.path = path
        self.color = color
        self.elevation = elevation
        self.transparent_occluder = transparent_occluder

    def _get_control_name(self):
        if False:
            i = 10
            return i + 15
        return 'shadow'

    def _before_build_command(self):
        if False:
            print('Hello World!')
        super()._before_build_command()
        self._set_attr_json('path', self.__path)

    @property
    def path(self):
        if False:
            while True:
                i = 10
        return self.__path

    @path.setter
    def path(self, value: Optional[List[Path.PathElement]]):
        if False:
            print('Hello World!')
        self.__path = value if value is not None else []

    @property
    def color(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('color')

    @color.setter
    def color(self, value: Optional[str]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('color', value)

    @property
    def elevation(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('elevation')

    @elevation.setter
    def elevation(self, value: OptionalNumber):
        if False:
            print('Hello World!')
        self._set_attr('elevation', value)

    @property
    def transparent_occluder(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('transparentOccluder', data_type='bool', def_value=False)

    @transparent_occluder.setter
    def transparent_occluder(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('transparentOccluder', value)