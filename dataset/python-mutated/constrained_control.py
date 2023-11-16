from typing import Any, Optional, Union
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, OffsetValue, ResponsiveNumber, RotateValue, ScaleValue

class ConstrainedControl(Control):

    def __init__(self, ref: Optional[Ref]=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, tooltip: Optional[str]=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, key: Optional[str]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None):
        if False:
            for i in range(10):
                print('nop')
        Control.__init__(self, ref=ref, expand=expand, col=col, opacity=opacity, tooltip=tooltip, visible=visible, disabled=disabled, data=data)
        self.key = key
        self.width = width
        self.height = height
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.scale = scale
        self.rotate = rotate
        self.offset = offset
        self.aspect_ratio = aspect_ratio
        self.animate_opacity = animate_opacity
        self.animate_size = animate_size
        self.animate_position = animate_position
        self.animate_rotation = animate_rotation
        self.animate_scale = animate_scale
        self.animate_offset = animate_offset
        self.on_animation_end = on_animation_end

    def _before_build_command(self):
        if False:
            return 10
        super()._before_build_command()
        self._set_attr_json('rotate', self.__rotate)
        self._set_attr_json('scale', self.__scale)
        self._set_attr_json('offset', self.__offset)
        self._set_attr_json('animateOpacity', self.__animate_opacity)
        self._set_attr_json('animateSize', self.__animate_size)
        self._set_attr_json('animatePosition', self.__animate_position)
        self._set_attr_json('animateRotation', self.__animate_rotation)
        self._set_attr_json('animateScale', self.__animate_scale)
        self._set_attr_json('animateOffset', self.__animate_offset)

    @property
    def key(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._get_attr('key')

    @key.setter
    def key(self, value: Optional[str]):
        if False:
            print('Hello World!')
        self._set_attr('key', value)

    @property
    def width(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        '\n        Control width.\n        '
        return self._get_attr('width')

    @width.setter
    def width(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('width', value)

    @property
    def height(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('height')

    @height.setter
    def height(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('height', value)

    @property
    def left(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('left')

    @left.setter
    def left(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('left', value)

    @property
    def top(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('top')

    @top.setter
    def top(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('top', value)

    @property
    def right(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('right')

    @right.setter
    def right(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('right', value)

    @property
    def bottom(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('bottom')

    @bottom.setter
    def bottom(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('bottom', value)

    @property
    def rotate(self) -> RotateValue:
        if False:
            return 10
        return self.__rotate

    @rotate.setter
    def rotate(self, value: RotateValue):
        if False:
            i = 10
            return i + 15
        self.__rotate = value

    @property
    def scale(self) -> ScaleValue:
        if False:
            print('Hello World!')
        return self.__scale

    @scale.setter
    def scale(self, value: ScaleValue):
        if False:
            for i in range(10):
                print('nop')
        self.__scale = value

    @property
    def offset(self) -> OffsetValue:
        if False:
            while True:
                i = 10
        return self.__offset

    @offset.setter
    def offset(self, value: OffsetValue):
        if False:
            print('Hello World!')
        self.__offset = value

    @property
    def aspect_ratio(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('aspectRatio')

    @aspect_ratio.setter
    def aspect_ratio(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('aspectRatio', value)

    @property
    def animate_opacity(self) -> AnimationValue:
        if False:
            i = 10
            return i + 15
        return self.__animate_opacity

    @animate_opacity.setter
    def animate_opacity(self, value: AnimationValue):
        if False:
            i = 10
            return i + 15
        self.__animate_opacity = value

    @property
    def animate_size(self) -> AnimationValue:
        if False:
            return 10
        return self.__animate_size

    @animate_size.setter
    def animate_size(self, value: AnimationValue):
        if False:
            print('Hello World!')
        self.__animate_size = value

    @property
    def animate_position(self) -> AnimationValue:
        if False:
            return 10
        return self.__animate_position

    @animate_position.setter
    def animate_position(self, value: AnimationValue):
        if False:
            return 10
        self.__animate_position = value

    @property
    def animate_rotation(self) -> AnimationValue:
        if False:
            for i in range(10):
                print('nop')
        return self.__animate_rotation

    @animate_rotation.setter
    def animate_rotation(self, value: AnimationValue):
        if False:
            while True:
                i = 10
        self.__animate_rotation = value

    @property
    def animate_scale(self) -> AnimationValue:
        if False:
            print('Hello World!')
        return self.__animate_scale

    @animate_scale.setter
    def animate_scale(self, value: AnimationValue):
        if False:
            while True:
                i = 10
        self.__animate_scale = value

    @property
    def animate_offset(self) -> AnimationValue:
        if False:
            i = 10
            return i + 15
        return self.__animate_offset

    @animate_offset.setter
    def animate_offset(self, value: AnimationValue):
        if False:
            for i in range(10):
                print('nop')
        self.__animate_offset = value

    @property
    def on_animation_end(self):
        if False:
            while True:
                i = 10
        return self._get_event_handler('animation_end')

    @on_animation_end.setter
    def on_animation_end(self, handler):
        if False:
            return 10
        self._add_event_handler('animation_end', handler)
        if handler is not None:
            self._set_attr('onAnimationEnd', True)
        else:
            self._set_attr('onAnimationEnd', None)