from enum import Enum
from typing import Any, Optional, Union
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.text_style import TextStyle
from flet_core.types import AnimationValue, BorderRadiusValue, OffsetValue, PaddingValue, ResponsiveNumber, RotateValue, ScaleValue
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
InputBorderString = Literal[None, 'outline', 'underline', 'none']

class InputBorder(Enum):
    NONE = 'none'
    OUTLINE = 'outline'
    UNDERLINE = 'underline'

class FormFieldControl(ConstrainedControl):

    def __init__(self, ref: Optional[Ref]=None, key: Optional[str]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, tooltip: Optional[str]=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, text_size: OptionalNumber=None, text_style: Optional[TextStyle]=None, label: Optional[str]=None, label_style: Optional[TextStyle]=None, icon: Optional[str]=None, border: Optional[InputBorder]=None, color: Optional[str]=None, bgcolor: Optional[str]=None, border_radius: BorderRadiusValue=None, border_width: OptionalNumber=None, border_color: Optional[str]=None, focused_color: Optional[str]=None, focused_bgcolor: Optional[str]=None, focused_border_width: OptionalNumber=None, focused_border_color: Optional[str]=None, content_padding: PaddingValue=None, dense: Optional[bool]=None, filled: Optional[bool]=None, hint_text: Optional[str]=None, hint_style: Optional[TextStyle]=None, helper_text: Optional[str]=None, helper_style: Optional[TextStyle]=None, counter_text: Optional[str]=None, counter_style: Optional[TextStyle]=None, error_text: Optional[str]=None, error_style: Optional[TextStyle]=None, prefix: Optional[Control]=None, prefix_icon: Optional[str]=None, prefix_text: Optional[str]=None, prefix_style: Optional[TextStyle]=None, suffix: Optional[Control]=None, suffix_icon: Optional[str]=None, suffix_text: Optional[str]=None, suffix_style: Optional[TextStyle]=None):
        if False:
            i = 10
            return i + 15
        ConstrainedControl.__init__(self, ref=ref, key=key, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, tooltip=tooltip, visible=visible, disabled=disabled, data=data)
        self.text_size = text_size
        self.text_style = text_style
        self.label = label
        self.label_style = label_style
        self.icon = icon
        self.border = border
        self.color = color
        self.bgcolor = bgcolor
        self.border_radius = border_radius
        self.border_width = border_width
        self.border_color = border_color
        self.focused_color = focused_color
        self.focused_bgcolor = focused_bgcolor
        self.focused_border_width = focused_border_width
        self.focused_border_color = focused_border_color
        self.content_padding = content_padding
        self.filled = filled
        self.dense = dense
        self.hint_text = hint_text
        self.hint_style = hint_style
        self.helper_text = helper_text
        self.helper_style = helper_style
        self.counter_text = counter_text
        self.counter_style = counter_style
        self.error_text = error_text
        self.error_style = error_style
        self.prefix = prefix
        self.prefix_icon = prefix_icon
        self.prefix_text = prefix_text
        self.prefix_style = prefix_style
        self.suffix = suffix
        self.suffix_icon = suffix_icon
        self.suffix_text = suffix_text
        self.suffix_style = suffix_style

    def _before_build_command(self):
        if False:
            i = 10
            return i + 15
        super()._before_build_command()
        self._set_attr_json('borderRadius', self.__border_radius)
        self._set_attr_json('contentPadding', self.__content_padding)
        self._set_attr_json('textStyle', self.__text_style)
        self._set_attr_json('labelStyle', self.__label_style)
        self._set_attr_json('hintStyle', self.__hint_style)
        self._set_attr_json('helperStyle', self.__helper_style)
        self._set_attr_json('counterStyle', self.__counter_style)
        self._set_attr_json('errorStyle', self.__error_style)
        self._set_attr_json('prefixStyle', self.__prefix_style)
        self._set_attr_json('suffixStyle', self.__suffix_style)

    def _get_children(self):
        if False:
            return 10
        children = []
        if self.__prefix:
            self.__prefix._set_attr_internal('n', 'prefix')
            children.append(self.__prefix)
        if self.__suffix:
            self.__suffix._set_attr_internal('n', 'suffix')
            children.append(self.__suffix)
        return children

    @property
    def text_size(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('textSize')

    @text_size.setter
    def text_size(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('textSize', value)

    @property
    def text_style(self):
        if False:
            print('Hello World!')
        return self.__text_style

    @text_style.setter
    def text_style(self, value: Optional[TextStyle]):
        if False:
            while True:
                i = 10
        self.__text_style = value

    @property
    def label(self):
        if False:
            while True:
                i = 10
        return self._get_attr('label')

    @label.setter
    def label(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('label', value)

    @property
    def label_style(self):
        if False:
            print('Hello World!')
        return self.__label_style

    @label_style.setter
    def label_style(self, value: Optional[TextStyle]):
        if False:
            i = 10
            return i + 15
        self.__label_style = value

    @property
    def icon(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('icon')

    @icon.setter
    def icon(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('icon', value)

    @property
    def border(self) -> Optional[InputBorder]:
        if False:
            return 10
        return self.__border

    @border.setter
    def border(self, value: Optional[InputBorder]):
        if False:
            for i in range(10):
                print('nop')
        self.__border = value
        if isinstance(value, InputBorder):
            self._set_attr('border', value.value)
        else:
            self.__set_border(value)

    def __set_border(self, value: Optional[InputBorderString]):
        if False:
            print('Hello World!')
        self._set_attr('border', value)

    @property
    def color(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('color')

    @color.setter
    def color(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('color', value)

    @property
    def bgcolor(self):
        if False:
            while True:
                i = 10
        return self._get_attr('bgcolor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            print('Hello World!')
        self._set_attr('bgcolor', value)

    @property
    def border_radius(self) -> BorderRadiusValue:
        if False:
            for i in range(10):
                print('nop')
        return self.__border_radius

    @border_radius.setter
    def border_radius(self, value: BorderRadiusValue):
        if False:
            return 10
        self.__border_radius = value

    @property
    def border_width(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('borderWidth')

    @border_width.setter
    def border_width(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('borderWidth', value)

    @property
    def border_color(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('borderColor')

    @border_color.setter
    def border_color(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('borderColor', value)

    @property
    def focused_color(self):
        if False:
            while True:
                i = 10
        return self._get_attr('focusedColor')

    @focused_color.setter
    def focused_color(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('focusedColor', value)

    @property
    def focused_bgcolor(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('focusedBgcolor')

    @focused_bgcolor.setter
    def focused_bgcolor(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('focusedBgcolor', value)

    @property
    def focused_border_width(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('focusedBorderWidth')

    @focused_border_width.setter
    def focused_border_width(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('focusedBorderWidth', value)

    @property
    def focused_border_color(self):
        if False:
            return 10
        return self._get_attr('focusedBorderColor')

    @focused_border_color.setter
    def focused_border_color(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('focusedBorderColor', value)

    @property
    def content_padding(self) -> PaddingValue:
        if False:
            print('Hello World!')
        return self.__content_padding

    @content_padding.setter
    def content_padding(self, value: PaddingValue):
        if False:
            while True:
                i = 10
        self.__content_padding = value

    @property
    def dense(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('dense')

    @dense.setter
    def dense(self, value: Optional[bool]):
        if False:
            i = 10
            return i + 15
        self._set_attr('dense', value)

    @property
    def filled(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('filled')

    @filled.setter
    def filled(self, value: Optional[bool]):
        if False:
            print('Hello World!')
        self._set_attr('filled', value)

    @property
    def hint_text(self):
        if False:
            print('Hello World!')
        return self._get_attr('hintText')

    @hint_text.setter
    def hint_text(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('hintText', value)

    @property
    def hint_style(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__hint_style

    @hint_style.setter
    def hint_style(self, value: Optional[TextStyle]):
        if False:
            while True:
                i = 10
        self.__hint_style = value

    @property
    def helper_text(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('helperText')

    @helper_text.setter
    def helper_text(self, value):
        if False:
            return 10
        self._set_attr('helperText', value)

    @property
    def helper_style(self):
        if False:
            while True:
                i = 10
        return self.__helper_style

    @helper_style.setter
    def helper_style(self, value: Optional[TextStyle]):
        if False:
            return 10
        self.__helper_style = value

    @property
    def counter_text(self):
        if False:
            print('Hello World!')
        return self._get_attr('counterText')

    @counter_text.setter
    def counter_text(self, value):
        if False:
            return 10
        self._set_attr('counterText', value)

    @property
    def counter_style(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__counter_style

    @counter_style.setter
    def counter_style(self, value: Optional[TextStyle]):
        if False:
            print('Hello World!')
        self.__counter_style = value

    @property
    def error_text(self):
        if False:
            return 10
        return self._get_attr('errorText')

    @error_text.setter
    def error_text(self, value):
        if False:
            return 10
        self._set_attr('errorText', value)

    @property
    def error_style(self):
        if False:
            while True:
                i = 10
        return self.__error_style

    @error_style.setter
    def error_style(self, value: Optional[TextStyle]):
        if False:
            return 10
        self.__error_style = value

    @property
    def prefix(self):
        if False:
            i = 10
            return i + 15
        return self.__prefix

    @prefix.setter
    def prefix(self, value):
        if False:
            print('Hello World!')
        self.__prefix = value

    @property
    def prefix_icon(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('prefixIcon')

    @prefix_icon.setter
    def prefix_icon(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('prefixIcon', value)

    @property
    def prefix_text(self):
        if False:
            return 10
        return self._get_attr('prefixText')

    @prefix_text.setter
    def prefix_text(self, value):
        if False:
            return 10
        self._set_attr('prefixText', value)

    @property
    def prefix_style(self):
        if False:
            while True:
                i = 10
        return self.__prefix_style

    @prefix_style.setter
    def prefix_style(self, value: Optional[TextStyle]):
        if False:
            i = 10
            return i + 15
        self.__prefix_style = value

    @property
    def suffix(self):
        if False:
            while True:
                i = 10
        return self.__suffix

    @suffix.setter
    def suffix(self, value):
        if False:
            while True:
                i = 10
        self.__suffix = value

    @property
    def suffix_icon(self):
        if False:
            return 10
        return self._get_attr('suffixIcon')

    @suffix_icon.setter
    def suffix_icon(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('suffixIcon', value)

    @property
    def suffix_text(self):
        if False:
            return 10
        return self._get_attr('suffixText')

    @suffix_text.setter
    def suffix_text(self, value):
        if False:
            print('Hello World!')
        self._set_attr('suffixText', value)

    @property
    def suffix_style(self):
        if False:
            print('Hello World!')
        return self.__suffix_style

    @suffix_style.setter
    def suffix_style(self, value: Optional[TextStyle]):
        if False:
            while True:
                i = 10
        self.__suffix_style = value