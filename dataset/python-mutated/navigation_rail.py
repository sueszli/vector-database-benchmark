from enum import Enum
from typing import Any, List, Optional, Union
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, OffsetValue, PaddingValue, ResponsiveNumber, RotateValue, ScaleValue
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
NavigationRailLabelTypeString = Literal[None, 'none', 'all', 'selected']

class NavigationRailLabelType(Enum):
    NONE = 'none'
    ALL = 'all'
    SELECTED = 'selected'

class NavigationRailDestination(Control):

    def __init__(self, ref: Optional[Ref]=None, icon: Optional[str]=None, icon_content: Optional[Control]=None, selected_icon: Optional[str]=None, selected_icon_content: Optional[Control]=None, label: Optional[str]=None, label_content: Optional[Control]=None, padding: PaddingValue=None):
        if False:
            while True:
                i = 10
        Control.__init__(self, ref=ref)
        self.label = label
        self.icon = icon
        self.__icon_content: Optional[Control] = None
        self.icon_content = icon_content
        self.selected_icon = selected_icon
        self.__selected_icon_content: Optional[Control] = None
        self.selected_icon_content = selected_icon_content
        self.__label_content: Optional[Control] = None
        self.label_content = label_content
        self.padding = padding

    def _get_control_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'navigationraildestination'

    def _before_build_command(self):
        if False:
            i = 10
            return i + 15
        super()._before_build_command()
        self._set_attr_json('padding', self.__padding)

    def _get_children(self):
        if False:
            i = 10
            return i + 15
        children = []
        if self.__label_content:
            self.__label_content._set_attr_internal('n', 'label_content')
            children.append(self.__label_content)
        if self.__icon_content:
            self.__icon_content._set_attr_internal('n', 'icon_content')
            children.append(self.__icon_content)
        if self.__selected_icon_content:
            self.__selected_icon_content._set_attr_internal('n', 'selected_icon_content')
            children.append(self.__selected_icon_content)
        return children

    @property
    def icon(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('icon')

    @icon.setter
    def icon(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('icon', value)

    @property
    def icon_content(self) -> Optional[Control]:
        if False:
            print('Hello World!')
        return self.__icon_content

    @icon_content.setter
    def icon_content(self, value: Optional[Control]):
        if False:
            for i in range(10):
                print('nop')
        self.__icon_content = value

    @property
    def selected_icon(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('selectedIcon')

    @selected_icon.setter
    def selected_icon(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('selectedIcon', value)

    @property
    def selected_icon_content(self) -> Optional[Control]:
        if False:
            while True:
                i = 10
        return self.__selected_icon_content

    @selected_icon_content.setter
    def selected_icon_content(self, value: Optional[Control]):
        if False:
            i = 10
            return i + 15
        self.__selected_icon_content = value

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('label')

    @label.setter
    def label(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('label', value)

    @property
    def label_content(self) -> Optional[Control]:
        if False:
            i = 10
            return i + 15
        return self.__label_content

    @label_content.setter
    def label_content(self, value: Optional[Control]):
        if False:
            for i in range(10):
                print('nop')
        self.__label_content = value

    @property
    def padding(self) -> PaddingValue:
        if False:
            print('Hello World!')
        return self.__padding

    @padding.setter
    def padding(self, value: PaddingValue):
        if False:
            return 10
        self.__padding = value

class NavigationRail(ConstrainedControl):
    """
    A material widget that is meant to be displayed at the left or right of an app to navigate between a small number of views, typically between three and five.

    Example:

    ```
    import flet as ft

    def main(page: ft.Page):

        rail = ft.NavigationRail(
            selected_index=1,
            label_type=ft.NavigationRailLabelType.ALL,
            # extended=True,
            min_width=100,
            min_extended_width=400,
            leading=ft.FloatingActionButton(icon=ft.icons.CREATE, text="Add"),
            group_alignment=-0.9,
            destinations=[
                ft.NavigationRailDestination(
                    icon=ft.icons.FAVORITE_BORDER, selected_icon=ft.icons.FAVORITE, label="First"
                ),
                ft.NavigationRailDestination(
                    icon_content=ft.Icon(ft.icons.BOOKMARK_BORDER),
                    selected_icon_content=ft.Icon(ft.icons.BOOKMARK),
                    label="Second",
                ),
                ft.NavigationRailDestination(
                    icon=ft.icons.SETTINGS_OUTLINED,
                    selected_icon_content=ft.Icon(ft.icons.SETTINGS),
                    label_content=ft.Text("Settings"),
                ),
            ],
            on_change=lambda e: print("Selected destination:", e.control.selected_index),
        )

        page.add(
            ft.Row(
                [
                    rail,
                    ft.VerticalDivider(width=1),
                    ft.Column([ ft.Text("Body!")], alignment=ft.MainAxisAlignment.START, expand=True),
                ],
                expand=True,
            )
        )

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/navigationrail
    """

    def __init__(self, ref: Optional[Ref]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, destinations: Optional[List[NavigationRailDestination]]=None, selected_index: Optional[int]=None, extended: Optional[bool]=None, label_type: Optional[NavigationRailLabelType]=None, bgcolor: Optional[str]=None, leading: Optional[Control]=None, trailing: Optional[Control]=None, min_width: OptionalNumber=None, min_extended_width: OptionalNumber=None, group_alignment: OptionalNumber=None, on_change=None):
        if False:
            while True:
                i = 10
        ConstrainedControl.__init__(self, ref=ref, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, visible=visible, disabled=disabled, data=data)
        self.destinations = destinations
        self.selected_index = selected_index
        self.extended = extended
        self.label_type = label_type
        self.bgcolor = bgcolor
        self.__leading = None
        self.leading = leading
        self.__trailing = trailing
        self.trailing = trailing
        self.min_width = min_width
        self.min_extended_width = min_extended_width
        self.group_alignment = group_alignment
        self.on_change = on_change

    def _get_control_name(self):
        if False:
            i = 10
            return i + 15
        return 'navigationrail'

    def _get_children(self):
        if False:
            print('Hello World!')
        children = []
        if self.__leading:
            self.__leading._set_attr_internal('n', 'leading')
            children.append(self.__leading)
        if self.__trailing:
            self.__trailing._set_attr_internal('n', 'trailing')
            children.append(self.__trailing)
        children.extend(self.__destinations)
        return children

    @property
    def destinations(self) -> Optional[List[NavigationRailDestination]]:
        if False:
            for i in range(10):
                print('nop')
        return self.__destinations

    @destinations.setter
    def destinations(self, value: Optional[List[NavigationRailDestination]]):
        if False:
            i = 10
            return i + 15
        self.__destinations = value if value is not None else []

    @property
    def on_change(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_event_handler('change')

    @on_change.setter
    def on_change(self, handler):
        if False:
            print('Hello World!')
        self._add_event_handler('change', handler)

    @property
    def selected_index(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('selectedIndex', data_type='int', def_value=0)

    @selected_index.setter
    def selected_index(self, value: Optional[int]):
        if False:
            i = 10
            return i + 15
        self._set_attr('selectedIndex', value)

    @property
    def label_type(self) -> Optional[NavigationRailLabelType]:
        if False:
            for i in range(10):
                print('nop')
        return self.__label_type

    @label_type.setter
    def label_type(self, value: Optional[NavigationRailLabelType]):
        if False:
            for i in range(10):
                print('nop')
        self.__label_type = value
        if isinstance(value, NavigationRailLabelType):
            self._set_attr('labelType', value.value)
        else:
            self.__set_label_type(value)

    def __set_label_type(self, value: NavigationRailLabelTypeString):
        if False:
            i = 10
            return i + 15
        self._set_attr('labelType', value)

    @property
    def bgcolor(self):
        if False:
            return 10
        return self._get_attr('bgcolor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('bgcolor', value)

    @property
    def extended(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('extended', data_type='bool', def_value=False)

    @extended.setter
    def extended(self, value: Optional[bool]):
        if False:
            print('Hello World!')
        self._set_attr('extended', value)

    @property
    def leading(self) -> Optional[Control]:
        if False:
            return 10
        return self.__leading

    @leading.setter
    def leading(self, value: Optional[Control]):
        if False:
            while True:
                i = 10
        self.__leading = value

    @property
    def trailing(self) -> Optional[Control]:
        if False:
            for i in range(10):
                print('nop')
        return self.__trailing

    @trailing.setter
    def trailing(self, value: Optional[Control]):
        if False:
            print('Hello World!')
        self.__trailing = value

    @property
    def min_width(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('minWidth')

    @min_width.setter
    def min_width(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('minWidth', value)

    @property
    def min_extended_width(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('minExtendedWidth')

    @min_extended_width.setter
    def min_extended_width(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('minExtendedWidth', value)

    @property
    def group_alignment(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('groupAlignment')

    @group_alignment.setter
    def group_alignment(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('groupAlignment', value)