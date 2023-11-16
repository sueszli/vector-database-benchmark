from typing import Any, Optional, Union
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, OffsetValue, ResponsiveNumber, RotateValue, ScaleValue, PaddingValue
from flet_core.text_style import TextStyle
from flet_core.buttons import OutlinedBorder

class Chip(ConstrainedControl):
    """
    Chips are compact elements that represent an attribute, text, entity, or action.

    Example:
    ```
    import flet as ft


    def main(page: ft.Page):
        def save_to_favorites_clicked(e):
            e.control.label.value = "Saved to favorites"
            e.control.leading = ft.Icon(ft.icons.FAVORITE_OUTLINED)
            e.control.disabled = True
            page.update()

        def open_google_maps(e):
            page.launch_url("https://maps.google.com")
            page.update()

        save_to_favourites = ft.Chip(
            label=ft.Text("Save to favourites"),
            leading=ft.Icon(ft.icons.FAVORITE_BORDER_OUTLINED),
            bgcolor=ft.colors.GREEN_200,
            disabled_color=ft.colors.GREEN_100,
            autofocus=True,
            on_click=save_to_favorites_clicked,
        )

        open_in_maps = ft.Chip(
            label=ft.Text("9 min walk"),
            leading=ft.Icon(ft.icons.MAP_SHARP),
            bgcolor=ft.colors.GREEN_200,
            on_click=open_google_maps,
        )

        page.add(ft.Row([save_to_favourites, open_in_maps]))

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/chip
    """

    def __init__(self, label: Control, ref: Optional[Ref]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, tooltip: Optional[str]=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, key: Optional[str]=None, autofocus: Optional[bool]=None, leading: Optional[Control]=None, bgcolor: Optional[str]=None, selected: Optional[bool]=False, check_color: Optional[str]=None, delete_icon_tooltip: Optional[str]=None, delete_icon: Optional[Control]=None, delete_icon_color: Optional[str]=None, disabled_color: Optional[str]=None, elevation: OptionalNumber=None, label_padding: PaddingValue=None, label_style: Optional[TextStyle]=None, padding: PaddingValue=None, selected_color: Optional[str]=None, selected_shadow_color: Optional[str]=None, shadow_color: Optional[str]=None, shape: Optional[OutlinedBorder]=None, show_checkmark: Optional[bool]=None, on_click=None, on_delete=None, on_select=None, on_focus=None, on_blur=None):
        if False:
            while True:
                i = 10
        ConstrainedControl.__init__(self, ref=ref, key=key, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, tooltip=tooltip, visible=visible, disabled=disabled, data=data)
        self.autofocus = autofocus
        self.label = label
        self.leading = leading
        self.bgcolor = bgcolor
        self.check_color = check_color
        self.selected = selected
        self.delete_icon_tooltip = delete_icon_tooltip
        self.delete_icon = delete_icon
        self.delete_icon_color = delete_icon_color
        self.disabled_color = disabled_color
        self.elevation = elevation
        self.label_padding = label_padding
        self.label_style = label_style
        self.padding = padding
        self.selected_color = selected_color
        self.selected_shadow_color = selected_shadow_color
        self.shadow_color = shadow_color
        self.shape = shape
        self.show_checkmark = show_checkmark
        self.on_click = on_click
        self.on_delete = on_delete
        self.on_select = on_select
        self.on_focus = on_focus
        self.on_blur = on_blur

    def _get_control_name(self):
        if False:
            return 10
        return 'chip'

    def _before_build_command(self):
        if False:
            while True:
                i = 10
        super()._before_build_command()
        self._set_attr_json('labelPadding', self.__label_padding)
        self._set_attr_json('labelStyle', self.__label_style)
        self._set_attr_json('padding', self.__padding)
        self._set_attr_json('shape', self.__shape)

    def _get_children(self):
        if False:
            return 10
        children = []
        if self.__label:
            self.__label._set_attr_internal('n', 'label')
            children.append(self.__label)
        if self.__leading:
            self.__leading._set_attr_internal('n', 'leading')
            children.append(self.__leading)
        if self.__delete_icon:
            self.__delete_icon._set_attr_internal('n', 'deleteIcon')
            children.append(self.__delete_icon)
        return children

    @property
    def padding(self) -> PaddingValue:
        if False:
            while True:
                i = 10
        return self.__padding

    @padding.setter
    def padding(self, value: PaddingValue):
        if False:
            return 10
        self.__padding = value

    @property
    def selected(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self._get_attr('selected', data_type='bool', def_value=False)

    @selected.setter
    def selected(self, value: Optional[bool]):
        if False:
            while True:
                i = 10
        self._set_attr('selected', value)

    @property
    def show_checkmark(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('showCheckmark')

    @show_checkmark.setter
    def show_checkmark(self, value: Optional[bool]):
        if False:
            print('Hello World!')
        self._set_attr('showCheckmark', value)

    @property
    def delete_icon_tooltip(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('deleteButtonTooltipMessage')

    @delete_icon_tooltip.setter
    def delete_icon_tooltip(self, value):
        if False:
            return 10
        self._set_attr('deleteButtonTooltipMessage', value)

    @property
    def label(self) -> Control:
        if False:
            for i in range(10):
                print('nop')
        return self.__label

    @label.setter
    def label(self, value: Control):
        if False:
            print('Hello World!')
        self.__label = value

    @property
    def label_padding(self) -> PaddingValue:
        if False:
            return 10
        return self.__label_padding

    @label_padding.setter
    def label_padding(self, value: PaddingValue):
        if False:
            for i in range(10):
                print('nop')
        self.__label_padding = value

    @property
    def label_style(self):
        if False:
            while True:
                i = 10
        return self.__label_style

    @label_style.setter
    def label_style(self, value: Optional[TextStyle]):
        if False:
            return 10
        self.__label_style = value

    @property
    def leading(self) -> Optional[Control]:
        if False:
            for i in range(10):
                print('nop')
        return self.__leading

    @leading.setter
    def leading(self, value: Optional[Control]):
        if False:
            while True:
                i = 10
        self.__leading = value

    @property
    def delete_icon(self) -> Optional[Control]:
        if False:
            print('Hello World!')
        return self.__delete_icon

    @delete_icon.setter
    def delete_icon(self, value: Optional[Control]):
        if False:
            return 10
        self.__delete_icon = value

    @property
    def delete_icon_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('deleteIconColor')

    @delete_icon_color.setter
    def delete_icon_color(self, value):
        if False:
            print('Hello World!')
        self._set_attr('deleteIconColor', value)

    @property
    def disabled_color(self):
        if False:
            print('Hello World!')
        return self._get_attr('disabledColor')

    @disabled_color.setter
    def disabled_color(self, value):
        if False:
            return 10
        self._set_attr('disabledColor', value)

    @property
    def autofocus(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('autofocus', data_type='bool', def_value=False)

    @autofocus.setter
    def autofocus(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('autofocus', value)

    @property
    def bgcolor(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('bgcolor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('bgcolor', value)

    @property
    def check_color(self):
        if False:
            print('Hello World!')
        return self._get_attr('checkColor')

    @check_color.setter
    def check_color(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('checkColor', value)

    @property
    def selected_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('selectedColor')

    @selected_color.setter
    def selected_color(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('selectedColor', value)

    @property
    def selected_shadow_color(self):
        if False:
            return 10
        return self._get_attr('selectedShadowColor')

    @selected_shadow_color.setter
    def selected_shadow_color(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('selectedShadowColor', value)

    @property
    def shadow_color(self):
        if False:
            print('Hello World!')
        return self._get_attr('shadowColor')

    @shadow_color.setter
    def shadow_color(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('shadowColor', value)

    @property
    def elevation(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('elevation')

    @elevation.setter
    def elevation(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('elevation', value)

    @property
    def shape(self) -> Optional[OutlinedBorder]:
        if False:
            return 10
        return self.__shape

    @shape.setter
    def shape(self, value: Optional[OutlinedBorder]):
        if False:
            for i in range(10):
                print('nop')
        self.__shape = value

    @property
    def on_click(self):
        if False:
            return 10
        return self._get_event_handler('click')

    @on_click.setter
    def on_click(self, handler):
        if False:
            print('Hello World!')
        self._add_event_handler('click', handler)
        self._set_attr('onclick', True if handler is not None else None)

    @property
    def on_delete(self):
        if False:
            while True:
                i = 10
        return self._get_event_handler('delete')

    @on_delete.setter
    def on_delete(self, handler):
        if False:
            while True:
                i = 10
        self._add_event_handler('delete', handler)
        self._set_attr('onDelete', True if handler is not None else None)

    @property
    def on_select(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_event_handler('select')

    @on_select.setter
    def on_select(self, handler):
        if False:
            print('Hello World!')
        self._add_event_handler('select', handler)
        self._set_attr('onSelect', True if handler is not None else None)

    @property
    def on_focus(self):
        if False:
            return 10
        return self._get_event_handler('focus')

    @on_focus.setter
    def on_focus(self, handler):
        if False:
            while True:
                i = 10
        self._add_event_handler('focus', handler)

    @property
    def on_blur(self):
        if False:
            while True:
                i = 10
        return self._get_event_handler('blur')

    @on_blur.setter
    def on_blur(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self._add_event_handler('blur', handler)