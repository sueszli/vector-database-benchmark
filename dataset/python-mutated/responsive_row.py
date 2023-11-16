from typing import Any, List, Optional, Union
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, CrossAxisAlignment, CrossAxisAlignmentString, MainAxisAlignment, MainAxisAlignmentString, OffsetValue, ResponsiveNumber, RotateValue, ScaleValue

class ResponsiveRow(ConstrainedControl):
    """
    ResponsiveRow allows aligning child controls to virtual columns. By default, a virtual grid has 12 columns, but that can be customized with `ResponsiveRow.columns` property.

    Similar to `expand` property, every control now has `col` property which allows specifying how many columns a control should span.

    Example:

    ```
    import flet as ft

    def main(page: ft.Page):

        page.add(
            ft.ResponsiveRow(
                [
                    ft.TextField(label="TextField 1", col={"md": 4}),
                    ft.TextField(label="TextField 2", col={"md": 4}),
                    ft.TextField(label="TextField 3", col={"md": 4}),
                ],
                run_spacing={"xs": 10},
            ),
        )

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/responsiverow
    """

    def __init__(self, controls: Optional[List[Control]]=None, ref: Optional[Ref]=None, key: Optional[str]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, columns: Optional[ResponsiveNumber]=None, alignment: MainAxisAlignment=MainAxisAlignment.NONE, vertical_alignment: CrossAxisAlignment=CrossAxisAlignment.NONE, spacing: Optional[ResponsiveNumber]=None, run_spacing: Optional[ResponsiveNumber]=None):
        if False:
            while True:
                i = 10
        ConstrainedControl.__init__(self, ref=ref, key=key, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, visible=visible, disabled=disabled, data=data)
        self.controls = controls
        self.alignment = alignment
        self.vertical_alignment = vertical_alignment
        self.spacing = spacing
        self.run_spacing = run_spacing
        self.columns = columns

    def _get_control_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'responsiverow'

    def _before_build_command(self):
        if False:
            print('Hello World!')
        super()._before_build_command()
        self._set_attr_json('columns', self.__columns)
        self._set_attr_json('spacing', self.__spacing)
        self._set_attr_json('runSpacing', self.__run_spacing)

    def _get_children(self):
        if False:
            i = 10
            return i + 15
        return self.__controls

    def clean(self):
        if False:
            print('Hello World!')
        super().clean()
        self.__controls.clear()

    async def clean_async(self):
        await super().clean_async()
        self.__controls.clear()

    @property
    def alignment(self) -> MainAxisAlignment:
        if False:
            i = 10
            return i + 15
        return self.__alignment

    @alignment.setter
    def alignment(self, value: MainAxisAlignment):
        if False:
            while True:
                i = 10
        self.__alignment = value
        if isinstance(value, MainAxisAlignment):
            self._set_attr('alignment', value.value)
        else:
            self.__set_alignment(value)

    def __set_alignment(self, value: MainAxisAlignmentString):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('alignment', value)

    @property
    def vertical_alignment(self) -> CrossAxisAlignment:
        if False:
            for i in range(10):
                print('nop')
        return self.__vertical_alignment

    @vertical_alignment.setter
    def vertical_alignment(self, value: CrossAxisAlignment):
        if False:
            for i in range(10):
                print('nop')
        self.__vertical_alignment = value
        if isinstance(value, CrossAxisAlignment):
            self._set_attr('verticalAlignment', value.value)
        else:
            self.__set_vertical_alignment(value)

    def __set_vertical_alignment(self, value: CrossAxisAlignmentString):
        if False:
            return 10
        self._set_attr('verticalAlignment', value)

    @property
    def columns(self) -> Optional[ResponsiveNumber]:
        if False:
            while True:
                i = 10
        return self.__columns

    @columns.setter
    def columns(self, value: Optional[ResponsiveNumber]):
        if False:
            for i in range(10):
                print('nop')
        self.__columns = value

    @property
    def spacing(self) -> Optional[ResponsiveNumber]:
        if False:
            print('Hello World!')
        return self.__spacing

    @spacing.setter
    def spacing(self, value: Optional[ResponsiveNumber]):
        if False:
            while True:
                i = 10
        self.__spacing = value

    @property
    def run_spacing(self) -> Optional[ResponsiveNumber]:
        if False:
            while True:
                i = 10
        return self.__run_spacing

    @run_spacing.setter
    def run_spacing(self, value: Optional[ResponsiveNumber]):
        if False:
            for i in range(10):
                print('nop')
        self.__run_spacing = value

    @property
    def controls(self):
        if False:
            return 10
        return self.__controls

    @controls.setter
    def controls(self, value):
        if False:
            return 10
        self.__controls = value if value is not None else []