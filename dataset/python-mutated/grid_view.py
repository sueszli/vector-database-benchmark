from typing import Any, List, Optional, Union
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.scrollable_control import ScrollableControl
from flet_core.types import AnimationValue, OffsetValue, PaddingValue, ResponsiveNumber, RotateValue, ScaleValue

class GridView(ConstrainedControl, ScrollableControl):
    """
    A scrollable, 2D array of controls.

    GridView is very effective for large lists (thousands of items). Prefer it over wrapping `Column` or `Row` for smooth scrolling.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        page.title = "GridView Example"
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 50
        page.update()

        images = ft.GridView(
            expand=1,
            runs_count=5,
            max_extent=150,
            child_aspect_ratio=1.0,
            spacing=5,
            run_spacing=5,
        )

        page.add(images)

        for i in range(0, 60):
            images.controls.append(
                ft.Image(
                    src=f"https://picsum.photos/150/150?{i}",
                    fit=ft.ImageFit.NONE,
                    repeat=ft.ImageRepeat.NO_REPEAT,
                    border_radius=ft.border_radius.all(10),
                )
            )
        page.update()

    ft.app(target=main)

    ```

    -----

    Online docs: https://flet.dev/docs/controls/gridview
    """

    def __init__(self, controls: Optional[List[Control]]=None, ref: Optional[Ref]=None, key: Optional[str]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, auto_scroll: Optional[bool]=None, on_scroll_interval: OptionalNumber=None, on_scroll: Any=None, horizontal: Optional[bool]=None, runs_count: Optional[int]=None, max_extent: Optional[int]=None, spacing: OptionalNumber=None, run_spacing: OptionalNumber=None, child_aspect_ratio: OptionalNumber=None, padding: PaddingValue=None):
        if False:
            print('Hello World!')
        ConstrainedControl.__init__(self, ref=ref, key=key, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, visible=visible, disabled=disabled, data=data)
        ScrollableControl.__init__(self, auto_scroll=auto_scroll, on_scroll_interval=on_scroll_interval, on_scroll=on_scroll)
        self.__controls: List[Control] = []
        self.controls = controls
        self.horizontal = horizontal
        self.runs_count = runs_count
        self.max_extent = max_extent
        self.spacing = spacing
        self.run_spacing = run_spacing
        self.child_aspect_ratio = child_aspect_ratio
        self.padding = padding

    def _get_control_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'gridview'

    def _before_build_command(self):
        if False:
            return 10
        super()._before_build_command()
        self._set_attr_json('padding', self.__padding)

    def _get_children(self):
        if False:
            return 10
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
    def horizontal(self) -> Optional[bool]:
        if False:
            print('Hello World!')
        return self._get_attr('horizontal')

    @horizontal.setter
    def horizontal(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('horizontal', value)

    @property
    def runs_count(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self._get_attr('runsCount')

    @runs_count.setter
    def runs_count(self, value: Optional[int]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('runsCount', value)

    @property
    def max_extent(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('maxExtent')

    @max_extent.setter
    def max_extent(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('maxExtent', value)

    @property
    def spacing(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('spacing')

    @spacing.setter
    def spacing(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('spacing', value)

    @property
    def run_spacing(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('runSpacing')

    @run_spacing.setter
    def run_spacing(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('runSpacing', value)

    @property
    def child_aspect_ratio(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('childAspectRatio')

    @child_aspect_ratio.setter
    def child_aspect_ratio(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('childAspectRatio', value)

    @property
    def padding(self) -> PaddingValue:
        if False:
            while True:
                i = 10
        return self.__padding

    @padding.setter
    def padding(self, value: PaddingValue):
        if False:
            while True:
                i = 10
        self.__padding = value

    @property
    def controls(self):
        if False:
            return 10
        return self.__controls

    @controls.setter
    def controls(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.__controls = value if value is not None else []