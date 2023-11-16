from typing import Any, Optional
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref

class ShakeDetector(Control):
    """
    Detects phone shakes.

    It is non-visual and should be added to `page.overlay` list.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        shd = ft.ShakeDetector(
            minimum_shake_count=2,
            shake_slop_time_ms=300,
            shake_count_reset_time_ms=1000,
            on_shake=lambda _: print("SHAKE DETECTED!"),
        )
        page.overlay.append(shd)

        page.add(ft.Text("Program body"))

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/shakedetector
    """

    def __init__(self, ref: Optional[Ref]=None, data: Any=None, minimum_shake_count: Optional[int]=None, shake_slop_time_ms: Optional[int]=None, shake_count_reset_time_ms: Optional[int]=None, shake_threshold_gravity: OptionalNumber=None, on_shake=None):
        if False:
            print('Hello World!')
        Control.__init__(self, ref=ref, data=data)
        self.minimum_shake_count = minimum_shake_count
        self.shake_slop_time_ms = shake_slop_time_ms
        self.shake_count_reset_time_ms = shake_count_reset_time_ms
        self.shake_threshold_gravity = shake_threshold_gravity
        self.on_shake = on_shake

    def _get_control_name(self):
        if False:
            print('Hello World!')
        return 'shakedetector'

    @property
    def minimum_shake_count(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self._get_attr('minimumShakeCount')

    @minimum_shake_count.setter
    def minimum_shake_count(self, value: Optional[int]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('minimumShakeCount', value)

    @property
    def shake_slop_time_ms(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        return self._get_attr('shakeSlopTimeMS')

    @shake_slop_time_ms.setter
    def shake_slop_time_ms(self, value: Optional[int]):
        if False:
            return 10
        self._set_attr('shakeSlopTimeMS', value)

    @property
    def shake_count_reset_time_ms(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        return self._get_attr('shakeCountResetTimeMs')

    @shake_count_reset_time_ms.setter
    def shake_count_reset_time_ms(self, value: Optional[int]):
        if False:
            while True:
                i = 10
        self._set_attr('shakeCountResetTimeMs', value)

    @property
    def shake_threshold_gravity(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('shakeThresholdGravity')

    @shake_threshold_gravity.setter
    def shake_threshold_gravity(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('shakeThresholdGravity', value)

    @property
    def on_shake(self):
        if False:
            return 10
        return self._get_event_handler('shake')

    @on_shake.setter
    def on_shake(self, handler):
        if False:
            return 10
        self._add_event_handler('shake', handler)