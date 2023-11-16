from typing import Any, Callable, Optional
from ..element import Element
from ..events import GenericEventArguments, JoystickEventArguments, handle_event

class Joystick(Element, component='joystick.vue', libraries=['lib/nipplejs/nipplejs.js']):

    def __init__(self, *, on_start: Optional[Callable[..., Any]]=None, on_move: Optional[Callable[..., Any]]=None, on_end: Optional[Callable[..., Any]]=None, throttle: float=0.05, **options: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Joystick\n\n        Create a joystick based on `nipple.js <https://yoannmoi.net/nipplejs/>`_.\n\n        :param on_start: callback for when the user touches the joystick\n        :param on_move: callback for when the user moves the joystick\n        :param on_end: callback for when the user releases the joystick\n        :param throttle: throttle interval in seconds for the move event (default: 0.05)\n        :param options: arguments like `color` which should be passed to the `underlying nipple.js library <https://github.com/yoannmoinet/nipplejs#options>`_\n        '
        super().__init__()
        self._props['options'] = options
        self.active = False

        def handle_start() -> None:
            if False:
                i = 10
                return i + 15
            self.active = True
            handle_event(on_start, JoystickEventArguments(sender=self, client=self.client, action='start'))

        def handle_move(e: GenericEventArguments) -> None:
            if False:
                i = 10
                return i + 15
            if self.active:
                handle_event(on_move, JoystickEventArguments(sender=self, client=self.client, action='move', x=float(e.args['data']['vector']['x']), y=float(e.args['data']['vector']['y'])))

        def handle_end() -> None:
            if False:
                i = 10
                return i + 15
            self.active = False
            handle_event(on_end, JoystickEventArguments(sender=self, client=self.client, action='end'))
        self.on('start', handle_start, [])
        self.on('move', handle_move, ['data'], throttle=throttle)
        self.on('end', handle_end, [])