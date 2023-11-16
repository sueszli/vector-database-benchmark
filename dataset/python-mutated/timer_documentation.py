from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        print('Hello World!')
    from datetime import datetime
    label = ui.label()
    ui.timer(1.0, lambda : label.set_text(f'{datetime.now():%X}'))

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Activate, deactivate and cancel a timer', '\n        You can activate and deactivate a timer using the `active` property.\n        You can cancel a timer using the `cancel` method.\n        After canceling a timer, it cannot be activated anymore.\n    ')
    def activate_deactivate_demo():
        if False:
            for i in range(10):
                print('nop')
        slider = ui.slider(min=0, max=1, value=0.5)
        timer = ui.timer(0.1, lambda : slider.set_value((slider.value + 0.01) % 1.0))
        ui.switch('active').bind_value_to(timer, 'active')
        ui.button('Cancel', on_click=timer.cancel)

    @text_demo('Call a function after a delay', '\n        You can call a function after a delay using a timer with the `once` parameter.\n    ')
    def call_after_delay_demo():
        if False:
            for i in range(10):
                print('nop')

        def handle_click():
            if False:
                return 10
            ui.timer(1.0, lambda : ui.notify('Hi!'), once=True)
        ui.button('Notify after 1 second', on_click=handle_click)