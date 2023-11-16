from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    slider = ui.slider(min=0, max=100, value=50)
    ui.label().bind_text_from(slider, 'value')

def more() -> None:
    if False:
        return 10

    @text_demo('Throttle events with leading and trailing options', '\n        By default the value change event of a slider is throttled to 0.05 seconds.\n        This means that if you move the slider quickly, the value will only be updated every 0.05 seconds.\n\n        By default both "leading" and "trailing" events are activated.\n        This means that the very first event is triggered immediately, and the last event is triggered after the throttle time.\n\n        This demo shows how disabling either of these options changes the behavior.\n        To see the effect more clearly, the throttle time is set to 1 second.\n        The first slider shows the default behavior, the second one only sends leading events, and the third only sends trailing events.\n    ')
    def throttle_events_with_leading_and_trailing_options():
        if False:
            i = 10
            return i + 15
        ui.label('default')
        ui.slider(min=0, max=10, step=0.1, value=5).props('label-always').on('update:model-value', lambda e: ui.notify(e.args), throttle=1.0)
        ui.label('leading events only')
        ui.slider(min=0, max=10, step=0.1, value=5).props('label-always').on('update:model-value', lambda e: ui.notify(e.args), throttle=1.0, trailing_events=False)
        ui.label('trailing events only')
        ui.slider(min=0, max=10, step=0.1, value=5).props('label-always').on('update:model-value', lambda e: ui.notify(e.args), throttle=1.0, leading_events=False)

    @text_demo('Disable slider', '\n        You can disable a slider with the `disable()` method.\n        This will prevent the user from moving the slider.\n        The slider will also be grayed out.\n    ')
    def disable_slider():
        if False:
            for i in range(10):
                print('nop')
        slider = ui.slider(min=0, max=100, value=50)
        ui.button('Disable slider', on_click=slider.disable)
        ui.button('Enable slider', on_click=slider.enable)