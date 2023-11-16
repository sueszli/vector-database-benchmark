from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    slider = ui.slider(min=0, max=1, step=0.01, value=0.5)
    ui.circular_progress().bind_value_from(slider, 'value')

def more() -> None:
    if False:
        return 10

    @text_demo('Nested Elements', '\n        You can put any element like icon, button etc inside a circular progress using the `with` statement.\n        Just make sure it fits the bounds and disable the default behavior of showing the value.\n    ')
    def icon() -> None:
        if False:
            for i in range(10):
                print('nop')
        with ui.row().classes('items-center m-auto'):
            with ui.circular_progress(value=0.1, show_value=False) as progress:
                ui.button(icon='star', on_click=lambda : progress.set_value(progress.value + 0.1)).props('flat round')
            ui.label('click to increase progress')