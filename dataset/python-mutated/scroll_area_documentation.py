from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        print('Hello World!')
    with ui.row():
        with ui.scroll_area().classes('w-32 h-32 border'):
            ui.label('I scroll. ' * 20)
        with ui.column().classes('p-4 w-32 h-32 border'):
            ui.label('I will not scroll. ' * 10)

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Handling Scroll Events', "\n        You can use the `on_scroll` argument in `ui.scroll_area` to handle scroll events.\n        The callback receives a `ScrollEventArguments` object with the following attributes:\n\n        - `sender`: the scroll area that generated the event\n        - `client`: the matching client\n        - additional arguments as described in [Quasar's documentation for the ScrollArea API](https://quasar.dev/vue-components/scroll-area/#qscrollarea-api)\n    ")
    def scroll_events():
        if False:
            for i in range(10):
                print('nop')
        position = ui.number('scroll position:').props('readonly')
        with ui.card().classes('w-32 h-32'):
            with ui.scroll_area(on_scroll=lambda e: position.set_value(e.vertical_percentage)):
                ui.label('I scroll. ' * 20)

    @text_demo('Setting the scroll position', '\n        You can use `scroll_to` to programmatically set the scroll position.\n        This can be useful for navigation or synchronization of multiple scroll areas.\n    ')
    def scroll_events():
        if False:
            print('Hello World!')
        ui.number('position', value=0, min=0, max=1, step=0.1, on_change=lambda e: area1.scroll_to(percent=e.value)).classes('w-32')
        with ui.row():
            with ui.card().classes('w-32 h-48'):
                with ui.scroll_area(on_scroll=lambda e: area2.scroll_to(percent=e.vertical_percentage)) as area1:
                    ui.label('I scroll. ' * 20)
            with ui.card().classes('w-32 h-48'):
                with ui.scroll_area() as area2:
                    ui.label('I scroll. ' * 20)