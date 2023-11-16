from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    with ui.column():
        ui.label('label 1')
        ui.label('label 2')
        ui.label('label 3')

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Masonry or Pinterest-Style Layout', '\n        To create a masonry/Pinterest layout, the normal `ui.column` can not be used.\n        But it can be achieved with a few TailwindCSS classes.\n    ')
    def masonry() -> None:
        if False:
            print('Hello World!')
        with ui.element('div').classes('columns-3 w-full gap-2'):
            for (i, height) in enumerate([50, 50, 50, 150, 100, 50]):
                tailwind = f'mb-2 p-2 h-[{height}px] bg-blue-100 break-inside-avoid'
                with ui.card().classes(tailwind):
                    ui.label(f'Card #{i + 1}')