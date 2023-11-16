from nicegui import context, ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        return 10

    def set_background(color: str) -> None:
        if False:
            while True:
                i = 10
        ui.query('body').style(f'background-color: {color}')
    ui.button('Blue', on_click=lambda e: e.sender.parent_slot.parent.style('background-color: #ddeeff'))
    ui.button('Orange', on_click=lambda e: e.sender.parent_slot.parent.style('background-color: #ffeedd'))

def more() -> None:
    if False:
        for i in range(10):
            print('nop')

    @text_demo('Set background gradient', "\n        It's easy to set a background gradient, image or similar. \n        See [w3schools.com](https://www.w3schools.com/cssref/pr_background-image.php) for more information about setting background with CSS.\n    ")
    def background_image():
        if False:
            while True:
                i = 10
        context.get_slot_stack()[-1].parent.classes('bg-gradient-to-t from-blue-400 to-blue-100')

    @text_demo('Modify default page padding', '\n        By default, NiceGUI provides a built-in padding around the content of the page.\n        You can modify it using the class selector `.nicegui-content`.\n    ')
    def remove_padding():
        if False:
            for i in range(10):
                print('nop')
        context.get_slot_stack()[-1].parent.classes(remove='p-4')
        with ui.column().classes('h-full w-full bg-gray-400 justify-between'):
            ui.label('top left')
            ui.label('bottom right').classes('self-end')