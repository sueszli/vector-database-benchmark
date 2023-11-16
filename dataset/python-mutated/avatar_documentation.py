from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    ui.avatar('favorite_border', text_color='grey-11', square=True)
    ui.avatar('img:https://nicegui.io/logo_square.png', color='blue-2')

def more() -> None:
    if False:
        return 10

    @text_demo('Photos', '\n        To use a photo as an avatar, you can use `ui.image` within `ui.avatar`.\n    ')
    def photos() -> None:
        if False:
            for i in range(10):
                print('nop')
        with ui.avatar():
            ui.image('https://robohash.org/robot?bgset=bg2')