from nicegui import ui

def main_demo() -> None:
    if False:
        while True:
            i = 10
    ui.button('NiceGUI Logo', on_click=lambda : ui.download('https://nicegui.io/logo.png'))