from nicegui import ui

def main_demo() -> None:
    if False:
        return 10
    from nicegui import app
    ui.button('shutdown', on_click=lambda : ui.notify('Nah. We do not actually shutdown the documentation server. Try it in your own app!'))