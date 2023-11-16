from nicegui import ui

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    ui.label('text above')
    ui.separator()
    ui.label('text below')