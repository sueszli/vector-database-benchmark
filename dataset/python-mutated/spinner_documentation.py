from nicegui import ui

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    with ui.row():
        ui.spinner(size='lg')
        ui.spinner('audio', size='lg', color='green')
        ui.spinner('dots', size='lg', color='red')