from nicegui import ui

def main_demo() -> None:
    if False:
        print('Hello World!')
    with ui.grid(columns=2):
        ui.label('Name:')
        ui.label('Tom')
        ui.label('Age:')
        ui.label('42')
        ui.label('Height:')
        ui.label('1.80m')