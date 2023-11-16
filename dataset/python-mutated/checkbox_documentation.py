from nicegui import ui

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    checkbox = ui.checkbox('check me')
    ui.label('Check!').bind_visibility_from(checkbox, 'value')