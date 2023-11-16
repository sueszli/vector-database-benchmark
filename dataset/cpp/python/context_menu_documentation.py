from nicegui import ui


def main_demo() -> None:
    with ui.image('https://picsum.photos/id/377/640/360'):
        with ui.context_menu():
            ui.menu_item('Flip horizontally')
            ui.menu_item('Flip vertically')
            ui.separator()
            ui.menu_item('Reset')
