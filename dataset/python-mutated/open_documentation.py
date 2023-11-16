from nicegui import ui

def main_demo() -> None:
    if False:
        return 10

    @ui.page('/yet_another_page')
    def yet_another_page():
        if False:
            print('Hello World!')
        ui.label('Welcome to yet another page')
        ui.button('RETURN', on_click=lambda : ui.open('documentation#open'))
    ui.button('REDIRECT', on_click=lambda : ui.open(yet_another_page))