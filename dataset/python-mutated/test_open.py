import pytest
from nicegui import ui
from .screen import Screen

@pytest.mark.parametrize('new_tab', [False, True])
def test_open_page(screen: Screen, new_tab: bool):
    if False:
        print('Hello World!')

    @ui.page('/test_page')
    def page():
        if False:
            return 10
        ui.label('Test page')
    ui.button('Open test page', on_click=lambda : ui.open('/test_page', new_tab=new_tab))
    screen.open('/')
    screen.click('Open test page')
    screen.wait(0.5)
    screen.switch_to(1 if new_tab else 0)
    screen.should_contain('Test page')