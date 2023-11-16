from nicegui import APIRouter, app, ui
from .screen import Screen

def test_prefix(screen: Screen):
    if False:
        while True:
            i = 10
    router = APIRouter(prefix='/some-prefix')

    @router.page('/')
    def page():
        if False:
            i = 10
            return i + 15
        ui.label('Hello, world!')
    app.include_router(router)
    screen.open('/some-prefix')
    screen.should_contain('NiceGUI')
    screen.should_contain('Hello, world!')

def test_passing_page_parameters(screen: Screen):
    if False:
        print('Hello World!')
    router = APIRouter()

    @router.page('/', title='My Custom Title')
    def page():
        if False:
            for i in range(10):
                print('nop')
        ui.label('Hello, world!')
    app.include_router(router)
    screen.open('/')
    screen.should_contain('My Custom Title')