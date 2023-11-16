import pytest
from nicegui import Client, ui
from .screen import Screen

def test_run_javascript_on_button_press(screen: Screen):
    if False:
        i = 10
        return i + 15
    ui.button('change title', on_click=lambda : ui.run_javascript('document.title = "A New Title"'))
    screen.open('/')
    assert screen.selenium.title == 'NiceGUI'
    screen.click('change title')
    screen.wait(0.5)
    screen.should_contain('A New Title')

def test_run_javascript_on_value_change(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/')
    async def page(client: Client):
        ui.radio(['A', 'B'], on_change=lambda e: ui.run_javascript(f'document.title = "Page {e.value}"'))
        await client.connected()
        ui.run_javascript('document.title = "Initial Title"')
    screen.open('/')
    screen.wait(0.5)
    screen.should_contain('Initial Title')
    screen.click('A')
    screen.wait(0.5)
    screen.should_contain('Page A')
    screen.click('B')
    screen.wait(0.5)
    screen.should_contain('Page B')

def test_run_javascript_before_client_connected(screen: Screen):
    if False:
        while True:
            i = 10

    @ui.page('/')
    def page():
        if False:
            return 10
        ui.label('before js')
        with pytest.raises(RuntimeError):
            ui.run_javascript('document.title = "A New Title"')
        ui.label('after js')
    screen.open('/')
    assert screen.selenium.title == 'NiceGUI'
    screen.should_contain('before js')
    screen.should_contain('after js')

def test_response_from_javascript(screen: Screen):
    if False:
        print('Hello World!')

    async def compute() -> None:
        response = await ui.run_javascript('1 + 41')
        ui.label(response)
    ui.button('compute', on_click=compute)
    screen.open('/')
    screen.click('compute')
    screen.should_contain('42')

def test_async_javascript(screen: Screen):
    if False:
        while True:
            i = 10

    async def run():
        result = await ui.run_javascript('await new Promise(r => setTimeout(r, 100)); return 42')
        ui.label(result)
    ui.button('run', on_click=run)
    screen.open('/')
    screen.click('run')
    screen.should_contain('42')

def test_simultaneous_async_javascript(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    async def runA():
        result = await ui.run_javascript('await new Promise(r => setTimeout(r, 500)); return 1')
        ui.label(f'A: {result}')

    async def runB():
        result = await ui.run_javascript('await new Promise(r => setTimeout(r, 250)); return 2')
        ui.label(f'B: {result}')
    ui.button('runA', on_click=runA)
    ui.button('runB', on_click=runB)
    screen.open('/')
    screen.click('runA')
    screen.click('runB')
    screen.should_contain('A: 1')
    screen.should_contain('B: 2')