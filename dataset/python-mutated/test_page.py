import asyncio
from uuid import uuid4
from fastapi.responses import PlainTextResponse
from selenium.webdriver.common.by import By
from nicegui import Client, background_tasks, ui
from .screen import Screen

def test_page(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/')
    def page():
        if False:
            print('Hello World!')
        ui.label('Hello, world!')
    screen.open('/')
    screen.should_contain('NiceGUI')
    screen.should_contain('Hello, world!')

def test_auto_index_page(screen: Screen):
    if False:
        i = 10
        return i + 15
    ui.label('Hello, world!')
    screen.open('/')
    screen.should_contain('NiceGUI')
    screen.should_contain('Hello, world!')

def test_custom_title(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/', title='My Custom Title')
    def page():
        if False:
            i = 10
            return i + 15
        ui.label('Hello, world!')
    screen.open('/')
    screen.should_contain('My Custom Title')
    screen.should_contain('Hello, world!')

def test_route_with_custom_path(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/test_route')
    def page():
        if False:
            print('Hello World!')
        ui.label('page with custom path')
    screen.open('/test_route')
    screen.should_contain('page with custom path')

def test_auto_index_page_with_link_to_subpage(screen: Screen):
    if False:
        print('Hello World!')
    ui.link('link to subpage', '/subpage')

    @ui.page('/subpage')
    def page():
        if False:
            return 10
        ui.label('the subpage')
    screen.open('/')
    screen.click('link to subpage')
    screen.should_contain('the subpage')

def test_link_to_page_by_passing_function(screen: Screen):
    if False:
        return 10

    @ui.page('/subpage')
    def page():
        if False:
            return 10
        ui.label('the subpage')
    ui.link('link to subpage', page)
    screen.open('/')
    screen.click('link to subpage')
    screen.should_contain('the subpage')

def test_creating_new_page_after_startup(screen: Screen):
    if False:
        while True:
            i = 10
    screen.start_server()

    @ui.page('/late_page')
    def page():
        if False:
            i = 10
            return i + 15
        ui.label('page created after startup')
    screen.open('/late_page')
    screen.should_contain('page created after startup')

def test_shared_and_private_pages(screen: Screen):
    if False:
        i = 10
        return i + 15

    @ui.page('/private_page')
    def private_page():
        if False:
            return 10
        ui.label(f'private page with uuid {uuid4()}')
    ui.label(f'shared page with uuid {uuid4()}')
    screen.open('/private_page')
    uuid1 = screen.find('private page').text.split()[-1]
    screen.open('/private_page')
    uuid2 = screen.find('private page').text.split()[-1]
    assert uuid1 != uuid2
    screen.open('/')
    uuid1 = screen.find('shared page').text.split()[-1]
    screen.open('/')
    uuid2 = screen.find('shared page').text.split()[-1]
    assert uuid1 == uuid2

def test_wait_for_connected(screen: Screen):
    if False:
        print('Hello World!')
    label: ui.label

    async def load() -> None:
        label.text = 'loading...'
        background_tasks.create(takes_a_while())

    async def takes_a_while() -> None:
        await asyncio.sleep(0.1)
        label.text = 'delayed data has been loaded'

    @ui.page('/')
    async def page(client: Client):
        nonlocal label
        label = ui.label()
        await client.connected()
        await load()
    screen.open('/')
    screen.should_contain('delayed data has been loaded')

def test_wait_for_disconnect(screen: Screen):
    if False:
        i = 10
        return i + 15
    events = []

    @ui.page('/', reconnect_timeout=0)
    async def page(client: Client):
        await client.connected()
        events.append('connected')
        await client.disconnected()
        events.append('disconnected')
    screen.open('/')
    screen.wait(0.5)
    screen.open('/')
    screen.wait(0.5)
    assert events == ['connected', 'disconnected', 'connected']

def test_wait_for_disconnect_without_awaiting_connected(screen: Screen):
    if False:
        return 10
    events = []

    @ui.page('/', reconnect_timeout=0)
    async def page(client: Client):
        await client.disconnected()
        events.append('disconnected')
    screen.open('/')
    screen.wait(0.5)
    screen.open('/')
    screen.wait(0.5)
    assert events == ['disconnected']

def test_adding_elements_after_connected(screen: Screen):
    if False:
        i = 10
        return i + 15

    @ui.page('/')
    async def page(client: Client):
        ui.label('before')
        await client.connected()
        ui.label('after')
    screen.open('/')
    screen.should_contain('before')
    screen.should_contain('after')

def test_exception(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/')
    def page():
        if False:
            i = 10
            return i + 15
        raise RuntimeError('some exception')
    screen.open('/')
    screen.should_contain('500')
    screen.should_contain('Server error')
    screen.assert_py_logger('ERROR', 'some exception')

def test_exception_after_connected(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/')
    async def page(client: Client):
        await client.connected()
        ui.label('this is shown')
        raise RuntimeError('some exception')
    screen.open('/')
    screen.should_contain('this is shown')
    screen.assert_py_logger('ERROR', 'some exception')

def test_page_with_args(screen: Screen):
    if False:
        return 10

    @ui.page('/page/{id_}')
    def page(id_: int):
        if False:
            return 10
        ui.label(f'Page {id_}')
    screen.open('/page/42')
    screen.should_contain('Page 42')

def test_adding_elements_during_onconnect(screen: Screen):
    if False:
        for i in range(10):
            print('nop')

    @ui.page('/')
    def page(client: Client):
        if False:
            while True:
                i = 10
        ui.label('Label 1')
        client.on_connect(lambda : ui.label('Label 2'))
    screen.open('/')
    screen.should_contain('Label 2')

def test_async_connect_handler(screen: Screen):
    if False:
        return 10

    @ui.page('/')
    def page(client: Client):
        if False:
            for i in range(10):
                print('nop')

        async def run_js():
            result.text = await ui.run_javascript('41 + 1')
        result = ui.label()
        client.on_connect(run_js)
    screen.open('/')
    screen.should_contain('42')

def test_dark_mode(screen: Screen):
    if False:
        print('Hello World!')

    @ui.page('/auto', dark=None)
    def page():
        if False:
            for i in range(10):
                print('nop')
        ui.label('A').classes('text-blue-400 dark:text-red-400')

    @ui.page('/light', dark=False)
    def light_page():
        if False:
            for i in range(10):
                print('nop')
        ui.label('B').classes('text-blue-400 dark:text-red-400')

    @ui.page('/dark', dark=True)
    def dark_page():
        if False:
            return 10
        ui.label('C').classes('text-blue-400 dark:text-red-400')
    blue = 'rgba(96, 165, 250, 1)'
    red = 'rgba(248, 113, 113, 1)'
    white = 'rgba(0, 0, 0, 0)'
    black = 'rgba(18, 18, 18, 1)'
    screen.open('/auto')
    assert screen.find('A').value_of_css_property('color') == blue
    assert screen.find_by_tag('body').value_of_css_property('background-color') == white
    screen.open('/light')
    assert screen.find('B').value_of_css_property('color') == blue
    assert screen.find_by_tag('body').value_of_css_property('background-color') == white
    screen.open('/dark')
    assert screen.find('C').value_of_css_property('color') == red
    assert screen.find_by_tag('body').value_of_css_property('background-color') == black

def test_returning_custom_response(screen: Screen):
    if False:
        i = 10
        return i + 15

    @ui.page('/')
    def page(plain: bool=False):
        if False:
            return 10
        if plain:
            return PlainTextResponse('custom response')
        else:
            ui.label('normal NiceGUI page')
    screen.open('/')
    screen.should_contain('normal NiceGUI page')
    screen.should_not_contain('custom response')
    screen.open('/?plain=true')
    screen.should_contain('custom response')
    screen.should_not_contain('normal NiceGUI page')

def test_returning_custom_response_async(screen: Screen):
    if False:
        while True:
            i = 10

    @ui.page('/')
    async def page(plain: bool=False):
        await asyncio.sleep(0.01)
        if plain:
            return PlainTextResponse('custom response')
        else:
            ui.label('normal NiceGUI page')
    screen.open('/')
    screen.should_contain('normal NiceGUI page')
    screen.should_not_contain('custom response')
    screen.open('/?plain=true')
    screen.should_contain('custom response')
    screen.should_not_contain('normal NiceGUI page')

def test_reconnecting_without_page_reload(screen: Screen):
    if False:
        while True:
            i = 10

    @ui.page('/', reconnect_timeout=3.0)
    def page():
        if False:
            while True:
                i = 10
        ui.input('Input').props('autofocus')
        ui.button('drop connection', on_click=lambda : ui.run_javascript('socket.io.engine.close()'))
    screen.open('/')
    screen.type('hello')
    screen.click('drop connection')
    screen.wait(2.0)
    element = screen.selenium.find_element(By.XPATH, '//*[@aria-label="Input"]')
    assert element.get_attribute('value') == 'hello', 'input should be preserved after reconnect (i.e. no page reload)'