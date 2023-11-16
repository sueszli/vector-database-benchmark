"""Integration tests for special server side events."""
import time
from typing import Generator
import pytest
from selenium.webdriver.common.by import By
from reflex.testing import AppHarness

def ServerSideEvent():
    if False:
        i = 10
        return i + 15
    'App with inputs set via event handlers and set_value.'
    import reflex as rx

    class SSState(rx.State):

        def set_value_yield(self):
            if False:
                while True:
                    i = 10
            yield rx.set_value('a', '')
            yield rx.set_value('b', '')
            yield rx.set_value('c', '')

        def set_value_yield_return(self):
            if False:
                for i in range(10):
                    print('nop')
            yield rx.set_value('a', '')
            yield rx.set_value('b', '')
            return rx.set_value('c', '')

        def set_value_return(self):
            if False:
                print('Hello World!')
            return [rx.set_value('a', ''), rx.set_value('b', ''), rx.set_value('c', '')]

        def set_value_return_c(self):
            if False:
                for i in range(10):
                    print('nop')
            return rx.set_value('c', '')
    app = rx.App(state=SSState)

    @app.add_page
    def index():
        if False:
            return 10
        return rx.fragment(rx.input(id='token', value=SSState.router.session.client_token, is_read_only=True), rx.input(default_value='a', id='a'), rx.input(default_value='b', id='b'), rx.input(default_value='c', id='c'), rx.button('Clear Immediate', id='clear_immediate', on_click=[rx.set_value('a', ''), rx.set_value('b', ''), rx.set_value('c', '')]), rx.button('Clear Chained Yield', id='clear_chained_yield', on_click=SSState.set_value_yield), rx.button('Clear Chained Yield+Return', id='clear_chained_yield_return', on_click=SSState.set_value_yield_return), rx.button('Clear Chained Return', id='clear_chained_return', on_click=SSState.set_value_return), rx.button('Clear C Return', id='clear_return_c', on_click=SSState.set_value_return_c))
    app.compile()

@pytest.fixture(scope='session')
def server_side_event(tmp_path_factory) -> Generator[AppHarness, None, None]:
    if False:
        i = 10
        return i + 15
    'Start ServerSideEvent app at tmp_path via AppHarness.\n\n    Args:\n        tmp_path_factory: pytest tmp_path_factory fixture\n\n    Yields:\n        running AppHarness instance\n    '
    with AppHarness.create(root=tmp_path_factory.mktemp('server_side_event'), app_source=ServerSideEvent) as harness:
        yield harness

@pytest.fixture
def driver(server_side_event: AppHarness):
    if False:
        while True:
            i = 10
    'Get an instance of the browser open to the server_side_event app.\n\n\n    Args:\n        server_side_event: harness for ServerSideEvent app\n\n    Yields:\n        WebDriver instance.\n    '
    assert server_side_event.app_instance is not None, 'app is not running'
    driver = server_side_event.frontend()
    try:
        token_input = driver.find_element(By.ID, 'token')
        assert token_input
        token = server_side_event.poll_for_value(token_input)
        assert token is not None
        yield driver
    finally:
        driver.quit()

@pytest.mark.parametrize('button_id', ['clear_immediate', 'clear_chained_yield', 'clear_chained_yield_return', 'clear_chained_return'])
def test_set_value(driver, button_id: str):
    if False:
        for i in range(10):
            print('nop')
    'Call set_value as an event chain, via yielding, via yielding with return.\n\n    Args:\n        driver: selenium WebDriver open to the app\n        button_id: id of the button to click (parametrized)\n    '
    input_a = driver.find_element(By.ID, 'a')
    input_b = driver.find_element(By.ID, 'b')
    input_c = driver.find_element(By.ID, 'c')
    btn = driver.find_element(By.ID, button_id)
    assert input_a
    assert input_b
    assert input_c
    assert btn
    assert input_a.get_attribute('value') == 'a'
    assert input_b.get_attribute('value') == 'b'
    assert input_c.get_attribute('value') == 'c'
    btn.click()
    time.sleep(0.2)
    assert input_a.get_attribute('value') == ''
    assert input_b.get_attribute('value') == ''
    assert input_c.get_attribute('value') == ''

def test_set_value_return_c(driver):
    if False:
        i = 10
        return i + 15
    'Call set_value returning single event.\n\n    Args:\n        driver: selenium WebDriver open to the app\n    '
    input_a = driver.find_element(By.ID, 'a')
    input_b = driver.find_element(By.ID, 'b')
    input_c = driver.find_element(By.ID, 'c')
    btn = driver.find_element(By.ID, 'clear_return_c')
    assert input_a
    assert input_b
    assert input_c
    assert btn
    assert input_a.get_attribute('value') == 'a'
    assert input_b.get_attribute('value') == 'b'
    assert input_c.get_attribute('value') == 'c'
    btn.click()
    time.sleep(0.2)
    assert input_a.get_attribute('value') == 'a'
    assert input_b.get_attribute('value') == 'b'
    assert input_c.get_attribute('value') == ''