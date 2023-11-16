"""Integration tests for client side storage."""
from __future__ import annotations
from typing import Generator
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from reflex.testing import AppHarness

def CallScript():
    if False:
        i = 10
        return i + 15
    'A test app for browser javascript integration.'
    import reflex as rx
    inline_scripts = '\n    let inline_counter = 0\n    function inline1() {\n        inline_counter += 1\n        return "inline1"\n    }\n    function inline2() {\n        inline_counter += 1\n        console.log("inline2")\n    }\n    function inline3() {\n        inline_counter += 1\n        return {inline3: 42, a: [1, 2, 3], s: \'js\', o: {a: 1, b: 2}}\n    }\n    async function inline4() {\n        inline_counter += 1\n        return "async inline4"\n    }\n    '
    external_scripts = inline_scripts.replace('inline', 'external')

    class CallScriptState(rx.State):
        results: list[str | dict | list | None] = []
        inline_counter: int = 0
        external_counter: int = 0

        def call_script_callback(self, result):
            if False:
                i = 10
                return i + 15
            self.results.append(result)

        def call_script_callback_other_arg(self, result, other_arg):
            if False:
                i = 10
                return i + 15
            self.results.append([other_arg, result])

        def call_scripts_inline_yield(self):
            if False:
                i = 10
                return i + 15
            yield rx.call_script('inline1()')
            yield rx.call_script('inline2()')
            yield rx.call_script('inline3()')
            yield rx.call_script('inline4()')

        def call_script_inline_return(self):
            if False:
                i = 10
                return i + 15
            return rx.call_script('inline2()')

        def call_scripts_inline_yield_callback(self):
            if False:
                print('Hello World!')
            yield rx.call_script('inline1()', callback=CallScriptState.call_script_callback)
            yield rx.call_script('inline2()', callback=CallScriptState.call_script_callback)
            yield rx.call_script('inline3()', callback=CallScriptState.call_script_callback)
            yield rx.call_script('inline4()', callback=CallScriptState.call_script_callback)

        def call_script_inline_return_callback(self):
            if False:
                for i in range(10):
                    print('nop')
            return rx.call_script('inline3()', callback=CallScriptState.call_script_callback)

        def call_script_inline_return_lambda(self):
            if False:
                for i in range(10):
                    print('nop')
            return rx.call_script('inline2()', callback=lambda result: CallScriptState.call_script_callback_other_arg(result, 'lambda'))

        def get_inline_counter(self):
            if False:
                while True:
                    i = 10
            return rx.call_script('inline_counter', callback=CallScriptState.set_inline_counter)

        def call_scripts_external_yield(self):
            if False:
                print('Hello World!')
            yield rx.call_script('external1()')
            yield rx.call_script('external2()')
            yield rx.call_script('external3()')
            yield rx.call_script('external4()')

        def call_script_external_return(self):
            if False:
                return 10
            return rx.call_script('external2()')

        def call_scripts_external_yield_callback(self):
            if False:
                return 10
            yield rx.call_script('external1()', callback=CallScriptState.call_script_callback)
            yield rx.call_script('external2()', callback=CallScriptState.call_script_callback)
            yield rx.call_script('external3()', callback=CallScriptState.call_script_callback)
            yield rx.call_script('external4()', callback=CallScriptState.call_script_callback)

        def call_script_external_return_callback(self):
            if False:
                while True:
                    i = 10
            return rx.call_script('external3()', callback=CallScriptState.call_script_callback)

        def call_script_external_return_lambda(self):
            if False:
                while True:
                    i = 10
            return rx.call_script('external2()', callback=lambda result: CallScriptState.call_script_callback_other_arg(result, 'lambda'))

        def get_external_counter(self):
            if False:
                while True:
                    i = 10
            return rx.call_script('external_counter', callback=CallScriptState.set_external_counter)

        def reset_(self):
            if False:
                while True:
                    i = 10
            yield rx.call_script('inline_counter = 0; external_counter = 0')
            self.reset()
    app = rx.App(state=CallScriptState)
    with open('assets/external.js', 'w') as f:
        f.write(external_scripts)

    @app.add_page
    def index():
        if False:
            i = 10
            return i + 15
        return rx.vstack(rx.input(value=CallScriptState.router.session.client_token, is_read_only=True, id='token'), rx.input(value=CallScriptState.inline_counter.to(str), id='inline_counter', is_read_only=True), rx.input(value=CallScriptState.external_counter.to(str), id='external_counter', is_read_only=True), rx.text_area(value=CallScriptState.results.to_string(), id='results', is_read_only=True), rx.script(inline_scripts), rx.script(src='/external.js'), rx.button('call_scripts_inline_yield', on_click=CallScriptState.call_scripts_inline_yield, id='inline_yield'), rx.button('call_script_inline_return', on_click=CallScriptState.call_script_inline_return, id='inline_return'), rx.button('call_scripts_inline_yield_callback', on_click=CallScriptState.call_scripts_inline_yield_callback, id='inline_yield_callback'), rx.button('call_script_inline_return_callback', on_click=CallScriptState.call_script_inline_return_callback, id='inline_return_callback'), rx.button('call_script_inline_return_lambda', on_click=CallScriptState.call_script_inline_return_lambda, id='inline_return_lambda'), rx.button('call_scripts_external_yield', on_click=CallScriptState.call_scripts_external_yield, id='external_yield'), rx.button('call_script_external_return', on_click=CallScriptState.call_script_external_return, id='external_return'), rx.button('call_scripts_external_yield_callback', on_click=CallScriptState.call_scripts_external_yield_callback, id='external_yield_callback'), rx.button('call_script_external_return_callback', on_click=CallScriptState.call_script_external_return_callback, id='external_return_callback'), rx.button('call_script_external_return_lambda', on_click=CallScriptState.call_script_external_return_lambda, id='external_return_lambda'), rx.button('Update Inline Counter', on_click=CallScriptState.get_inline_counter, id='update_inline_counter'), rx.button('Update External Counter', on_click=CallScriptState.get_external_counter, id='update_external_counter'), rx.button('Reset', id='reset', on_click=CallScriptState.reset_))
    app.compile()

@pytest.fixture(scope='session')
def call_script(tmp_path_factory) -> Generator[AppHarness, None, None]:
    if False:
        i = 10
        return i + 15
    'Start CallScript app at tmp_path via AppHarness.\n\n    Args:\n        tmp_path_factory: pytest tmp_path_factory fixture\n\n    Yields:\n        running AppHarness instance\n    '
    with AppHarness.create(root=tmp_path_factory.mktemp('call_script'), app_source=CallScript) as harness:
        yield harness

@pytest.fixture
def driver(call_script: AppHarness) -> Generator[WebDriver, None, None]:
    if False:
        return 10
    'Get an instance of the browser open to the call_script app.\n\n    Args:\n        call_script: harness for CallScript app\n\n    Yields:\n        WebDriver instance.\n    '
    assert call_script.app_instance is not None, 'app is not running'
    driver = call_script.frontend()
    try:
        yield driver
    finally:
        driver.quit()

def assert_token(call_script: AppHarness, driver: WebDriver) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the token associated with backend state.\n\n    Args:\n        call_script: harness for CallScript app.\n        driver: WebDriver instance.\n\n    Returns:\n        The token visible in the driver browser.\n    '
    assert call_script.app_instance is not None
    token_input = driver.find_element(By.ID, 'token')
    assert token_input
    token = call_script.poll_for_value(token_input)
    assert token is not None
    return token

@pytest.mark.parametrize('script', ['inline', 'external'])
def test_call_script(call_script: AppHarness, driver: WebDriver, script: str):
    if False:
        while True:
            i = 10
    'Test calling javascript functions from python.\n\n    Args:\n        call_script: harness for CallScript app.\n        driver: WebDriver instance.\n        script: The type of script to test.\n    '
    assert_token(call_script, driver)
    reset_button = driver.find_element(By.ID, 'reset')
    update_counter_button = driver.find_element(By.ID, f'update_{script}_counter')
    counter = driver.find_element(By.ID, f'{script}_counter')
    results = driver.find_element(By.ID, 'results')
    yield_button = driver.find_element(By.ID, f'{script}_yield')
    return_button = driver.find_element(By.ID, f'{script}_return')
    yield_callback_button = driver.find_element(By.ID, f'{script}_yield_callback')
    return_callback_button = driver.find_element(By.ID, f'{script}_return_callback')
    return_lambda_button = driver.find_element(By.ID, f'{script}_return_lambda')
    yield_button.click()
    update_counter_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='0') == '4'
    reset_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='3') == '0'
    return_button.click()
    update_counter_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='0') == '1'
    reset_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='1') == '0'
    yield_callback_button.click()
    update_counter_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='0') == '4'
    assert call_script.poll_for_value(results, exp_not_equal='[]') == '["%s1",null,{"%s3":42,"a":[1,2,3],"s":"js","o":{"a":1,"b":2}},"async %s4"]' % (script, script, script)
    reset_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='3') == '0'
    return_callback_button.click()
    update_counter_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='0') == '1'
    assert call_script.poll_for_value(results, exp_not_equal='[]') == '[{"%s3":42,"a":[1,2,3],"s":"js","o":{"a":1,"b":2}}]' % script
    reset_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='1') == '0'
    return_lambda_button.click()
    update_counter_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='0') == '1'
    assert call_script.poll_for_value(results, exp_not_equal='[]') == '[["lambda",null]]'
    reset_button.click()
    assert call_script.poll_for_value(counter, exp_not_equal='1') == '0'