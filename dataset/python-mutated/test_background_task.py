"""Test @rx.background task functionality."""
from typing import Generator
import pytest
from selenium.webdriver.common.by import By
from reflex.testing import DEFAULT_TIMEOUT, AppHarness, WebDriver

def BackgroundTask():
    if False:
        while True:
            i = 10
    'Test that background tasks work as expected.'
    import asyncio
    import reflex as rx

    class State(rx.State):
        counter: int = 0
        _task_id: int = 0
        iterations: int = 10

        @rx.background
        async def handle_event(self):
            async with self:
                self._task_id += 1
            for _ix in range(int(self.iterations)):
                async with self:
                    self.counter += 1
                await asyncio.sleep(0.005)

        @rx.background
        async def handle_event_yield_only(self):
            async with self:
                self._task_id += 1
            for ix in range(int(self.iterations)):
                if ix % 2 == 0:
                    yield State.increment_arbitrary(1)
                else:
                    yield State.increment()
                await asyncio.sleep(0.005)

        def increment(self):
            if False:
                while True:
                    i = 10
            self.counter += 1

        @rx.background
        async def increment_arbitrary(self, amount: int):
            async with self:
                self.counter += int(amount)

        def reset_counter(self):
            if False:
                for i in range(10):
                    print('nop')
            self.counter = 0

        async def blocking_pause(self):
            await asyncio.sleep(0.02)

        @rx.background
        async def non_blocking_pause(self):
            await asyncio.sleep(0.02)

    def index() -> rx.Component:
        if False:
            while True:
                i = 10
        return rx.vstack(rx.input(id='token', value=State.router.session.client_token, is_read_only=True), rx.heading(State.counter, id='counter'), rx.input(id='iterations', placeholder='Iterations', value=State.iterations.to_string(), on_change=State.set_iterations), rx.button('Delayed Increment', on_click=State.handle_event, id='delayed-increment'), rx.button('Yield Increment', on_click=State.handle_event_yield_only, id='yield-increment'), rx.button('Increment 1', on_click=State.increment, id='increment'), rx.button('Blocking Pause', on_click=State.blocking_pause, id='blocking-pause'), rx.button('Non-Blocking Pause', on_click=State.non_blocking_pause, id='non-blocking-pause'), rx.button('Reset', on_click=State.reset_counter, id='reset'))
    app = rx.App(state=State)
    app.add_page(index)
    app.compile()

@pytest.fixture(scope='session')
def background_task(tmp_path_factory) -> Generator[AppHarness, None, None]:
    if False:
        print('Hello World!')
    'Start BackgroundTask app at tmp_path via AppHarness.\n\n    Args:\n        tmp_path_factory: pytest tmp_path_factory fixture\n\n    Yields:\n        running AppHarness instance\n    '
    with AppHarness.create(root=tmp_path_factory.mktemp(f'background_task'), app_source=BackgroundTask) as harness:
        yield harness

@pytest.fixture
def driver(background_task: AppHarness) -> Generator[WebDriver, None, None]:
    if False:
        while True:
            i = 10
    'Get an instance of the browser open to the background_task app.\n\n    Args:\n        background_task: harness for BackgroundTask app\n\n    Yields:\n        WebDriver instance.\n    '
    assert background_task.app_instance is not None, 'app is not running'
    driver = background_task.frontend()
    try:
        yield driver
    finally:
        driver.quit()

@pytest.fixture()
def token(background_task: AppHarness, driver: WebDriver) -> str:
    if False:
        while True:
            i = 10
    'Get a function that returns the active token.\n\n    Args:\n        background_task: harness for BackgroundTask app.\n        driver: WebDriver instance.\n\n    Returns:\n        The token for the connected client\n    '
    assert background_task.app_instance is not None
    token_input = driver.find_element(By.ID, 'token')
    assert token_input
    token = background_task.poll_for_value(token_input, timeout=DEFAULT_TIMEOUT * 2)
    assert token is not None
    return token

def test_background_task(background_task: AppHarness, driver: WebDriver, token: str):
    if False:
        i = 10
        return i + 15
    'Test that background tasks work as expected.\n\n    Args:\n        background_task: harness for BackgroundTask app.\n        driver: WebDriver instance.\n        token: The token for the connected client.\n    '
    assert background_task.app_instance is not None
    delayed_increment_button = driver.find_element(By.ID, 'delayed-increment')
    yield_increment_button = driver.find_element(By.ID, 'yield-increment')
    increment_button = driver.find_element(By.ID, 'increment')
    blocking_pause_button = driver.find_element(By.ID, 'blocking-pause')
    non_blocking_pause_button = driver.find_element(By.ID, 'non-blocking-pause')
    driver.find_element(By.ID, 'reset')
    counter = driver.find_element(By.ID, 'counter')
    iterations_input = driver.find_element(By.ID, 'iterations')
    iterations_input.clear()
    iterations_input.send_keys('50')
    delayed_increment_button.click()
    blocking_pause_button.click()
    delayed_increment_button.click()
    for _ in range(10):
        increment_button.click()
    blocking_pause_button.click()
    delayed_increment_button.click()
    delayed_increment_button.click()
    yield_increment_button.click()
    non_blocking_pause_button.click()
    yield_increment_button.click()
    blocking_pause_button.click()
    yield_increment_button.click()
    for _ in range(10):
        increment_button.click()
    yield_increment_button.click()
    blocking_pause_button.click()
    assert background_task._poll_for(lambda : counter.text == '420', timeout=40)
    assert background_task._poll_for(lambda : not background_task.app_instance.background_tasks)