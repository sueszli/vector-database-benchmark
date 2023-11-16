"""Test case for displaying the connection banner when the websocket drops."""
from typing import Generator
import pytest
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from reflex.testing import AppHarness, WebDriver

def ConnectionBanner():
    if False:
        return 10
    'App with a connection banner.'
    import reflex as rx

    class State(rx.State):
        foo: int = 0

    def index():
        if False:
            for i in range(10):
                print('nop')
        return rx.text('Hello World')
    app = rx.App(state=State)
    app.add_page(index)
    app.compile()

@pytest.fixture()
def connection_banner(tmp_path) -> Generator[AppHarness, None, None]:
    if False:
        while True:
            i = 10
    'Start ConnectionBanner app at tmp_path via AppHarness.\n\n    Args:\n        tmp_path: pytest tmp_path fixture\n\n    Yields:\n        running AppHarness instance\n    '
    with AppHarness.create(root=tmp_path, app_source=ConnectionBanner) as harness:
        yield harness
CONNECTION_ERROR_XPATH = "//*[ text() = 'Connection Error' ]"

def has_error_modal(driver: WebDriver) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if the connection error modal is displayed.\n\n    Args:\n        driver: Selenium webdriver instance.\n\n    Returns:\n        True if the modal is displayed, False otherwise.\n    '
    try:
        driver.find_element(By.XPATH, CONNECTION_ERROR_XPATH)
        return True
    except NoSuchElementException:
        return False

def test_connection_banner(connection_banner: AppHarness):
    if False:
        print('Hello World!')
    'Test that the connection banner is displayed when the websocket drops.\n\n    Args:\n        connection_banner: AppHarness instance.\n    '
    assert connection_banner.app_instance is not None
    assert connection_banner.backend is not None
    driver = connection_banner.frontend()
    connection_banner._poll_for(lambda : not has_error_modal(driver))
    backend_port = connection_banner._poll_for_servers().getsockname()[1]
    connection_banner.backend.should_exit = True
    if connection_banner.backend_thread is not None:
        connection_banner.backend_thread.join()
    connection_banner._poll_for(lambda : has_error_modal(driver))
    connection_banner._start_backend(port=backend_port)
    connection_banner._poll_for(lambda : not has_error_modal(driver))