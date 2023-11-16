import time
from typing import List
from selenium.common.exceptions import NoAlertPresentException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

class NavigationHelper:
    _TIMEOUT = 10
    _POLL_FREQUENCY = 0.1

    def __init__(self, web_driver: WebDriver) -> None:
        if False:
            while True:
                i = 10
        self.driver = web_driver

    def wait_for(self, function_with_assertion, timeout=_TIMEOUT):
        if False:
            for i in range(10):
                print('nop')
        'Polling wait for an arbitrary assertion.'
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                return function_with_assertion()
            except (AssertionError, WebDriverException):
                time.sleep(self._POLL_FREQUENCY)
        return function_with_assertion()

    def safe_click_by_id(self, element_id: str) -> WebElement:
        if False:
            print('Hello World!')
        '\n        Clicks the element with the given ID attribute.\n\n        Returns:\n            el: The element, if found.\n\n        Raises:\n            selenium.common.exceptions.TimeoutException: If the element cannot be found in time.\n\n        '
        el = WebDriverWait(self.driver, self._TIMEOUT, self._POLL_FREQUENCY).until(expected_conditions.element_to_be_clickable((By.ID, element_id)))
        el.location_once_scrolled_into_view
        el.click()
        return el

    def safe_click_by_css_selector(self, selector: str) -> WebElement:
        if False:
            while True:
                i = 10
        '\n        Clicks the first element with the given CSS selector.\n\n        Returns:\n            el: The element, if found.\n\n        Raises:\n            selenium.common.exceptions.TimeoutException: If the element cannot be found in time.\n\n        '
        el = WebDriverWait(self.driver, self._TIMEOUT, self._POLL_FREQUENCY).until(expected_conditions.element_to_be_clickable((By.CSS_SELECTOR, selector)))
        el.click()
        return el

    def safe_click_all_by_css_selector(self, selector: str) -> List[WebElement]:
        if False:
            while True:
                i = 10
        '\n        Clicks each element that matches the given CSS selector.\n\n        Returns:\n            els (list): The list of elements that matched the selector.\n\n        Raises:\n            selenium.common.exceptions.TimeoutException: If the element cannot be found in time.\n\n        '
        els = self.wait_for(lambda : self.driver.find_elements_by_css_selector(selector))
        for el in els:
            clickable_el = WebDriverWait(self.driver, self._TIMEOUT, self._POLL_FREQUENCY).until(expected_conditions.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            clickable_el.click()
        return els

    def safe_send_keys_by_id(self, element_id: str, text: str) -> WebElement:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends the given text to the element with the specified ID.\n\n        Returns:\n            el: The element, if found.\n\n        Raises:\n            selenium.common.exceptions.TimeoutException: If the element cannot be found in time.\n\n        '
        el = WebDriverWait(self.driver, self._TIMEOUT, self._POLL_FREQUENCY).until(expected_conditions.element_to_be_clickable((By.ID, element_id)))
        el.send_keys(text)
        return el

    def safe_send_keys_by_css_selector(self, selector: str, text: str) -> WebElement:
        if False:
            return 10
        '\n        Sends the given text to the first element with the given CSS selector.\n\n        Returns:\n            el: The element, if found.\n\n        Raises:\n            selenium.common.exceptions.TimeoutException: If the element cannot be found in time.\n\n        '
        el = WebDriverWait(self.driver, self._TIMEOUT, self._POLL_FREQUENCY).until(expected_conditions.element_to_be_clickable((By.CSS_SELECTOR, selector)))
        el.send_keys(text)
        return el

    def alert_wait(self, timeout: int=_TIMEOUT * 10) -> None:
        if False:
            print('Hello World!')
        WebDriverWait(self.driver, timeout, self._POLL_FREQUENCY).until(expected_conditions.alert_is_present(), 'Timed out waiting for confirmation popup.')

    def alert_accept(self) -> None:
        if False:
            while True:
                i = 10

        def alert_is_not_present(object):
            if False:
                print('Hello World!')
            'Expect an alert to not be present.'
            try:
                alert = self.driver.switch_to.alert
                alert.text
                return False
            except NoAlertPresentException:
                return True
        self.driver.switch_to.alert.accept()
        WebDriverWait(self.driver, self._TIMEOUT, self._POLL_FREQUENCY).until(alert_is_not_present, 'Timed out waiting for confirmation popup to disappear.')