"""This module contains useful methods for waiting on elements to load.

These methods improve and expand on existing WebDriver commands.
Improvements include making WebDriver commands more robust and more reliable
by giving page elements enough time to load before taking action on them.

The default option for searching for elements is by "css selector".
This can be changed by overriding the "By" parameter from this import:
> from selenium.webdriver.common.by import By
Options are:
By.CSS_SELECTOR        # "css selector"
By.CLASS_NAME          # "class name"
By.ID                  # "id"
By.NAME                # "name"
By.LINK_TEXT           # "link text"
By.XPATH               # "xpath"
By.TAG_NAME            # "tag name"
By.PARTIAL_LINK_TEXT   # "partial link text"
"""
import codecs
import os
import time
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import NoAlertPresentException
from selenium.common.exceptions import NoSuchAttributeException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchWindowException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from seleniumbase.common.exceptions import LinkTextNotFoundException
from seleniumbase.common.exceptions import TextNotVisibleException
from seleniumbase.config import settings
from seleniumbase.fixtures import page_utils
from seleniumbase.fixtures import shared_utils

def is_element_present(driver, selector, by='css selector'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns whether the specified element selector is present on the page.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is element present)\n    '
    (selector, by) = page_utils.swap_selector_and_by_if_reversed(selector, by)
    try:
        driver.find_element(by=by, value=selector)
        return True
    except Exception:
        return False

def is_element_visible(driver, selector, by='css selector'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns whether the specified element selector is visible on the page.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is element visible)\n    '
    (selector, by) = page_utils.swap_selector_and_by_if_reversed(selector, by)
    try:
        element = driver.find_element(by=by, value=selector)
        return element.is_displayed()
    except Exception:
        return False

def is_element_clickable(driver, selector, by='css selector'):
    if False:
        print('Hello World!')
    '\n    Returns whether the specified element selector is clickable.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is element clickable)\n    '
    try:
        element = driver.find_element(by=by, value=selector)
        if element.is_displayed() and element.is_enabled():
            return True
        return False
    except Exception:
        return False

def is_element_enabled(driver, selector, by='css selector'):
    if False:
        while True:
            i = 10
    '\n    Returns whether the specified element selector is enabled on the page.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is element enabled)\n    '
    try:
        element = driver.find_element(by=by, value=selector)
        return element.is_enabled()
    except Exception:
        return False

def is_text_visible(driver, text, selector='html', by='css selector', browser=None):
    if False:
        print('Hello World!')
    '\n    Returns whether the text substring is visible in the given selector.\n    @Params\n    driver - the webdriver object (required)\n    text - the text string to search for (required)\n    selector - the locator for identifying the page element\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is text visible)\n    '
    (selector, by) = page_utils.swap_selector_and_by_if_reversed(selector, by)
    text = str(text)
    try:
        element = driver.find_element(by=by, value=selector)
        element_text = element.text
        if browser == 'safari':
            if element.tag_name.lower() in ['input', 'textarea']:
                element_text = element.get_attribute('value')
            else:
                element_text = element.get_attribute('innerText')
        elif element.tag_name.lower() in ['input', 'textarea']:
            element_text = element.get_property('value')
        return element.is_displayed() and text in element_text
    except Exception:
        return False

def is_exact_text_visible(driver, text, selector, by='css selector', browser=None):
    if False:
        print('Hello World!')
    '\n    Returns whether the exact text is visible in the given selector.\n    (Ignores leading and trailing whitespace)\n    @Params\n    driver - the webdriver object (required)\n    text - the text string to search for (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is text visible)\n    '
    (selector, by) = page_utils.swap_selector_and_by_if_reversed(selector, by)
    text = str(text)
    try:
        element = driver.find_element(by=by, value=selector)
        element_text = element.text
        if browser == 'safari':
            if element.tag_name.lower() in ['input', 'textarea']:
                element_text = element.get_attribute('value')
            else:
                element_text = element.get_attribute('innerText')
        elif element.tag_name.lower() in ['input', 'textarea']:
            element_text = element.get_property('value')
        return element.is_displayed() and text.strip() == element_text.strip()
    except Exception:
        return False

def is_attribute_present(driver, selector, attribute, value=None, by='css selector'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns whether the specified attribute is present in the given selector.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    attribute - the attribute that is expected for the element (required)\n    value - the attribute value that is expected (Default: None)\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is attribute present)\n    '
    try:
        element = driver.find_element(by=by, value=selector)
        found_value = element.get_attribute(attribute)
        if found_value is None:
            return False
        if value is not None:
            if found_value == value:
                return True
            else:
                return False
        else:
            return True
    except Exception:
        return False

def is_non_empty_text_visible(driver, selector, by='css selector'):
    if False:
        return 10
    '\n    Returns whether the element has any text visible for the given selector.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    @Returns\n    Boolean (is any text visible in the element with the selector)\n    '
    browser = None
    try:
        if 'safari:platformVersion' in driver.capabilities:
            browser = 'safari'
    except Exception:
        pass
    try:
        element = driver.find_element(by=by, value=selector)
        element_text = element.text
        if browser == 'safari':
            if element.tag_name.lower() in ['input', 'textarea']:
                element_text = element.get_attribute('value')
            else:
                element_text = element.get_attribute('innerText')
        elif element.tag_name.lower() in ['input', 'textarea']:
            element_text = element.get_property('value')
        element_text = element_text.strip()
        return element.is_displayed() and len(element_text) > 0
    except Exception:
        return False

def hover_on_element(driver, selector, by='css selector'):
    if False:
        i = 10
        return i + 15
    '\n    Fires the hover event for the specified element by the given selector.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    '
    element = driver.find_element(by=by, value=selector)
    hover = ActionChains(driver).move_to_element(element)
    hover.perform()
    return element

def hover_element(driver, element):
    if False:
        while True:
            i = 10
    '\n    Similar to hover_on_element(), but uses found element, not a selector.\n    '
    hover = ActionChains(driver).move_to_element(element)
    hover.perform()
    return element

def timeout_exception(exception, message):
    if False:
        while True:
            i = 10
    (exc, msg) = shared_utils.format_exc(exception, message)
    raise exc(msg)

def hover_and_click(driver, hover_selector, click_selector, hover_by='css selector', click_by='css selector', timeout=settings.SMALL_TIMEOUT, js_click=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fires the hover event for a specified element by a given selector, then\n    clicks on another element specified. Useful for dropdown hover based menus.\n    @Params\n    driver - the webdriver object (required)\n    hover_selector - the css selector to hover over (required)\n    click_selector - the css selector to click on (required)\n    hover_by - the hover selector type to search by (Default: "css selector")\n    click_by - the click selector type to search by (Default: "css selector")\n    timeout - number of seconds to wait for click element to appear after hover\n    js_click - the option to use js_click() instead of click() on the last part\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    element = driver.find_element(by=hover_by, value=hover_selector)
    hover = ActionChains(driver).move_to_element(element)
    for x in range(int(timeout * 10)):
        try:
            hover.perform()
            element = driver.find_element(by=click_by, value=click_selector)
            if js_click:
                driver.execute_script('arguments[0].click();', element)
            else:
                element.click()
            return element
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    message = 'Element {%s} was not present after %s second%s!' % (click_selector, timeout, plural)
    timeout_exception(NoSuchElementException, message)

def hover_element_and_click(driver, element, click_selector, click_by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        print('Hello World!')
    '\n    Similar to hover_and_click(), but assumes top element is already found.\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    hover = ActionChains(driver).move_to_element(element)
    for x in range(int(timeout * 10)):
        try:
            hover.perform()
            element = driver.find_element(by=click_by, value=click_selector)
            element.click()
            return element
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    message = 'Element {%s} was not present after %s second%s!' % (click_selector, timeout, plural)
    timeout_exception(NoSuchElementException, message)

def hover_element_and_double_click(driver, element, click_selector, click_by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    hover = ActionChains(driver).move_to_element(element)
    for x in range(int(timeout * 10)):
        try:
            hover.perform()
            element_2 = driver.find_element(by=click_by, value=click_selector)
            actions = ActionChains(driver)
            actions.move_to_element(element_2)
            actions.double_click(element_2)
            actions.perform()
            return element_2
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    message = 'Element {%s} was not present after %s second%s!' % (click_selector, timeout, plural)
    timeout_exception(NoSuchElementException, message)

def wait_for_element_present(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, original_selector=None, ignore_test_time_limit=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Searches for the specified element by the given selector. Returns the\n    element object if it exists in the HTML. (The element can be invisible.)\n    Raises NoSuchElementException if the element does not exist in the HTML\n    within the specified timeout.\n    @Params\n    driver - the webdriver object\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    original_selector - handle pre-converted ":contains(TEXT)" selector\n    ignore_test_time_limit - ignore test time limit (NOT related to timeout)\n    @Returns\n    A web element object\n    '
    element = None
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        if not ignore_test_time_limit:
            shared_utils.check_if_time_limit_exceeded()
        try:
            element = driver.find_element(by=by, value=selector)
            return element
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    if not element:
        if original_selector and ':contains(' in original_selector and ('contains(.' in selector):
            selector = original_selector
        message = 'Element {%s} was not present after %s second%s!' % (selector, timeout, plural)
        timeout_exception(NoSuchElementException, message)
    else:
        return element

def wait_for_element_visible(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, original_selector=None, ignore_test_time_limit=False):
    if False:
        print('Hello World!')
    '\n    Searches for the specified element by the given selector. Returns the\n    element object if the element is present and visible on the page.\n    Raises NoSuchElementException if the element does not exist in the HTML\n    within the specified timeout.\n    Raises ElementNotVisibleException if the element exists in the HTML,\n    but is not visible (eg. opacity is "0") within the specified timeout.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    original_selector - handle pre-converted ":contains(TEXT)" selector\n    ignore_test_time_limit - ignore test time limit (NOT related to timeout)\n    @Returns\n    A web element object\n    '
    element = None
    is_present = False
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        if not ignore_test_time_limit:
            shared_utils.check_if_time_limit_exceeded()
        try:
            element = driver.find_element(by=by, value=selector)
            is_present = True
            if element.is_displayed():
                return element
            else:
                element = None
                raise Exception()
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    if not element and by != 'link text':
        if original_selector and ':contains(' in original_selector and ('contains(.' in selector):
            selector = original_selector
        if not is_present:
            message = 'Element {%s} was not present after %s second%s!' % (selector, timeout, plural)
            timeout_exception(NoSuchElementException, message)
        message = 'Element {%s} was not visible after %s second%s!' % (selector, timeout, plural)
        timeout_exception(ElementNotVisibleException, message)
    elif not element and by == 'link text':
        message = 'Link text {%s} was not found after %s second%s!' % (selector, timeout, plural)
        timeout_exception(LinkTextNotFoundException, message)
    else:
        return element

def wait_for_text_visible(driver, text, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, browser=None):
    if False:
        while True:
            i = 10
    '\n    Searches for the specified element by the given selector. Returns the\n    element object if the text is present in the element and visible\n    on the page.\n    Raises NoSuchElementException if the element does not exist in the HTML\n    within the specified timeout.\n    Raises ElementNotVisibleException if the element exists in the HTML,\n    but the text is not visible within the specified timeout.\n    @Params\n    driver - the webdriver object (required)\n    text - the text that is being searched for in the element (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    browser - used to handle a special edge case when using Safari\n    @Returns\n    A web element object that contains the text searched for\n    '
    element = None
    is_present = False
    full_text = None
    text = str(text)
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        full_text = None
        try:
            element = driver.find_element(by=by, value=selector)
            is_present = True
            if element.tag_name.lower() in ['input', 'textarea'] and browser != 'safari':
                if element.is_displayed() and text in element.get_property('value'):
                    return element
                else:
                    if element.is_displayed():
                        full_text = element.get_property('value').strip()
                    element = None
                    raise Exception()
            elif browser == 'safari':
                text_attr = 'innerText'
                if element.tag_name.lower() in ['input', 'textarea']:
                    text_attr = 'value'
                if element.is_displayed() and text in element.get_attribute(text_attr):
                    return element
                else:
                    if element.is_displayed():
                        full_text = element.get_attribute(text_attr)
                        full_text = full_text.strip()
                    element = None
                    raise Exception()
            elif element.is_displayed() and text in element.text:
                return element
            else:
                if element.is_displayed():
                    full_text = element.text.strip()
                element = None
                raise Exception()
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    if not element:
        if not is_present:
            message = 'Element {%s} was not present after %s second%s!' % (selector, timeout, plural)
            timeout_exception(NoSuchElementException, message)
        message = None
        if not full_text or len(str(full_text.replace('\n', ''))) > 320:
            message = 'Expected text substring {%s} for {%s} was not visible after %s second%s!' % (text, selector, timeout, plural)
        else:
            full_text = full_text.replace('\n', '\\n ')
            message = 'Expected text substring {%s} for {%s} was not visible after %s second%s!\n (Actual string found was {%s})' % (text, selector, timeout, plural, full_text)
        timeout_exception(TextNotVisibleException, message)
    else:
        return element

def wait_for_exact_text_visible(driver, text, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, browser=None):
    if False:
        print('Hello World!')
    '\n    Searches for the specified element by the given selector. Returns the\n    element object if the text matches exactly with the text in the element,\n    and the text is visible.\n    Raises NoSuchElementException if the element does not exist in the HTML\n    within the specified timeout.\n    Raises ElementNotVisibleException if the element exists in the HTML,\n    but the exact text is not visible within the specified timeout.\n    @Params\n    driver - the webdriver object (required)\n    text - the exact text that is expected for the element (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    browser - used to handle a special edge case when using Safari\n    @Returns\n    A web element object that contains the text searched for\n    '
    element = None
    is_present = False
    actual_text = None
    text = str(text)
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        actual_text = None
        try:
            element = driver.find_element(by=by, value=selector)
            is_present = True
            if element.tag_name.lower() in ['input', 'textarea']:
                if element.is_displayed() and text.strip() == element.get_property('value').strip():
                    return element
                else:
                    if element.is_displayed():
                        actual_text = element.get_property('value').strip()
                    element = None
                    raise Exception()
            elif browser == 'safari':
                text_attr = 'innerText'
                if element.tag_name.lower() in ['input', 'textarea']:
                    text_attr = 'value'
                if element.is_displayed() and text.strip() == element.get_attribute(text_attr).strip():
                    return element
                else:
                    if element.is_displayed():
                        actual_text = element.get_attribute(text_attr)
                        actual_text = actual_text.strip()
                    element = None
                    raise Exception()
            elif element.is_displayed() and text.strip() == element.text.strip():
                return element
            else:
                if element.is_displayed():
                    actual_text = element.text.strip()
                element = None
                raise Exception()
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    if not element:
        if not is_present:
            message = 'Element {%s} was not present after %s second%s!' % (selector, timeout, plural)
            timeout_exception(NoSuchElementException, message)
        message = None
        if not actual_text or len(str(actual_text)) > 120:
            message = 'Expected exact text {%s} for {%s} was not visible after %s second%s!' % (text, selector, timeout, plural)
        else:
            actual_text = actual_text.replace('\n', '\\n')
            message = 'Expected exact text {%s} for {%s} was not visible after %s second%s!\n (Actual text was {%s})' % (text, selector, timeout, plural, actual_text)
        timeout_exception(TextNotVisibleException, message)
    else:
        return element

def wait_for_attribute(driver, selector, attribute, value=None, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        i = 10
        return i + 15
    '\n    Searches for the specified element attribute by the given selector.\n    Returns the element object if the expected attribute is present\n    and the expected attribute value is present (if specified).\n    Raises NoSuchElementException if the element does not exist in the HTML\n    within the specified timeout.\n    Raises NoSuchAttributeException if the element exists in the HTML,\n    but the expected attribute/value is not present within the timeout.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    attribute - the attribute that is expected for the element (required)\n    value - the attribute value that is expected (Default: None)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for the element attribute in seconds\n    @Returns\n    A web element object that contains the expected attribute/value\n    '
    element = None
    element_present = False
    attribute_present = False
    found_value = None
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        try:
            element = driver.find_element(by=by, value=selector)
            element_present = True
            attribute_present = False
            found_value = element.get_attribute(attribute)
            if found_value is not None:
                attribute_present = True
            else:
                element = None
                raise Exception()
            if value is not None:
                if found_value == value:
                    return element
                else:
                    element = None
                    raise Exception()
            else:
                return element
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    if not element:
        if not element_present:
            message = 'Element {%s} was not present after %s second%s!' % (selector, timeout, plural)
            timeout_exception(NoSuchElementException, message)
        if not attribute_present:
            message = 'Expected attribute {%s} of element {%s} was not present after %s second%s!' % (attribute, selector, timeout, plural)
            timeout_exception(NoSuchAttributeException, message)
        message = 'Expected value {%s} for attribute {%s} of element {%s} was not present after %s second%s! (The actual value was {%s})' % (value, attribute, selector, timeout, plural, found_value)
        timeout_exception(NoSuchAttributeException, message)
    else:
        return element

def wait_for_element_clickable(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, original_selector=None):
    if False:
        while True:
            i = 10
    '\n    Searches for the specified element by the given selector. Returns the\n    element object if the element is present, visible, & clickable on the page.\n    Raises NoSuchElementException if the element does not exist in the HTML\n    within the specified timeout.\n    Raises ElementNotVisibleException if the element exists in the HTML,\n    but is not visible (eg. opacity is "0") within the specified timeout.\n    Raises ElementNotInteractableException if the element is not clickable.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    original_selector - handle pre-converted ":contains(TEXT)" selector\n    @Returns\n    A web element object\n    '
    element = None
    is_present = False
    is_visible = False
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        try:
            element = driver.find_element(by=by, value=selector)
            is_present = True
            if element.is_displayed():
                is_visible = True
                if element.is_enabled():
                    return element
                else:
                    element = None
                    raise Exception()
            else:
                element = None
                raise Exception()
        except Exception:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    if not element and by != 'link text':
        if original_selector and ':contains(' in original_selector and ('contains(.' in selector):
            selector = original_selector
        if not is_present:
            message = 'Element {%s} was not present after %s second%s!' % (selector, timeout, plural)
            timeout_exception(NoSuchElementException, message)
        if not is_visible:
            message = 'Element {%s} was not visible after %s second%s!' % (selector, timeout, plural)
            timeout_exception(ElementNotVisibleException, message)
        message = 'Element {%s} was not clickable after %s second%s!' % (selector, timeout, plural)
        timeout_exception(ElementNotInteractableException, message)
    elif not element and by == 'link text' and (not is_visible):
        message = 'Link text {%s} was not found after %s second%s!' % (selector, timeout, plural)
        timeout_exception(LinkTextNotFoundException, message)
    elif not element and by == 'link text' and is_visible:
        message = 'Link text {%s} was not clickable after %s second%s!' % (selector, timeout, plural)
        timeout_exception(ElementNotInteractableException, message)
    else:
        return element

def wait_for_element_absent(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, original_selector=None):
    if False:
        return 10
    '\n    Searches for the specified element by the given selector.\n    Raises an exception if the element is still present after the\n    specified timeout.\n    @Params\n    driver - the webdriver object\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    original_selector - handle pre-converted ":contains(TEXT)" selector\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        try:
            driver.find_element(by=by, value=selector)
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
        except Exception:
            return True
    plural = 's'
    if timeout == 1:
        plural = ''
    if original_selector and ':contains(' in original_selector and ('contains(.' in selector):
        selector = original_selector
    message = 'Element {%s} was still present after %s second%s!' % (selector, timeout, plural)
    timeout_exception(Exception, message)

def wait_for_element_not_visible(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, original_selector=None):
    if False:
        i = 10
        return i + 15
    '\n    Searches for the specified element by the given selector.\n    Raises an exception if the element is still visible after the\n    specified timeout.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for the element in seconds\n    original_selector - handle pre-converted ":contains(TEXT)" selector\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        try:
            element = driver.find_element(by=by, value=selector)
            if element.is_displayed():
                now_ms = time.time() * 1000.0
                if now_ms >= stop_ms:
                    break
                time.sleep(0.1)
            else:
                return True
        except Exception:
            return True
    plural = 's'
    if timeout == 1:
        plural = ''
    if original_selector and ':contains(' in original_selector and ('contains(.' in selector):
        selector = original_selector
    message = 'Element {%s} was still visible after %s second%s!' % (selector, timeout, plural)
    timeout_exception(Exception, message)

def wait_for_text_not_visible(driver, text, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, browser=None):
    if False:
        i = 10
        return i + 15
    '\n    Searches for the text in the element of the given selector on the page.\n    Returns True if the text is not visible on the page within the timeout.\n    Raises an exception if the text is still present after the timeout.\n    @Params\n    driver - the webdriver object (required)\n    text - the text that is being searched for in the element (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    @Returns\n    A web element object that contains the text searched for\n    '
    text = str(text)
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        if not is_text_visible(driver, text, selector, by=by, browser=browser):
            return True
        now_ms = time.time() * 1000.0
        if now_ms >= stop_ms:
            break
        time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    message = 'Text {%s} in {%s} was still visible after %s second%s!' % (text, selector, timeout, plural)
    timeout_exception(Exception, message)

def wait_for_exact_text_not_visible(driver, text, selector, by='css selector', timeout=settings.LARGE_TIMEOUT, browser=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Searches for the text in the element of the given selector on the page.\n    Returns True if the element is missing the exact text within the timeout.\n    Raises an exception if the exact text is still present after the timeout.\n    @Params\n    driver - the webdriver object (required)\n    text - the text that is being searched for in the element (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    @Returns\n    A web element object that contains the text searched for\n    '
    text = str(text)
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        if not is_exact_text_visible(driver, text, selector, by=by, browser=browser):
            return True
        now_ms = time.time() * 1000.0
        if now_ms >= stop_ms:
            break
        time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    message = 'Exact text {%s} for {%s} was still visible after %s second%s!' % (text, selector, timeout, plural)
    timeout_exception(Exception, message)

def wait_for_non_empty_text_visible(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        print('Hello World!')
    '\n    Searches for any text in the element of the given selector.\n    Returns the element if it has visible text within the timeout.\n    Raises an exception if the element has no text within the timeout.\n    Whitespace-only text is considered empty text.\n    @Params\n    driver - the webdriver object (required)\n    text - the text that is being searched for in the element (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for elements in seconds\n    @Returns\n    The web element object that has text\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    element = None
    visible = None
    browser = None
    try:
        if 'safari:platformVersion' in driver.capabilities:
            browser = 'safari'
    except Exception:
        pass
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        try:
            element = None
            visible = False
            element = driver.find_element(by=by, value=selector)
            if element.is_displayed():
                visible = True
            element_text = element.text
            if browser == 'safari':
                if element.tag_name.lower() in ['input', 'textarea']:
                    element_text = element.get_attribute('value')
                else:
                    element_text = element.get_attribute('innerText')
            elif element.tag_name.lower() in ['input', 'textarea']:
                element_text = element.get_property('value')
            element_text = element_text.strip()
            if element.is_displayed() and len(element_text) > 0:
                return element
        except Exception:
            element = None
        now_ms = time.time() * 1000.0
        if now_ms >= stop_ms:
            break
        time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    if not element:
        message = 'Element {%s} was not present after %s second%s!' % (selector, timeout, plural)
        timeout_exception(NoSuchElementException, message)
    elif not visible:
        message = 'Element {%s} was not visible after %s second%s!' % (selector, timeout, plural)
        timeout_exception(ElementNotVisibleException, message)
    else:
        message = 'Element {%s} has no visible text after %s second%s!' % (selector, timeout, plural)
        timeout_exception(TextNotVisibleException, message)

def wait_for_attribute_not_present(driver, selector, attribute, value=None, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        i = 10
        return i + 15
    '\n    Searches for the specified element attribute by the given selector.\n    Returns True if the attribute isn\'t present on the page within the timeout.\n    Also returns True if the element is not present within the timeout.\n    Raises an exception if the attribute is still present after the timeout.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    attribute - the element attribute (required)\n    value - the attribute value (Default: None)\n    by - the type of selector being used (Default: "css selector")\n    timeout - the time to wait for the element attribute in seconds\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        if not is_attribute_present(driver, selector, attribute, value=value, by=by):
            return True
        now_ms = time.time() * 1000.0
        if now_ms >= stop_ms:
            break
        time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    message = 'Attribute {%s} of element {%s} was still present after %s second%s!' % (attribute, selector, timeout, plural)
    if value:
        message = 'Value {%s} for attribute {%s} of element {%s} was still present after %s second%s!' % (value, attribute, selector, timeout, plural)
    timeout_exception(Exception, message)

def find_visible_elements(driver, selector, by='css selector'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Finds all WebElements that match a selector and are visible.\n    Similar to webdriver.find_elements.\n    @Params\n    driver - the webdriver object (required)\n    selector - the locator for identifying the page element (required)\n    by - the type of selector being used (Default: "css selector")\n    '
    elements = driver.find_elements(by=by, value=selector)
    try:
        v_elems = [element for element in elements if element.is_displayed()]
        return v_elems
    except (StaleElementReferenceException, ElementNotInteractableException):
        time.sleep(0.1)
        elements = driver.find_elements(by=by, value=selector)
        v_elems = []
        for element in elements:
            if element.is_displayed():
                v_elems.append(element)
        return v_elems

def save_screenshot(driver, name, folder=None, selector=None, by='css selector'):
    if False:
        return 10
    "\n    Saves a screenshot of the current page.\n    If no folder is specified, uses the folder where pytest was called.\n    The screenshot will include the entire page unless a selector is given.\n    If a provided selector is not found, then takes a full-page screenshot.\n    If the folder provided doesn't exist, it will get created.\n    The screenshot will be in PNG format: (*.png)\n    "
    if not name.endswith('.png'):
        name = name + '.png'
    if folder:
        abs_path = os.path.abspath('.')
        file_path = os.path.join(abs_path, folder)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        screenshot_path = os.path.join(file_path, name)
    else:
        screenshot_path = name
    if selector:
        try:
            element = driver.find_element(by=by, value=selector)
            element_png = element.screenshot_as_png
            with open(screenshot_path, 'wb') as file:
                file.write(element_png)
        except Exception:
            if driver:
                driver.get_screenshot_as_file(screenshot_path)
            else:
                pass
    elif driver:
        driver.get_screenshot_as_file(screenshot_path)
    else:
        pass

def save_page_source(driver, name, folder=None):
    if False:
        print('Hello World!')
    "\n    Saves the page HTML to the current directory (or given subfolder).\n    If the folder specified doesn't exist, it will get created.\n    @Params\n    name - The file name to save the current page's HTML to.\n    folder - The folder to save the file to. (Default = current folder)\n    "
    from seleniumbase.core import log_helper
    if not name.endswith('.html'):
        name = name + '.html'
    if folder:
        abs_path = os.path.abspath('.')
        file_path = os.path.join(abs_path, folder)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        html_file_path = os.path.join(file_path, name)
    else:
        html_file_path = name
    page_source = driver.page_source
    html_file = codecs.open(html_file_path, 'w+', 'utf-8')
    rendered_source = log_helper.get_html_source_with_base_href(driver, page_source)
    html_file.write(rendered_source)
    html_file.close()

def wait_for_and_accept_alert(driver, timeout=settings.LARGE_TIMEOUT):
    if False:
        print('Hello World!')
    '\n    Wait for and accept an alert. Returns the text from the alert.\n    @Params\n    driver - the webdriver object (required)\n    timeout - the time to wait for the alert in seconds\n    '
    alert = wait_for_and_switch_to_alert(driver, timeout)
    alert_text = alert.text
    alert.accept()
    return alert_text

def wait_for_and_dismiss_alert(driver, timeout=settings.LARGE_TIMEOUT):
    if False:
        return 10
    '\n    Wait for and dismiss an alert. Returns the text from the alert.\n    @Params\n    driver - the webdriver object (required)\n    timeout - the time to wait for the alert in seconds\n    '
    alert = wait_for_and_switch_to_alert(driver, timeout)
    alert_text = alert.text
    alert.dismiss()
    return alert_text

def wait_for_and_switch_to_alert(driver, timeout=settings.LARGE_TIMEOUT):
    if False:
        print('Hello World!')
    '\n    Wait for a browser alert to appear, and switch to it. This should be usable\n    as a drop-in replacement for driver.switch_to.alert when the alert box\n    may not exist yet.\n    @Params\n    driver - the webdriver object (required)\n    timeout - the time to wait for the alert in seconds\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        try:
            alert = driver.switch_to.alert
            dummy_variable = alert.text
            return alert
        except NoAlertPresentException:
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    message = 'Alert was not present after %s seconds!' % timeout
    timeout_exception(Exception, message)

def switch_to_frame(driver, frame, timeout=settings.SMALL_TIMEOUT):
    if False:
        return 10
    '\n    Wait for an iframe to appear, and switch to it. This should be\n    usable as a drop-in replacement for driver.switch_to.frame().\n    @Params\n    driver - the webdriver object (required)\n    frame - the frame element, name, id, index, or selector\n    timeout - the time to wait for the alert in seconds\n    '
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    for x in range(int(timeout * 10)):
        shared_utils.check_if_time_limit_exceeded()
        try:
            driver.switch_to.frame(frame)
            return True
        except Exception:
            if type(frame) is str:
                by = None
                if page_utils.is_xpath_selector(frame):
                    by = 'xpath'
                else:
                    by = 'css selector'
                if is_element_visible(driver, frame, by=by):
                    try:
                        element = driver.find_element(by=by, value=frame)
                        driver.switch_to.frame(element)
                        return True
                    except Exception:
                        pass
            now_ms = time.time() * 1000.0
            if now_ms >= stop_ms:
                break
            time.sleep(0.1)
    plural = 's'
    if timeout == 1:
        plural = ''
    message = 'Frame {%s} was not visible after %s second%s!' % (frame, timeout, plural)
    timeout_exception(Exception, message)

def switch_to_window(driver, window, timeout=settings.SMALL_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wait for a window to appear, and switch to it. This should be usable\n    as a drop-in replacement for driver.switch_to.window().\n    @Params\n    driver - the webdriver object (required)\n    window - the window index or window handle\n    timeout - the time to wait for the window in seconds\n    '
    if window == -1:
        window = len(driver.window_handles) - 1
    start_ms = time.time() * 1000.0
    stop_ms = start_ms + timeout * 1000.0
    if isinstance(window, int):
        caps = driver.capabilities
        if caps['browserName'].lower() == 'safari' and 'safari:platformVersion' in caps:
            window = len(driver.window_handles) - 1 - window
            if window < 0:
                window = 0
        for x in range(int(timeout * 10)):
            shared_utils.check_if_time_limit_exceeded()
            try:
                window_handle = driver.window_handles[window]
                driver.switch_to.window(window_handle)
                return True
            except IndexError:
                now_ms = time.time() * 1000.0
                if now_ms >= stop_ms:
                    break
                time.sleep(0.1)
        plural = 's'
        if timeout == 1:
            plural = ''
        message = 'Window {%s} was not present after %s second%s!' % (window, timeout, plural)
        timeout_exception(Exception, message)
    else:
        window_handle = window
        for x in range(int(timeout * 10)):
            shared_utils.check_if_time_limit_exceeded()
            try:
                driver.switch_to.window(window_handle)
                return True
            except NoSuchWindowException:
                now_ms = time.time() * 1000.0
                if now_ms >= stop_ms:
                    break
                time.sleep(0.1)
        plural = 's'
        if timeout == 1:
            plural = ''
        message = 'Window {%s} was not present after %s second%s!' % (window, timeout, plural)
        timeout_exception(Exception, message)

def open_url(driver, url):
    if False:
        while True:
            i = 10
    url = str(url).strip()
    if not page_utils.looks_like_a_page_url(url):
        if page_utils.is_valid_url('https://' + url):
            url = 'https://' + url
    driver.get(url)

def click(driver, selector, by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        while True:
            i = 10
    (selector, by) = page_utils.recalculate_selector(selector, by)
    element = wait_for_element_clickable(driver, selector, by=by, timeout=timeout)
    element.click()

def click_link(driver, link_text, timeout=settings.SMALL_TIMEOUT):
    if False:
        print('Hello World!')
    element = wait_for_element_clickable(driver, link_text, by='link text', timeout=timeout)
    element.click()

def click_if_visible(driver, selector, by='css selector', timeout=0):
    if False:
        print('Hello World!')
    (selector, by) = page_utils.recalculate_selector(selector, by)
    if is_element_visible(driver, selector, by=by):
        click(driver, selector, by=by, timeout=1)
    elif timeout > 0:
        try:
            wait_for_element_visible(driver, selector, by=by, timeout=timeout)
        except Exception:
            pass
        if is_element_visible(driver, selector, by=by):
            click(driver, selector, by=by, timeout=1)

def click_active_element(driver):
    if False:
        return 10
    driver.execute_script('document.activeElement.click();')

def js_click(driver, selector, by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    (selector, by) = page_utils.recalculate_selector(selector, by)
    element = wait_for_element_present(driver, selector, by=by, timeout=timeout)
    if not element.is_displayed() or not element.is_enabled():
        time.sleep(0.2)
        element = wait_for_element_present(driver, selector, by=by, timeout=1)
    script = "var simulateClick = function (elem) {\n               var evt = new MouseEvent('click', {\n                   bubbles: true,\n                   cancelable: true,\n                   view: window\n               });\n               var canceled = !elem.dispatchEvent(evt);\n           };\n           var someLink = arguments[0];\n           simulateClick(someLink);"
    driver.execute_script(script, element)

def send_keys(driver, selector, text, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    (selector, by) = page_utils.recalculate_selector(selector, by)
    element = wait_for_element_clickable(driver, selector, by=by, timeout=timeout)
    if not text.endswith('\n'):
        element.send_keys(text)
    else:
        element.send_keys(text[:-1])
        element.submit()

def press_keys(driver, selector, text, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        return 10
    (selector, by) = page_utils.recalculate_selector(selector, by)
    element = wait_for_element_clickable(driver, selector, by=by, timeout=timeout)
    if not text.endswith('\n'):
        for key in text:
            element.send_keys(key)
    else:
        for key in text[:-1]:
            element.send_keys(key)
        element.send_keys(Keys.RETURN)

def update_text(driver, selector, text, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        return 10
    (selector, by) = page_utils.recalculate_selector(selector, by)
    element = wait_for_element_clickable(driver, selector, by=by, timeout=timeout)
    element.clear()
    if not text.endswith('\n'):
        element.send_keys(text)
    else:
        element.send_keys(text[:-1])
        element.submit()

def submit(driver, selector, by='css selector'):
    if False:
        return 10
    (selector, by) = page_utils.recalculate_selector(selector, by)
    element = wait_for_element_clickable(driver, selector, by=by, timeout=settings.SMALL_TIMEOUT)
    element.submit()

def has_attribute(driver, selector, attribute, value=None, by='css selector'):
    if False:
        for i in range(10):
            print('nop')
    (selector, by) = page_utils.recalculate_selector(selector, by)
    return is_attribute_present(driver, selector, attribute, value=value, by=by)

def assert_element_visible(driver, selector, by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        i = 10
        return i + 15
    original_selector = None
    if page_utils.is_valid_by(by):
        original_selector = selector
    elif page_utils.is_valid_by(selector):
        original_selector = by
    (selector, by) = page_utils.recalculate_selector(selector, by)
    wait_for_element_visible(driver, selector, by=by, timeout=timeout, original_selector=original_selector)

def assert_element_present(driver, selector, by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        print('Hello World!')
    original_selector = None
    if page_utils.is_valid_by(by):
        original_selector = selector
    elif page_utils.is_valid_by(selector):
        original_selector = by
    (selector, by) = page_utils.recalculate_selector(selector, by)
    wait_for_element_present(driver, selector, by=by, timeout=timeout, original_selector=original_selector)

def assert_element_not_visible(driver, selector, by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    original_selector = None
    if page_utils.is_valid_by(by):
        original_selector = selector
    elif page_utils.is_valid_by(selector):
        original_selector = by
    (selector, by) = page_utils.recalculate_selector(selector, by)
    wait_for_element_not_visible(driver, selector, by=by, timeout=timeout, original_selector=original_selector)

def assert_text(driver, text, selector='html', by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        return 10
    browser = driver.capabilities['browserName'].lower()
    wait_for_text_visible(driver, text.strip(), selector, by=by, timeout=timeout, browser=browser)

def assert_exact_text(driver, text, selector, by='css selector', timeout=settings.SMALL_TIMEOUT):
    if False:
        return 10
    browser = driver.capabilities['browserName'].lower()
    wait_for_exact_text_visible(driver, text.strip(), selector, by=by, timeout=timeout, browser=browser)

def wait_for_element(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    original_selector = None
    if page_utils.is_valid_by(by):
        original_selector = selector
    elif page_utils.is_valid_by(selector):
        original_selector = by
    (selector, by) = page_utils.recalculate_selector(selector, by)
    return wait_for_element_visible(driver=driver, selector=selector, by=by, timeout=timeout, original_selector=original_selector)

def wait_for_text(driver, text, selector, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        while True:
            i = 10
    browser = None
    try:
        if 'safari:platformVersion' in driver.capabilities:
            browser = 'safari'
    except Exception:
        pass
    return wait_for_text_visible(driver=driver, text=text, selector=selector, by=by, timeout=timeout, browser=browser)

def wait_for_exact_text(driver, text, selector, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        while True:
            i = 10
    browser = None
    try:
        if 'safari:platformVersion' in driver.capabilities:
            browser = 'safari'
    except Exception:
        pass
    return wait_for_exact_text_visible(driver=driver, text=text, selector=selector, by=by, timeout=timeout, browser=browser)

def wait_for_non_empty_text(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    return wait_for_non_empty_text_visible(driver=driver, selector=selector, by=by, timeout=timeout)

def get_text(driver, selector, by='css selector', timeout=settings.LARGE_TIMEOUT):
    if False:
        i = 10
        return i + 15
    browser = None
    try:
        if 'safari:platformVersion' in driver.capabilities:
            browser = 'safari'
    except Exception:
        pass
    element = wait_for_element(driver=driver, selector=selector, by=by, timeout=timeout)
    element_text = element.text
    if browser == 'safari':
        if element.tag_name.lower() in ['input', 'textarea']:
            element_text = element.get_attribute('value')
        else:
            element_text = element.get_attribute('innerText')
    elif element.tag_name.lower() in ['input', 'textarea']:
        element_text = element.get_property('value')
    return element_text