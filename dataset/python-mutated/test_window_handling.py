from helium import write, click, switch_to, TextField, Text, get_driver, Link, wait_until
from selenium.webdriver.common.by import By
from tests.api import BrowserAT, test_browser_name
from unittest import skipIf

class WindowHandlingTest(BrowserAT):

    def get_page(self):
        if False:
            return 10
        return 'test_window_handling/main.html'

    def test_write_writes_in_active_window(self):
        if False:
            for i in range(10):
                print('nop')
        write('Main window')
        self.assertEqual('Main window', self._get_value('mainTextField'))
        self._open_popup()
        write('Popup')
        self.assertEqual('Popup', self._get_value('popupTextField'))

    def test_write_searches_in_active_window(self):
        if False:
            for i in range(10):
                print('nop')
        write('Main window', into='Text field')
        self.assertEqual('Main window', self._get_value('mainTextField'))
        self._open_popup()
        write('Popup', into='Text field')
        self.assertEqual('Popup', self._get_value('popupTextField'))

    def test_switch_to_search_text_field(self):
        if False:
            return 10
        write('Main window', into='Text field')
        self.assertEqual('Main window', TextField('Text field').value)
        self._open_popup()
        write('Popup', into='Text field')
        self.assertEqual('Popup', TextField('Text field').value)
        switch_to('test_window_handling - Main')
        self.assertEqual('Main window', TextField('Text field').value)

    def test_handles_closed_window_gracefully(self):
        if False:
            i = 10
            return i + 15
        self._open_popup()
        get_driver().close()
        is_back_in_main_window = Link('Open popup').exists()
        self.assertTrue(is_back_in_main_window)

    def test_switch_to_after_window_closed(self):
        if False:
            while True:
                i = 10
        self._open_popup()
        get_driver().close()
        switch_to('test_window_handling - Main')

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.main_window_handle = self.driver.current_window_handle

    def tearDown(self):
        if False:
            print('Hello World!')
        for window_handle in self.driver.window_handles:
            if window_handle != self.main_window_handle:
                self.driver.switch_to.window(window_handle)
                self.driver.close()
        self.driver.switch_to.window(self.main_window_handle)
        super().tearDown()

    def _get_value(self, element_id):
        if False:
            for i in range(10):
                print('nop')
        return self.driver.find_element(By.ID, element_id).get_attribute('value')

    def _open_popup(self):
        if False:
            return 10
        click('Open popup')
        wait_until(self._is_in_popup)

    def _is_in_popup(self):
        if False:
            while True:
                i = 10
        return get_driver().title == 'test_window_handling - Popup'

class WindowHandlingOnStartBrowserTest(BrowserAT):

    def get_page(self):
        if False:
            print('Hello World!')
        return 'test_window_handling/main_immediate_popup.html'

    @skipIf(test_browser_name() == 'firefox', 'This test fails on Firefox')
    def test_switches_to_popup(self):
        if False:
            print('Hello World!')
        self.assertTrue(Text('In popup.').exists())