import unittest
import time
import sys
import os
import remi
examples_dir = os.path.realpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../examples'))
sys.path.append(examples_dir)
from helloworld_app import MyApp
try:
    from selenium import webdriver
except ImportError:
    webdriver = None

class TestHelloWorld(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.options = self.OptionsClass()
            self.options.headless = True
            self.driver = self.DriverClass(options=self.options)
            self.driver.implicitly_wait(30)
        except Exception:
            self.skipTest('Selenium webdriver is not installed')
        self.server = remi.Server(MyApp, start=False, address='0.0.0.0', start_browser=False, multiple_instance=True)
        self.server.start()

    def test_should_open_browser(self):
        if False:
            while True:
                i = 10
        self.driver.get(self.server.address)
        button = self.driver.find_element_by_tag_name('button')
        self.assertTrue('Press me!' in button.text)

    def test_button_press(self):
        if False:
            i = 10
            return i + 15
        self.driver.get(self.server.address)
        body = self.driver.find_element_by_tag_name('body')
        self.assertNotIn('Hello World!', body.text)
        time.sleep(1.0)
        button = self.driver.find_element_by_tag_name('button')
        button.click()
        time.sleep(1.0)
        body = self.driver.find_elements_by_tag_name('body')[-1]
        self.assertIn('Hello World!', body.text)
        time.sleep(1.0)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.driver.quit()
        self.server.stop()

class TestHelloWorldChrome(TestHelloWorld):
    DriverClass = getattr(webdriver, 'Chrome', None)
    OptionsClass = getattr(webdriver, 'ChromeOptions', None)

class TestHelloWorldFirefox(TestHelloWorld):
    DriverClass = getattr(webdriver, 'Firefox', None)
    OptionsClass = getattr(webdriver, 'FirefoxOptions', None)
del TestHelloWorld
if __name__ == '__main__':
    unittest.main(buffer=True, verbosity=2)