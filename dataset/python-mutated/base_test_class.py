from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoAlertPresentException, NoSuchElementException
import unittest
import os
import re
dd_driver = None
dd_driver_options = None

def on_exception_html_source_logger(func):
    if False:
        print('Hello World!')

    def wrapper(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print('exception occured at url:', self.driver.current_url)
            print('page source:', self.driver.page_source)
            f = open('selenium_page_source.html', 'w', encoding='utf-8')
            f.writelines(self.driver.page_source)
            raise e
    return wrapper

def set_suite_settings(suite, jira=False, github=False, block_execution=False):
    if False:
        i = 10
        return i + 15
    if jira:
        suite.addTest(BaseTestCase('enable_jira'))
    else:
        suite.addTest(BaseTestCase('disable_jira'))
    if github:
        suite.addTest(BaseTestCase('enable_github'))
    else:
        suite.addTest(BaseTestCase('disable_github'))
    if block_execution:
        suite.addTest(BaseTestCase('enable_block_execution'))
    else:
        suite.addTest(BaseTestCase('disable_block_execution'))

class BaseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.export_path = '/app'
        global dd_driver
        if not dd_driver:
            print('launching browser for: ', cls.__name__)
            global dd_driver_options
            dd_driver_options = Options()
            dd_driver_options.add_argument('--headless')
            dd_driver_options.add_argument('--no-sandbox')
            dd_driver_options.add_argument('--disable-dev-shm-usage')
            dd_driver_options.add_argument('--disable-gpu')
            dd_driver_options.add_argument('--window-size=1280,1024')
            dd_driver_options.set_capability('acceptInsecureCerts', True)
            desired = webdriver.DesiredCapabilities.CHROME
            desired['goog:loggingPrefs'] = {'browser': 'ALL'}
            prefs = {'download.default_directory': cls.export_path}
            dd_driver_options.add_experimental_option('prefs', prefs)
            print('starting chromedriver with options: ', vars(dd_driver_options), desired)
            dd_driver = webdriver.Chrome(os.environ['CHROMEDRIVER'], chrome_options=dd_driver_options, desired_capabilities=desired)
            dd_driver.implicitly_wait(1)
        cls.driver = dd_driver
        cls.base_url = os.environ['DD_BASE_URL']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.verificationErrors = []
        self.accept_next_alert = True
        self.accept_javascript_errors = False
        self.driver.execute_script('console.clear()')

    def login_page(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'login')
        driver.find_element(By.ID, 'id_username').clear()
        driver.find_element(By.ID, 'id_username').send_keys(os.environ['DD_ADMIN_USER'])
        driver.find_element(By.ID, 'id_password').clear()
        driver.find_element(By.ID, 'id_password').send_keys(os.environ['DD_ADMIN_PASSWORD'])
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-success').click()
        self.assertFalse(self.is_element_by_css_selector_present('.alert-danger', 'Please enter a correct username and password'))
        return driver

    def login_standard_page(self):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(self.base_url + 'login')
        driver.find_element(By.ID, 'id_username').clear()
        driver.find_element(By.ID, 'id_username').send_keys('propersahm')
        driver.find_element(By.ID, 'id_password').clear()
        driver.find_element(By.ID, 'id_password').send_keys('Def3ctD0jo&')
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-success').click()
        self.assertFalse(self.is_element_by_css_selector_present('.alert-danger', 'Please enter a correct username and password'))
        return driver

    def test_login(self):
        if False:
            while True:
                i = 10
        return self.login_page()

    def logout(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        driver.get(self.base_url + 'logout')
        self.assertTrue(self.is_text_present_on_page('Login'))
        return driver

    def test_logout(self):
        if False:
            while True:
                i = 10
        return self.logout()

    @on_exception_html_source_logger
    def delete_product_if_exists(self, name='QA Test'):
        if False:
            while True:
                i = 10
        driver = self.driver
        self.goto_product_overview(driver)
        qa_products = driver.find_elements(By.LINK_TEXT, name)
        if len(qa_products) > 0:
            self.test_delete_product(name)

    @on_exception_html_source_logger
    def delete_finding_template_if_exists(self, name='App Vulnerable to XSS'):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        driver.get(self.base_url + 'template')
        templates = driver.find_elements(By.LINK_TEXT, name)
        if len(templates) > 0:
            driver.find_element(By.ID, 'id_delete').click()
            driver.switch_to.alert.accept()

    def goto_some_page(self):
        if False:
            return 10
        driver = self.driver
        driver.get(self.base_url + 'user')
        return driver

    def goto_product_overview(self, driver):
        if False:
            i = 10
            return i + 15
        driver.get(self.base_url + 'product')
        self.wait_for_datatable_if_content('no_products', 'products_wrapper')
        return driver

    def goto_product_type_overview(self, driver):
        if False:
            while True:
                i = 10
        driver.get(self.base_url + 'product/type')
        return driver

    def goto_component_overview(self, driver):
        if False:
            print('Hello World!')
        driver.get(self.base_url + 'components')
        return driver

    def goto_google_sheets_configuration_form(self, driver):
        if False:
            for i in range(10):
                print('nop')
        driver.get(self.base_url + 'configure_google_sheets')
        return driver

    def goto_active_engagements_overview(self, driver):
        if False:
            while True:
                i = 10
        driver.get(self.base_url + 'engagement/active')
        return driver

    def goto_all_engagements_overview(self, driver):
        if False:
            i = 10
            return i + 15
        driver.get(self.base_url + 'engagement/all')
        return driver

    def goto_all_engagements_by_product_overview(self, driver):
        if False:
            return 10
        return self.goto_engagements_internal(driver, 'engagements_all')

    def goto_engagements_internal(self, driver, rel_url):
        if False:
            return 10
        driver.get(self.base_url + rel_url)
        self.wait_for_datatable_if_content('no_engagements', 'engagements_wrapper')
        return driver

    def goto_all_findings_list(self, driver):
        if False:
            i = 10
            return i + 15
        driver.get(self.base_url + 'finding')
        self.wait_for_datatable_if_content('no_findings', 'open_findings_wrapper')
        return driver

    def wait_for_datatable_if_content(self, no_content_id, wrapper_id):
        if False:
            i = 10
            return i + 15
        no_content = None
        try:
            no_content = self.driver.find_element(By.ID, no_content_id)
        except:
            pass
        if no_content is None:
            WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.ID, wrapper_id)))

    def is_element_by_css_selector_present(self, selector, text=None):
        if False:
            for i in range(10):
                print('nop')
        elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
        if len(elems) == 0:
            return False
        if text is None:
            return True
        for elem in elems:
            print(elem.text)
            if text in elem.text:
                return True
        return False

    def is_element_by_id_present(self, id):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.driver.find_element(By.ID, id)
            return True
        except NoSuchElementException:
            return False

    def is_success_message_present(self, text=None):
        if False:
            print('Hello World!')
        return self.is_element_by_css_selector_present('.alert-success', text=text)

    def is_error_message_present(self, text=None):
        if False:
            for i in range(10):
                print('nop')
        return self.is_element_by_css_selector_present('.alert-danger', text=text)

    def is_help_message_present(self, text=None):
        if False:
            print('Hello World!')
        return self.is_element_by_css_selector_present('.help-block', text=text)

    def is_text_present_on_page(self, text):
        if False:
            while True:
                i = 10
        body = self.driver.find_element(By.TAG_NAME, 'body')
        return re.search(text, body.text)

    def element_exists_by_id(self, id):
        if False:
            return 10
        elems = self.driver.find_elements(By.ID, id)
        return len(elems) > 0

    def change_system_setting(self, id, enable=True):
        if False:
            while True:
                i = 10
        print('changing system setting ' + id + ' enable: ' + str(enable))
        driver = self.driver
        driver.get(self.base_url + 'system_settings')
        is_enabled = driver.find_element(By.ID, id).is_selected()
        if enable and (not is_enabled) or (not enable and is_enabled):
            driver.find_element(By.ID, id).click()
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        is_enabled = driver.find_element(By.ID, id).is_selected()
        if enable:
            self.assertTrue(is_enabled)
        if not enable:
            self.assertFalse(is_enabled)
        return is_enabled

    def enable_system_setting(self, id):
        if False:
            return 10
        return self.change_system_setting(id, enable=True)

    def disable_system_setting(self, id):
        if False:
            print('Hello World!')
        return self.change_system_setting(id, enable=False)

    def enable_jira(self):
        if False:
            for i in range(10):
                print('nop')
        return self.enable_system_setting('id_enable_jira')

    def disable_jira(self):
        if False:
            while True:
                i = 10
        return self.disable_system_setting('id_enable_jira')

    def disable_github(self):
        if False:
            for i in range(10):
                print('nop')
        return self.disable_system_setting('id_enable_github')

    def enable_github(self):
        if False:
            return 10
        return self.enable_system_setting('id_enable_github')

    def set_block_execution(self, block_execution=True):
        if False:
            i = 10
            return i + 15
        print('setting block execution to: ', str(block_execution))
        driver = self.driver
        driver.get(self.base_url + 'profile')
        if driver.find_element(By.ID, 'id_block_execution').is_selected() != block_execution:
            driver.find_element(By.XPATH, '//*[@id="id_block_execution"]').click()
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
            self.assertTrue(driver.find_element(By.ID, 'id_block_execution').is_selected() == block_execution)
        return driver

    def enable_block_execution(self):
        if False:
            return 10
        self.set_block_execution()

    def disable_block_execution(self):
        if False:
            i = 10
            return i + 15
        self.set_block_execution(block_execution=False)

    def enable_deduplication(self):
        if False:
            return 10
        return self.enable_system_setting('id_enable_deduplication')

    def disable_deduplication(self):
        if False:
            return 10
        return self.disable_system_setting('id_enable_deduplication')

    def enable_false_positive_history(self):
        if False:
            i = 10
            return i + 15
        return self.enable_system_setting('id_false_positive_history')

    def disable_false_positive_history(self):
        if False:
            return 10
        return self.disable_system_setting('id_false_positive_history')

    def enable_retroactive_false_positive_history(self):
        if False:
            i = 10
            return i + 15
        return self.enable_system_setting('id_retroactive_false_positive_history')

    def disable_retroactive_false_positive_history(self):
        if False:
            print('Hello World!')
        return self.disable_system_setting('id_retroactive_false_positive_history')

    def is_alert_present(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.driver.switch_to_alert()
        except NoAlertPresentException:
            return False
        return True

    def close_alert_and_get_its_text(self):
        if False:
            while True:
                i = 10
        try:
            alert = self.driver.switch_to_alert()
            alert_text = alert.text
            if self.accept_next_alert:
                alert.accept()
            else:
                alert.dismiss()
            return alert_text
        finally:
            self.accept_next_alert = True

    def assertNoConsoleErrors(self):
        if False:
            i = 10
            return i + 15
        '\n        Sample output for levels (i.e. errors are SEVERE)\n        {\'level\': \'DEBUG\', \'message\': \'http://localhost:8080/product/type/4/edit 560:12 "debug"\', \'source\': \'console-api\', \'timestamp\': 1583952828410}\n        {\'level\': \'INFO\', \'message\': \'http://localhost:8080/product/type/4/edit 561:16 "info"\', \'source\': \'console-api\', \'timestamp\': 1583952828410}\n        {\'level\': \'WARNING\', \'message\': \'http://localhost:8080/product/type/4/edit 562:16 "warning"\', \'source\': \'console-api\', \'timestamp\': 1583952828410}\n        {\'level\': \'SEVERE\', \'message\': \'http://localhost:8080/product/type/4/edit 563:16 "error"\', \'source\': \'console-api\', \'timestamp\': 1583952828410}\n        '
        for entry in WebdriverOnlyNewLogFacade(self.driver).get_log('browser'):
            '\n            Images are now working after https://github.com/DefectDojo/django-DefectDojo/pull/3954,\n            but http://localhost:8080/static/dojo/img/zoom-in.cur still produces a 404\n\n            The addition of the trigger exception is due to the Report Builder tests.\n            The addition of the innerHTML exception is due to the test for quick reports in finding_test.py\n            '
            accepted_javascript_messages = "(zoom\\-in\\.cur.*)404\\ \\(Not\\ Found\\)|Uncaught TypeError: Cannot read properties of null \\(reading \\'trigger\\'\\)|Uncaught TypeError: Cannot read properties of null \\(reading \\'innerHTML\\'\\)"
            if entry['level'] == 'SEVERE':
                print(entry)
                print('There was a SEVERE javascript error in the console, please check all steps fromt the current test to see where it happens')
                print('Currently there is no reliable way to find out at which url the error happened, but it could be: .' + self.driver.current_url)
                if self.accept_javascript_errors:
                    print('WARNING: skipping SEVERE javascript error because accept_javascript_errors is True!')
                elif re.search(accepted_javascript_messages, entry['message']):
                    print('WARNING: skipping javascript errors related to known issues images, see https://github.com/DefectDojo/django-DefectDojo/blob/master/tests/base_test_class.py#L324')
                else:
                    self.assertNotEqual(entry['level'], 'SEVERE')
        return True

    def tearDown(self):
        if False:
            return 10
        self.assertNoConsoleErrors()
        self.assertEqual([], self.verificationErrors)

    @classmethod
    def tearDownDriver(cls):
        if False:
            print('Hello World!')
        print('tearDownDriver: ', cls.__name__)
        global dd_driver
        if dd_driver:
            if not dd_driver_options.experimental_options or not dd_driver_options.experimental_options.get('detach'):
                print('closing browser')
                dd_driver.quit()

class WebdriverOnlyNewLogFacade(object):
    last_timestamp = 0

    def __init__(self, webdriver):
        if False:
            while True:
                i = 10
        self._webdriver = webdriver

    def get_log(self, log_type):
        if False:
            while True:
                i = 10
        last_timestamp = self.last_timestamp
        entries = self._webdriver.get_log(log_type)
        filtered = []
        for entry in entries:
            if entry['timestamp'] > self.last_timestamp:
                filtered.append(entry)
                if entry['timestamp'] > last_timestamp:
                    last_timestamp = entry['timestamp']
        self.last_timestamp = last_timestamp
        return filtered