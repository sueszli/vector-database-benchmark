from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import unittest
import sys
import time
from base_test_class import BaseTestCase, on_exception_html_source_logger, set_suite_settings
from notifications_test import NotificationTest

class WaitForPageLoad(object):

    def __init__(self, browser, timeout):
        if False:
            for i in range(10):
                print('nop')
        self.browser = browser
        self.timeout = time.time() + timeout

    def __enter__(self):
        if False:
            return 10
        self.old_page = self.browser.find_element(By.TAG_NAME, 'html')

    def page_has_loaded(self):
        if False:
            while True:
                i = 10
        new_page = self.browser.find_element(By.TAG_NAME, 'html')
        return new_page.id != self.old_page.id

    def __exit__(self, *_):
        if False:
            i = 10
            return i + 15
        while time.time() < self.timeout:
            if self.page_has_loaded():
                return True
            else:
                time.sleep(0.2)
        raise Exception('Timeout waiting for {}s'.format(self.timeout))

class ProductTest(BaseTestCase):

    @on_exception_html_source_logger
    def test_create_product(self):
        if False:
            i = 10
            return i + 15
        self.delete_product_if_exists()
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Add Product').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('QA Test')
        driver.find_element(By.ID, 'id_name').send_keys('\tThis is just a test. Be very afraid.')
        Select(driver.find_element(By.ID, 'id_prod_type')).select_by_visible_text('Research and Development')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product added successfully') or self.is_success_message_present(text='Product with this Name already exists.'))
        self.assertFalse(self.is_error_message_present())

    @on_exception_html_source_logger
    def test_list_products(self):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)

    @on_exception_html_source_logger
    def test_list_components(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.LINK_TEXT, 'Components').click()
        driver.find_element(By.ID, 'product_component_view').click()
        self.assertTrue(self.is_element_by_css_selector_present('table'))

    @on_exception_html_source_logger
    def test_edit_product_description(self):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Edit').click()
        driver.find_element(By.ID, 'id_name').send_keys(Keys.TAB, 'Updated Desription: ')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product updated successfully') or self.is_success_message_present(text='Product with this Name already exists.'))
        self.assertFalse(self.is_error_message_present())

    @on_exception_html_source_logger
    def test_enable_simple_risk_acceptance(self):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Edit').click()
        driver.find_element(By.XPATH, '//*[@id="id_enable_simple_risk_acceptance"]').click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product updated successfully') or self.is_success_message_present(text='Product with this Name already exists.'))
        self.assertFalse(self.is_error_message_present())

    @on_exception_html_source_logger
    def test_add_product_engagement(self):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Engagement').click()
        driver.find_element(By.LINK_TEXT, 'Add New Interactive Engagement').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Beta Test')
        driver.find_element(By.ID, 'id_name').send_keys(Keys.TAB, 'Running Test on product before approving and push to production.')
        Select(driver.find_element(By.ID, 'id_lead')).select_by_visible_text('Admin User (admin)')
        Select(driver.find_element(By.ID, 'id_status')).select_by_visible_text('In Progress')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully'))

    @on_exception_html_source_logger
    def test_add_technology(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.ID, 'addTechnology').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Technology Test')
        driver.find_element(By.ID, 'id_version').clear()
        driver.find_element(By.ID, 'id_version').send_keys('2.1.0-RELEASE')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Technology added successfully'))
        self.assertEqual(driver.find_elements(By.NAME, 'technology_name')[0].text, 'Technology Test')
        self.assertEqual(driver.find_elements(By.NAME, 'technology_version')[0].text, 'v.2.1.0-RELEASE')

    @on_exception_html_source_logger
    def test_edit_technology(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_elements(By.NAME, 'dropdownManageTechnologies')[0].click()
        driver.find_elements(By.NAME, 'editTechnology')[0].click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Technology Changed')
        driver.find_element(By.ID, 'id_version').clear()
        driver.find_element(By.ID, 'id_version').send_keys('2.2.0-RELEASE')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Technology changed successfully'))
        self.assertEqual(driver.find_elements(By.NAME, 'technology_name')[0].text, 'Technology Changed')
        self.assertEqual(driver.find_elements(By.NAME, 'technology_version')[0].text, 'v.2.2.0-RELEASE')

    @on_exception_html_source_logger
    def test_delete_technology(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_elements(By.NAME, 'dropdownManageTechnologies')[0].click()
        driver.find_elements(By.NAME, 'deleteTechnology')[0].click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Technology deleted successfully'))
        self.assertFalse(driver.find_elements(By.NAME, 'technology_name'))

    @on_exception_html_source_logger
    def test_add_product_finding(self):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Findings').click()
        driver.find_element(By.LINK_TEXT, 'Add New Finding').click()
        driver.find_element(By.ID, 'id_title').clear()
        driver.find_element(By.ID, 'id_title').send_keys('App Vulnerable to XSS')
        Select(driver.find_element(By.ID, 'id_severity')).select_by_visible_text('High')
        driver.find_element(By.ID, 'id_cvssv3').send_keys('CVSS:3.0/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H')
        driver.find_element(By.ID, 'id_cvssv3').send_keys(Keys.TAB, 'This is just a Test Case Finding')
        driver.find_element(By.ID, 'id_vulnerability_ids').send_keys('REF-1\nREF-2')
        driver.execute_script("document.getElementsByName('mitigation')[0].style.display = 'inline'")
        driver.find_element(By.NAME, 'mitigation').send_keys(Keys.TAB, 'How to mitigate this finding')
        driver.execute_script("document.getElementsByName('impact')[0].style.display = 'inline'")
        driver.find_element(By.NAME, 'impact').send_keys(Keys.TAB, 'This has a very critical effect on production')
        driver.find_element(By.ID, 'id_endpoints_to_add').send_keys('product.finding.com')
        with WaitForPageLoad(driver, timeout=30):
            driver.find_element(By.XPATH, "//input[@name='_Finished']").click()
        self.assertTrue(self.is_text_present_on_page(text='App Vulnerable to XSS'))
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        self.assertTrue(self.is_text_present_on_page(text='product.finding.com'))
        self.assertTrue(self.is_text_present_on_page(text='REF-1'))
        self.assertTrue(self.is_text_present_on_page(text='REF-2'))
        self.assertTrue(self.is_text_present_on_page(text='Additional Vulnerability Ids'))

    @on_exception_html_source_logger
    def test_add_product_endpoints(self):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Endpoints').click()
        driver.find_element(By.LINK_TEXT, 'Add New Endpoint').click()
        driver.find_element(By.ID, 'id_endpoint').clear()
        driver.find_element(By.ID, 'id_endpoint').send_keys('strange.prod.dev\n123.45.6.30')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Endpoint added successfully'))

    @on_exception_html_source_logger
    def test_add_product_custom_field(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Add Custom Fields').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Security Level')
        driver.find_element(By.ID, 'id_value').clear()
        driver.find_element(By.ID, 'id_value').send_keys('Loose')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Metadata added successfully') or self.is_success_message_present(text='A metadata entry with the same name exists already for this object.'))

    @on_exception_html_source_logger
    def test_edit_product_custom_field(self):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Edit Custom Fields').click()
        driver.find_element(By.XPATH, "//input[@value='Loose']").clear()
        driver.find_element(By.XPATH, "//input[@value='Loose']").send_keys('Strong')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Metadata edited successfully') or self.is_success_message_present(text='A metadata entry with the same name exists already for this object.'))

    @on_exception_html_source_logger
    def test_add_product_tracking_files(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Add Product Tracking Files').click()
        driver.find_element(By.ID, 'id_path').clear()
        driver.find_element(By.ID, 'id_path').send_keys('/strange/folder/')
        Select(driver.find_element(By.ID, 'id_review_status')).select_by_visible_text('Untracked')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Added Tracked File to a Product'))

    @on_exception_html_source_logger
    def test_edit_product_tracking_files(self):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'View Product Tracking Files').click()
        driver.find_element(By.LINK_TEXT, 'Edit').click()
        driver.find_element(By.ID, 'id_path').clear()
        driver.find_element(By.ID, 'id_path').send_keys('/unknown/folder/')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Tool Product Configuration Successfully Updated'))

    def test_product_metrics(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Metrics').click()

    @on_exception_html_source_logger
    def test_delete_product(self, name='QA Test'):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, name).click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Delete').click()
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Product and relationships removed.'))

    @on_exception_html_source_logger
    def test_product_notifications_change(self):
        if False:
            i = 10
            return i + 15
        NotificationTest('enable_notification', 'mail').enable_notification()
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.XPATH, "//input[@name='engagement_added' and @value='mail']").click()
        self.assertTrue(self.is_success_message_present(text='Notification settings updated'))
        self.assertTrue(driver.find_element(By.XPATH, "//input[@name='engagement_added' and @value='mail']").is_selected())
        self.assertFalse(driver.find_element(By.XPATH, "//input[@name='scan_added' and @value='mail']").is_selected())
        self.assertFalse(driver.find_element(By.XPATH, "//input[@name='test_added' and @value='mail']").is_selected())
        driver.find_element(By.XPATH, "//input[@name='scan_added' and @value='mail']").click()
        self.assertTrue(self.is_success_message_present(text='Notification settings updated'))
        self.assertTrue(driver.find_element(By.XPATH, "//input[@name='engagement_added' and @value='mail']").is_selected())
        self.assertTrue(driver.find_element(By.XPATH, "//input[@name='scan_added' and @value='mail']").is_selected())
        self.assertFalse(driver.find_element(By.XPATH, "//input[@name='test_added' and @value='mail']").is_selected())

    def test_critical_product_metrics(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        driver.get(self.base_url + 'critical_product_metrics')

    def test_product_type_metrics(self):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(self.base_url + 'metrics/product/type')

    def test_product_type_counts_metrics(self):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(self.base_url + 'metrics/product/type/counts')
        my_select = Select(driver.find_element(By.ID, 'id_product_type'))
        my_select.select_by_index(1)
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()

    def test_simple_metrics(self):
        if False:
            return 10
        driver = self.driver
        driver.get(self.base_url + 'metrics/simple')

    def test_engineer_metrics(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        driver.get(self.base_url + 'metrics/engineer')

    def test_metrics_dashboard(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        driver.get(self.base_url + 'metrics?date=5&view=dashboard')

def add_product_tests_to_suite(suite, jira=False, github=False, block_execution=False):
    if False:
        for i in range(10):
            print('nop')
    suite.addTest(BaseTestCase('test_login'))
    set_suite_settings(suite, jira=jira, github=github, block_execution=block_execution)
    suite.addTest(ProductTest('test_create_product'))
    suite.addTest(ProductTest('test_edit_product_description'))
    suite.addTest(ProductTest('test_add_technology'))
    suite.addTest(ProductTest('test_edit_technology'))
    suite.addTest(ProductTest('test_delete_technology'))
    suite.addTest(ProductTest('test_add_product_engagement'))
    suite.addTest(ProductTest('test_add_product_finding'))
    suite.addTest(ProductTest('test_add_product_endpoints'))
    suite.addTest(ProductTest('test_add_product_custom_field'))
    suite.addTest(ProductTest('test_edit_product_custom_field'))
    suite.addTest(ProductTest('test_add_product_tracking_files'))
    suite.addTest(ProductTest('test_edit_product_tracking_files'))
    suite.addTest(ProductTest('test_list_products'))
    suite.addTest(ProductTest('test_list_components'))
    suite.addTest(ProductTest('test_product_notifications_change'))
    suite.addTest(ProductTest('test_product_metrics'))
    suite.addTest(ProductTest('test_critical_product_metrics'))
    suite.addTest(ProductTest('test_product_type_metrics'))
    suite.addTest(ProductTest('test_product_type_counts_metrics'))
    suite.addTest(ProductTest('test_simple_metrics'))
    suite.addTest(ProductTest('test_engineer_metrics'))
    suite.addTest(ProductTest('test_metrics_dashboard'))
    suite.addTest(ProductTest('test_delete_product'))
    return suite

def suite():
    if False:
        for i in range(10):
            print('nop')
    suite = unittest.TestSuite()
    add_product_tests_to_suite(suite, jira=False, github=False, block_execution=False)
    add_product_tests_to_suite(suite, jira=True, github=True, block_execution=True)
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)