import unittest
import sys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from base_test_class import BaseTestCase
from product_test import ProductTest, WaitForPageLoad

class FalsePositiveHistoryTest(BaseTestCase):

    def create_finding(self, product_name, engagement_name, test_name, finding_name):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.ID, 'products_wrapper')
        driver.find_element(By.LINK_TEXT, product_name).click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Engagement').click()
        driver.find_element(By.LINK_TEXT, 'Add New Interactive Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys(engagement_name)
        driver.find_element(By.NAME, '_Add Tests').click()
        driver.find_element(By.ID, 'id_title').send_keys(test_name)
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Manual Code Review')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Test')
        driver.find_element(By.NAME, '_Add Findings').click()
        driver.find_element(By.ID, 'id_title').send_keys(finding_name)
        driver.find_element(By.ID, 'id_cvssv3').send_keys('CVSS:3.0/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H')
        driver.find_element(By.ID, 'id_cvssv3').send_keys(Keys.TAB, 'This is just a Test Case Finding')
        driver.find_element(By.ID, 'id_vulnerability_ids').send_keys('REF-1\nREF-2')
        with WaitForPageLoad(driver, timeout=30):
            driver.find_element(By.XPATH, "//input[@name='_Finished']").click()
        self.assertTrue(self.is_text_present_on_page(text=finding_name))
        driver.find_element(By.LINK_TEXT, finding_name).click()
        return driver.current_url

    def assert_is_active(self, finding_url):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(finding_url)
        self.assertTrue(self.is_element_by_css_selector_present(selector='#notes', text='Active'))
        self.assertFalse(self.is_element_by_css_selector_present(selector='#notes', text='False Positive'))

    def assert_is_false_positive(self, finding_url):
        if False:
            return 10
        driver = self.driver
        driver.get(finding_url)
        self.assertFalse(self.is_element_by_css_selector_present(selector='#notes', text='Active'))
        self.assertTrue(self.is_element_by_css_selector_present(selector='#notes', text='False Positive'))

    def edit_toggle_false_positive(self, finding_url):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        driver.get(finding_url)
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Edit Finding').click()
        driver.find_element(By.ID, 'id_active').click()
        driver.find_element(By.ID, 'id_verified').click()
        driver.find_element(By.ID, 'id_false_p').click()
        driver.find_element(By.XPATH, "//input[@name='_Finished']").click()

    def bulk_edit(self, finding_url, status_id):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(finding_url)
        driver.find_element(By.CSS_SELECTOR, "a[title='Test']").click()
        driver.find_element(By.ID, 'select_all').click()
        driver.find_element(By.ID, 'dropdownMenu2').click()
        driver.find_element(By.ID, 'id_bulk_status').click()
        driver.find_element(By.ID, status_id).click()
        driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()

    def test_retroactive_edit_finding(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        finding_1 = self.create_finding(product_name='QA Test', engagement_name='FP History Eng 1', test_name='FP History Test', finding_name='Fake Vulnerability for Edit Test')
        finding_2 = self.create_finding(product_name='QA Test', engagement_name='FP History Eng 2', test_name='FP History Test', finding_name='Fake Vulnerability for Edit Test')
        self.assert_is_active(finding_1)
        self.assert_is_active(finding_2)
        self.edit_toggle_false_positive(finding_1)
        self.assert_is_false_positive(finding_1)
        self.assert_is_false_positive(finding_2)
        self.edit_toggle_false_positive(finding_2)
        self.assert_is_active(finding_1)
        self.assert_is_active(finding_2)

    def test_retroactive_bulk_edit_finding(self):
        if False:
            print('Hello World!')
        driver = self.driver
        finding_1 = self.create_finding(product_name='QA Test', engagement_name='FP History Eng 1', test_name='FP History Test', finding_name='Fake Vulnerability for Bulk Edit Test')
        finding_2 = self.create_finding(product_name='QA Test', engagement_name='FP History Eng 2', test_name='FP History Test', finding_name='Fake Vulnerability for Bulk Edit Test')
        self.assert_is_active(finding_1)
        self.assert_is_active(finding_2)
        self.bulk_edit(finding_1, status_id='id_bulk_false_p')
        self.assert_is_false_positive(finding_1)
        self.assert_is_false_positive(finding_2)
        self.bulk_edit(finding_2, status_id='id_bulk_active')
        self.assert_is_active(finding_1)
        self.assert_is_active(finding_2)

def suite():
    if False:
        print('Hello World!')
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(BaseTestCase('enable_block_execution'))
    suite.addTest(BaseTestCase('disable_deduplication'))
    suite.addTest(BaseTestCase('enable_false_positive_history'))
    suite.addTest(BaseTestCase('enable_retroactive_false_positive_history'))
    suite.addTest(ProductTest('test_create_product'))
    suite.addTest(FalsePositiveHistoryTest('test_retroactive_edit_finding'))
    suite.addTest(ProductTest('test_create_product'))
    suite.addTest(FalsePositiveHistoryTest('test_retroactive_bulk_edit_finding'))
    suite.addTest(ProductTest('test_delete_product'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)