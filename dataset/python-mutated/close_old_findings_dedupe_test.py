import logging
import os
import sys
import time
import unittest
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from base_test_class import BaseTestCase, on_exception_html_source_logger, set_suite_settings
from product_test import ProductTest
logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))

class CloseOldDedupeTest(BaseTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.relative_path = dir_path = os.path.dirname(os.path.realpath(__file__))

    def check_nb_duplicates(self, expected_number_of_duplicates):
        if False:
            return 10
        logger.debug('checking duplicates...')
        driver = self.driver
        retries = 0
        for i in range(0, 18):
            time.sleep(5)
            self.goto_all_findings_list(driver)
            dupe_count = 0
            trs = driver.find_elements(By.XPATH, '//*[@id="open_findings"]/tbody/tr')
            for row in trs:
                concatRow = ' '.join([td.text for td in row.find_elements(By.XPATH, './/td')])
                if '(DUPE)' and 'Duplicate' in concatRow:
                    dupe_count += 1
            if dupe_count != expected_number_of_duplicates:
                logger.debug("duplicate count mismatch, let's wait a bit for the celery dedupe task to finish and try again (5s)")
            else:
                break
        if dupe_count != expected_number_of_duplicates:
            findings_table = driver.find_element(By.ID, 'open_findings')
            print(findings_table.get_attribute('innerHTML'))
        self.assertEqual(dupe_count, expected_number_of_duplicates)

    @on_exception_html_source_logger
    def test_enable_deduplication(self):
        if False:
            while True:
                i = 10
        logger.debug('enabling deduplication...')
        driver = self.driver
        driver.get(self.base_url + 'system_settings')
        if not driver.find_element(By.ID, 'id_enable_deduplication').is_selected():
            driver.find_element(By.XPATH, '//*[@id="id_enable_deduplication"]').click()
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
            driver.get(self.base_url + 'system_settings')
            self.assertTrue(driver.find_element(By.ID, 'id_enable_deduplication').is_selected())

    @on_exception_html_source_logger
    def test_delete_findings(self):
        if False:
            return 10
        logger.debug('removing previous findings...')
        driver = self.driver
        driver.get(self.base_url + 'finding?page=1')
        if self.element_exists_by_id('no_findings'):
            text = driver.find_element(By.ID, 'no_findings').text
            if 'No findings found.' in text:
                return
        driver.find_element(By.ID, 'select_all').click()
        driver.find_element(By.CSS_SELECTOR, 'i.fa-solid.fa-trash').click()
        try:
            WebDriverWait(driver, 1).until(EC.alert_is_present(), 'Timed out waiting for finding delete ' + 'confirmation popup to appear.')
            driver.switch_to.alert.accept()
        except TimeoutException:
            self.fail('Confirmation dialogue not shown, cannot delete previous findings')
        logger.debug('page source when checking for no_findings element')
        logger.debug(self.driver.page_source)
        text = driver.find_element(By.ID, 'no_findings').text
        self.assertIsNotNone(text)
        self.assertTrue('No findings found.' in text)
        self.assertTrue(driver.current_url.endswith('page=1'))

    @on_exception_html_source_logger
    def test_add_same_engagement_engagement(self):
        if False:
            i = 10
            return i + 15
        logger.debug('Same scanner deduplication - Close Old Findings Same Engagement - dynamic. Creating tests...')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Close Same Engagement Test')
        driver.find_element(By.XPATH, '//*[@id="id_deduplication_on_engagement"]').click()
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))

    @on_exception_html_source_logger
    def test_import_same_engagement_tests(self):
        if False:
            i = 10
            return i + 15
        logger.debug('Importing reports...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Close Same Engagement Test').click()
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        driver.find_element(By.LINK_TEXT, 'Import Scan Results').click()
        scan_type = Select(driver.find_element(By.ID, 'id_scan_type'))
        scan_type.select_by_visible_text('Immuniweb Scan')
        scan_environment = Select(driver.find_element(By.ID, 'id_environment'))
        scan_environment.select_by_visible_text('Development')
        driver.find_element(By.ID, 'id_close_old_findings').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_1.xml')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='3 findings and closed 0 findings'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Close Same Engagement Test').click()
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        driver.find_element(By.LINK_TEXT, 'Import Scan Results').click()
        scan_type = Select(driver.find_element(By.ID, 'id_scan_type'))
        scan_type.select_by_visible_text('Immuniweb Scan')
        scan_environment = Select(driver.find_element(By.ID, 'id_environment'))
        scan_environment.select_by_visible_text('Development')
        driver.find_element(By.ID, 'id_close_old_findings').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_1.xml')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='3 findings and closed 0 findings'))

    @on_exception_html_source_logger
    def test_close_same_engagement_tests(self):
        if False:
            while True:
                i = 10
        logger.debug('Importing reports...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Close Same Engagement Test').click()
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        driver.find_element(By.LINK_TEXT, 'Import Scan Results').click()
        scan_type = Select(driver.find_element(By.ID, 'id_scan_type'))
        scan_type.select_by_visible_text('Immuniweb Scan')
        scan_environment = Select(driver.find_element(By.ID, 'id_environment'))
        scan_environment.select_by_visible_text('Development')
        driver.find_element(By.ID, 'id_close_old_findings').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_and_close_1.xml')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='1 findings and closed 2 findings'))

    @on_exception_html_source_logger
    def test_check_endpoint_status(self):
        if False:
            return 10
        self.check_nb_duplicates(4)

    @on_exception_html_source_logger
    def test_add_same_product_engagement(self):
        if False:
            return 10
        logger.debug('Same scanner deduplication - Close Old Findings Same Product - dynamic. Creating tests...')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Close Same Product Test 1')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Close Same Product Test 2')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Close Same Product Test 3')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))

    @on_exception_html_source_logger
    def test_import_same_product_tests(self):
        if False:
            while True:
                i = 10
        logger.debug('Importing reports...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Close Same Product Test 1').click()
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        driver.find_element(By.LINK_TEXT, 'Import Scan Results').click()
        scan_type = Select(driver.find_element(By.ID, 'id_scan_type'))
        scan_type.select_by_visible_text('Immuniweb Scan')
        scan_environment = Select(driver.find_element(By.ID, 'id_environment'))
        scan_environment.select_by_visible_text('Development')
        driver.find_element(By.ID, 'id_close_old_findings_product_scope').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_1.xml')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='3 findings and closed 0 findings'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Close Same Product Test 2').click()
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        driver.find_element(By.LINK_TEXT, 'Import Scan Results').click()
        scan_type = Select(driver.find_element(By.ID, 'id_scan_type'))
        scan_type.select_by_visible_text('Immuniweb Scan')
        scan_environment = Select(driver.find_element(By.ID, 'id_environment'))
        scan_environment.select_by_visible_text('Development')
        driver.find_element(By.ID, 'id_close_old_findings_product_scope').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_1.xml')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='3 findings and closed 0 findings'))

    @on_exception_html_source_logger
    def test_close_same_product_tests(self):
        if False:
            return 10
        logger.debug('Importing reports...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Close Same Product Test 3').click()
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[3].click()
        driver.find_element(By.LINK_TEXT, 'Import Scan Results').click()
        scan_type = Select(driver.find_element(By.ID, 'id_scan_type'))
        scan_type.select_by_visible_text('Immuniweb Scan')
        scan_environment = Select(driver.find_element(By.ID, 'id_environment'))
        scan_environment.select_by_visible_text('Development')
        driver.find_element(By.ID, 'id_close_old_findings_product_scope').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_and_close_1.xml')
        driver.find_elements(By.CLASS_NAME, 'btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='1 findings and closed 2 findings'))

    @on_exception_html_source_logger
    def test_check_same_product_status(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_nb_duplicates(4)

def add_close_old_tests_to_suite(suite, jira=False, github=False, block_execution=False):
    if False:
        print('Hello World!')
    suite.addTest(BaseTestCase('test_login'))
    set_suite_settings(suite, jira=jira, github=github, block_execution=block_execution)
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
    suite.addTest(ProductTest('test_create_product'))
    suite.addTest(CloseOldDedupeTest('test_enable_deduplication'))
    suite.addTest(CloseOldDedupeTest('test_delete_findings'))
    suite.addTest(CloseOldDedupeTest('test_add_same_engagement_engagement'))
    suite.addTest(CloseOldDedupeTest('test_import_same_engagement_tests'))
    suite.addTest(CloseOldDedupeTest('test_close_same_engagement_tests'))
    suite.addTest(CloseOldDedupeTest('test_check_endpoint_status'))
    suite.addTest(CloseOldDedupeTest('test_delete_findings'))
    suite.addTest(CloseOldDedupeTest('test_add_same_product_engagement'))
    suite.addTest(CloseOldDedupeTest('test_import_same_product_tests'))
    suite.addTest(CloseOldDedupeTest('test_close_same_product_tests'))
    suite.addTest(CloseOldDedupeTest('test_check_same_product_status'))
    suite.addTest(ProductTest('test_delete_product'))
    return suite

def suite():
    if False:
        print('Hello World!')
    suite = unittest.TestSuite()
    add_close_old_tests_to_suite(suite, jira=False, github=False, block_execution=False)
    add_close_old_tests_to_suite(suite, jira=True, github=True, block_execution=True)
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)