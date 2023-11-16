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

class DedupeTest(BaseTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.relative_path = dir_path = os.path.dirname(os.path.realpath(__file__))

    def check_nb_duplicates(self, expected_number_of_duplicates):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
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
    def test_add_path_test_suite(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Same scanner deduplication - Deduplication on engagement - static. Creating tests...')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Dedupe Path Test')
        driver.find_element(By.XPATH, '//*[@id="id_deduplication_on_engagement"]').click()
        driver.find_element(By.NAME, '_Add Tests').click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        driver.find_element(By.ID, 'id_title').send_keys('Path Test 1')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Bandit Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.NAME, '_Add Another Test').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))
        driver.find_element(By.ID, 'id_title').send_keys('Path Test 2')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Bandit Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))

    @on_exception_html_source_logger
    def test_import_path_tests(self):
        if False:
            print('Hello World!')
        '\n        Re-upload dedupe_path_1.json bandit report into "Path Test 1" empty test (nothing uploaded before)\n        Then do the same with dedupe_path_2.json / "Path Test 2"\n        '
        logger.debug('importing reports...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Path Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 1').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_path_1.json')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 1 findings'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Path Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 2').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_path_2.json')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 2 findings'))

    @on_exception_html_source_logger
    def test_check_path_status(self):
        if False:
            return 10
        self.check_nb_duplicates(1)

    @on_exception_html_source_logger
    def test_add_endpoint_test_suite(self):
        if False:
            return 10
        logger.debug('Same scanner deduplication - Deduplication on engagement - dynamic. Creating tests...')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Dedupe Endpoint Test')
        driver.find_element(By.XPATH, '//*[@id="id_deduplication_on_engagement"]').click()
        driver.find_element(By.NAME, '_Add Tests').click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        driver.find_element(By.ID, 'id_title').send_keys('Endpoint Test 1')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Immuniweb Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.NAME, '_Add Another Test').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))
        driver.find_element(By.ID, 'id_title').send_keys('Endpoint Test 2')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Immuniweb Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))

    @on_exception_html_source_logger
    def test_import_endpoint_tests(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Importing reports...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Endpoint Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Endpoint Test 1').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_1.xml')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 3 findings'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Endpoint Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Endpoint Test 2').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_2.xml')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 3 findings'))

    @on_exception_html_source_logger
    def test_check_endpoint_status(self):
        if False:
            return 10
        self.check_nb_duplicates(1)

    @on_exception_html_source_logger
    def test_add_same_eng_test_suite(self):
        if False:
            print('Hello World!')
        logger.debug('Test different scanners - same engagement - dynamic; Adding tests on the same engagement...')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Dedupe Same Eng Test')
        driver.find_element(By.XPATH, '//*[@id="id_deduplication_on_engagement"]').click()
        driver.find_element(By.NAME, '_Add Tests').click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        driver.find_element(By.ID, 'id_title').send_keys('Same Eng Test 1')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Immuniweb Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.NAME, '_Add Another Test').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))
        driver.find_element(By.ID, 'id_title').send_keys('Same Eng Test 2')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Generic Findings Import')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))

    @on_exception_html_source_logger
    def test_import_same_eng_tests(self):
        if False:
            while True:
                i = 10
        'Test different scanners - different engagement - dynamic'
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Same Eng Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Same Eng Test 1').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_1.xml')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 3 findings'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Same Eng Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Same Eng Test 2').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_cross_1.csv')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 3 findings'))

    @on_exception_html_source_logger
    def test_check_same_eng_status(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_nb_duplicates(1)

    def test_add_path_test_suite_checkmarx_scan(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Same scanner deduplication - Deduplication on engagement. Test dedupe on checkmarx aggregated with custom hash_code computation')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Dedupe on hash_code only')
        driver.find_element(By.XPATH, '//*[@id="id_deduplication_on_engagement"]').click()
        driver.find_element(By.NAME, '_Add Tests').click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        driver.find_element(By.ID, 'id_title').send_keys('Path Test 1')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Checkmarx Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.NAME, '_Add Another Test').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))
        driver.find_element(By.ID, 'id_title').send_keys('Path Test 2')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Checkmarx Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))

    def test_import_path_tests_checkmarx_scan(self):
        if False:
            return 10
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe on hash_code only').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 1').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(os.path.realpath(self.relative_path + '/dedupe_scans/multiple_findings.xml'))
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 2 findings'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe on hash_code only').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 2').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(os.path.realpath(self.relative_path + '/dedupe_scans/multiple_findings_line_changed.xml'))
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 2 findings'))

    def test_check_path_status_checkmarx_scan(self):
        if False:
            return 10
        self.check_nb_duplicates(2)

    def test_add_cross_test_suite(self):
        if False:
            i = 10
            return i + 15
        logger.debug('Cross scanners deduplication dynamic; generic finding vs immuniweb. Creating tests...')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Dedupe Generic Test')
        driver.find_element(By.NAME, '_Add Tests').click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        driver.find_element(By.ID, 'id_title').send_keys('Generic Test')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Generic Findings Import')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))
        self.goto_product_overview(driver)
        driver.find_element(By.CSS_SELECTOR, '.dropdown-toggle.pull-left').click()
        driver.find_element(By.LINK_TEXT, 'Add New Engagement').click()
        driver.find_element(By.ID, 'id_name').send_keys('Dedupe Immuniweb Test')
        driver.find_element(By.NAME, '_Add Tests').click()
        self.assertTrue(self.is_success_message_present(text='Engagement added successfully.'))
        driver.find_element(By.ID, 'id_title').send_keys('Immuniweb Test')
        Select(driver.find_element(By.ID, 'id_test_type')).select_by_visible_text('Immuniweb Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Test added successfully'))

    def test_import_cross_test(self):
        if False:
            print('Hello World!')
        logger.debug('Importing findings...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Immuniweb Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Immuniweb Test').click()
        driver.find_element(By.CSS_SELECTOR, 'i.fa-solid.fa-ellipsis-vertical').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan Results').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_endpoint_1.xml')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 3 findings'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe Generic Test').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Generic Test').click()
        driver.find_element(By.CSS_SELECTOR, 'i.fa-solid.fa-ellipsis-vertical').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan Results').click()
        driver.find_element(By.ID, 'id_file').send_keys(self.relative_path + '/dedupe_scans/dedupe_cross_1.csv')
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='a total of 3 findings'))

    def test_check_cross_status(self):
        if False:
            print('Hello World!')
        self.check_nb_duplicates(1)

    def test_import_no_service(self):
        if False:
            print('Hello World!')
        logger.debug('Importing findings...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe on hash_code only').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 1').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(os.path.realpath(self.relative_path + '/dedupe_scans/multiple_findings.xml'))
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='Checkmarx Scan processed a total of 2 findings created 2 findings.'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe on hash_code only').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 2').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_file').send_keys(os.path.realpath(self.relative_path + '/dedupe_scans/multiple_findings.xml'))
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='Checkmarx Scan processed a total of 2 findings created 2 findings.'))

    def test_check_no_service(self):
        if False:
            i = 10
            return i + 15
        self.check_nb_duplicates(2)

    def test_import_service(self):
        if False:
            return 10
        logger.debug('Importing findings...')
        driver = self.driver
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe on hash_code only').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 1').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_service').send_keys('service_1')
        driver.find_element(By.ID, 'id_file').send_keys(os.path.realpath(self.relative_path + '/dedupe_scans/multiple_findings.xml'))
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='Checkmarx Scan processed a total of 2 findings created 2 findings.'))
        self.goto_active_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Dedupe on hash_code only').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Path Test 2').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Re-Upload Scan').click()
        driver.find_element(By.ID, 'id_service').send_keys('service_2')
        driver.find_element(By.ID, 'id_file').send_keys(os.path.realpath(self.relative_path + '/dedupe_scans/multiple_findings.xml'))
        driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='Checkmarx Scan processed a total of 2 findings created 2 findings.'))

    def test_check_service(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_nb_duplicates(0)

def add_dedupe_tests_to_suite(suite, jira=False, github=False, block_execution=False):
    if False:
        while True:
            i = 10
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
    suite.addTest(DedupeTest('test_enable_deduplication'))
    suite.addTest(DedupeTest('test_delete_findings'))
    suite.addTest(DedupeTest('test_add_path_test_suite'))
    suite.addTest(DedupeTest('test_import_path_tests'))
    suite.addTest(DedupeTest('test_check_path_status'))
    suite.addTest(DedupeTest('test_delete_findings'))
    suite.addTest(DedupeTest('test_add_endpoint_test_suite'))
    suite.addTest(DedupeTest('test_import_endpoint_tests'))
    suite.addTest(DedupeTest('test_check_endpoint_status'))
    suite.addTest(DedupeTest('test_delete_findings'))
    suite.addTest(DedupeTest('test_add_same_eng_test_suite'))
    suite.addTest(DedupeTest('test_import_same_eng_tests'))
    suite.addTest(DedupeTest('test_check_same_eng_status'))
    suite.addTest(DedupeTest('test_delete_findings'))
    suite.addTest(DedupeTest('test_add_path_test_suite_checkmarx_scan'))
    suite.addTest(DedupeTest('test_import_path_tests_checkmarx_scan'))
    suite.addTest(DedupeTest('test_check_path_status_checkmarx_scan'))
    suite.addTest(DedupeTest('test_delete_findings'))
    suite.addTest(DedupeTest('test_add_cross_test_suite'))
    suite.addTest(DedupeTest('test_import_cross_test'))
    suite.addTest(DedupeTest('test_check_cross_status'))
    suite.addTest(DedupeTest('test_delete_findings'))
    suite.addTest(DedupeTest('test_import_no_service'))
    suite.addTest(DedupeTest('test_check_no_service'))
    suite.addTest(DedupeTest('test_delete_findings'))
    suite.addTest(DedupeTest('test_import_service'))
    suite.addTest(DedupeTest('test_check_service'))
    suite.addTest(ProductTest('test_delete_product'))
    return suite

def suite():
    if False:
        while True:
            i = 10
    suite = unittest.TestSuite()
    add_dedupe_tests_to_suite(suite, jira=False, github=False, block_execution=False)
    add_dedupe_tests_to_suite(suite, jira=True, github=True, block_execution=True)
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)