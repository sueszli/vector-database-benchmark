from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import unittest
import sys
import os
from base_test_class import BaseTestCase, on_exception_html_source_logger, set_suite_settings
from product_test import ProductTest, WaitForPageLoad
from user_test import UserTest
from pathlib import Path
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

class FindingTest(BaseTestCase):

    def test_list_findings_all(self):
        if False:
            while True:
                i = 10
        return self.test_list_findings('finding')

    def test_list_findings_closed(self):
        if False:
            print('Hello World!')
        return self.test_list_findings('finding/closed')

    def test_list_findings_accepted(self):
        if False:
            return 10
        return self.test_list_findings('finding/accepted')

    def test_list_findings_open(self):
        if False:
            while True:
                i = 10
        return self.test_list_findings('finding/open')

    def test_list_findings(self, suffix):
        if False:
            return 10
        driver = self.driver
        driver.get(self.base_url + suffix)
        driver.find_element(By.ID, 'select_all').click()
        driver.find_element(By.ID, 'dropdownMenu2').click()
        bulk_edit_menu = driver.find_element(By.ID, 'bulk_edit_menu')
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_active').is_enabled(), False)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_verified').is_enabled(), False)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_false_p').is_enabled(), False)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_out_of_scope').is_enabled(), False)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_is_mitigated').is_enabled(), False)
        driver.find_element(By.ID, 'id_bulk_status').click()
        bulk_edit_menu = driver.find_element(By.ID, 'bulk_edit_menu')
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_active').is_enabled(), True)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_verified').is_enabled(), True)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_false_p').is_enabled(), True)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_out_of_scope').is_enabled(), True)
        self.assertEqual(bulk_edit_menu.find_element(By.ID, 'id_bulk_is_mitigated').is_enabled(), True)

    def test_quick_report(self):
        if False:
            return 10
        driver = self.driver
        driver.get(self.base_url + 'finding')
        driver.find_element(By.ID, 'downloadMenu').click()
        driver.find_element(By.ID, 'report').click()
        self.assertIn('<title>Finding Report</title>', driver.page_source)

    def check_file(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        file_found = False
        for i in range(1, 30):
            time.sleep(1)
            if Path(file_name).is_file():
                file_found = True
                break
        self.assertTrue(file_found, f'Cannot find {file_name}')
        os.remove(file_name)

    def test_csv_export(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'finding')
        driver.find_element(By.ID, 'downloadMenu').click()
        driver.find_element(By.ID, 'csv_export').click()
        time.sleep(5)
        self.check_file(f'{self.export_path}/findings.csv')

    def test_excel_export(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'finding')
        driver.find_element(By.ID, 'downloadMenu').click()
        driver.find_element(By.ID, 'excel_export').click()
        time.sleep(5)
        self.check_file(f'{self.export_path}/findings.xlsx')

    @on_exception_html_source_logger
    def test_edit_finding(self):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Edit Finding').click()
        Select(driver.find_element(By.ID, 'id_severity')).select_by_visible_text('Critical')
        driver.find_element(By.ID, 'id_cvssv3').send_keys('CVSS:3.0/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H')
        driver.find_element(By.ID, 'id_vulnerability_ids').send_keys('\nREF-3\nREF-4\n')
        driver.find_element(By.XPATH, "//input[@name='_Finished']").click()
        self.assertTrue(self.is_success_message_present(text='Finding saved successfully'))
        self.assertTrue(self.is_text_present_on_page(text='REF-1'))
        self.assertTrue(self.is_text_present_on_page(text='REF-2'))
        self.assertTrue(self.is_text_present_on_page(text='REF-3'))
        self.assertTrue(self.is_text_present_on_page(text='REF-4'))
        self.assertTrue(self.is_text_present_on_page(text='Additional Vulnerability Ids'))

    def test_add_image(self):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Manage Files').click()
        image_path = os.path.join(dir_path, 'finding_image.png')
        driver.find_element(By.ID, 'id_form-0-file').send_keys(image_path)
        driver.find_element(By.ID, 'id_form-0-title').send_keys('Image Title')
        with WaitForPageLoad(driver, timeout=50):
            driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-success').click()
        self.assertTrue(self.is_success_message_present(text='Files updated successfully.'))

    @on_exception_html_source_logger
    def test_add_note_to_finding(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'id_entry').clear()
        driver.find_element(By.ID, 'id_entry').send_keys('This is a sample note for all to see.')
        driver.find_element(By.XPATH, "//input[@value='Add Note']").click()
        self.assertTrue(self.is_success_message_present(text='Note saved.'))

    def test_mark_finding_for_review(self):
        if False:
            return 10
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Request Peer Review').click()
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'id_reviewers')))
        except TimeoutException:
            self.fail('Timed out waiting for reviewer dropdown to initialize ')
        driver.execute_script("document.getElementsByName('reviewers')[0].style.display = 'inline'")
        element = driver.find_element(By.XPATH, "//select[@name='reviewers']")
        reviewer_option = element.find_elements(By.TAG_NAME, 'option')[0]
        Select(element).select_by_value(reviewer_option.get_attribute('value'))
        driver.find_element(By.ID, 'id_entry').clear()
        driver.find_element(By.ID, 'id_entry').send_keys('This is to be reviewed critically. Make sure it is well handled.')
        driver.find_element(By.NAME, 'submit').click()
        self.assertTrue(self.is_success_message_present(text='Finding marked for review and reviewers notified.'))

    @on_exception_html_source_logger
    def test_clear_review_from_finding(self):
        if False:
            return 10
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.LINK_TEXT, 'Clear Review').click()
        driver.find_element(By.ID, 'id_active').click()
        driver.find_element(By.ID, 'id_verified').click()
        driver.find_element(By.ID, 'id_entry').clear()
        driver.find_element(By.ID, 'id_entry').send_keys('This has been reviewed and confirmed. A fix needed here.')
        driver.find_element(By.NAME, 'submit').click()
        self.assertTrue(self.is_success_message_present(text='Finding review has been updated successfully.'))

    def test_delete_image(self):
        if False:
            return 10
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Manage Files').click()
        driver.find_element(By.ID, 'id_form-0-DELETE').click()
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-success').click()
        self.assertTrue(self.is_success_message_present(text='Files updated successfully.'))

    def test_close_finding(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        pre_status = driver.find_element(By.XPATH, '//*[@id="vuln_endpoints"]/tbody/tr/td[3]').text
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Close Finding').click()
        driver.find_element(By.ID, 'id_entry').send_keys('All issues in this Finding have been resolved successfully')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Finding closed.'))
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        post_status = driver.find_element(By.XPATH, '//*[@id="remd_endpoints"]/tbody/tr/td[3]').text
        self.assertTrue(pre_status != post_status)

    def test_open_finding(self):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        pre_status = driver.find_element(By.XPATH, '//*[@id="remd_endpoints"]/tbody/tr/td[3]').text
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Open Finding').click()
        self.assertTrue(self.is_success_message_present(text='Finding Reopened.'))
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        post_status = driver.find_element(By.XPATH, '//*[@id="vuln_endpoints"]/tbody/tr/td[3]').text
        self.assertTrue(pre_status != post_status)

    @on_exception_html_source_logger
    def test_simple_accept_finding(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        pre_status = driver.find_element(By.XPATH, '//*[@id="vuln_endpoints"]/tbody/tr/td[3]').text
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Accept Risk').click()
        self.assertTrue(self.is_success_message_present(text='Finding risk accepted.'))
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()

    def test_unaccept_finding(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        pre_status = driver.find_element(By.XPATH, '//*[@id="vuln_endpoints"]/tbody/tr/td[3]').text
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Unaccept Risk').click()
        self.assertTrue(self.is_success_message_present(text='Finding risk unaccepted.'))
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()

    def test_make_finding_a_template(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Make Finding a Template').click()
        self.assertTrue(self.is_success_message_present(text='Finding template added successfully. You may edit it here.'))

    def test_apply_template_to_a_finding(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        print('\nListing findings \n')
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        self.assertNoConsoleErrors()
        driver.find_element(By.LINK_TEXT, 'Apply Template to Finding').click()
        self.assertNoConsoleErrors()
        print('\nClicking on the template \n')
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        self.assertNoConsoleErrors()
        print('\nClicking on replace all \n')
        driver.find_element(By.XPATH, "//button[@data-option='Replace']").click()
        self.assertNoConsoleErrors()
        driver.find_element(By.NAME, '_Finished').click()
        self.assertNoConsoleErrors()
        self.assertTrue(self.is_text_present_on_page(text='App Vulnerable to XSS'))

    @on_exception_html_source_logger
    def test_create_finding_from_template(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_all_engagements_overview(driver)
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Ad Hoc Engagement').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Pen Test').click()
        driver.find_element(By.ID, 'dropdownMenu_test_add').click()
        self.assertNoConsoleErrors()
        driver.find_element(By.LINK_TEXT, 'Finding From Template').click()
        self.assertNoConsoleErrors()
        print('\nClicking on the template \n')
        driver.find_element(By.LINK_TEXT, 'Use This Template').click()
        self.assertNoConsoleErrors()
        driver.find_element(By.ID, 'id_title').clear()
        driver.find_element(By.ID, 'id_title').send_keys('App Vulnerable to XSS from Template')
        self.assertNoConsoleErrors()
        driver.find_element(By.ID, 'id_finished').click()
        self.assertNoConsoleErrors()
        self.assertTrue(self.is_success_message_present(text='Finding from template added successfully.'))
        self.assertTrue(self.is_text_present_on_page(text='App Vulnerable to XSS From Template'))

    @on_exception_html_source_logger
    def test_delete_finding_template(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        driver.get(self.base_url + 'template')
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.XPATH, "//button[text()='Delete Template']").click()
        driver.switch_to.alert.accept()
        self.assertTrue(self.is_success_message_present(text='Finding Template deleted successfully.'))

    def test_import_scan_result(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Findings').click()
        driver.find_element(By.LINK_TEXT, 'Import Scan Results').click()
        Select(driver.find_element(By.ID, 'id_scan_type')).select_by_visible_text('ZAP Scan')
        Select(driver.find_element(By.ID, 'id_environment')).select_by_visible_text('Development')
        file_path = os.path.join(dir_path, 'zap_sample.xml')
        driver.find_element(By.NAME, 'file').send_keys(file_path)
        with WaitForPageLoad(driver, timeout=50):
            driver.find_elements(By.CSS_SELECTOR, 'button.btn.btn-primary')[1].click()
        self.assertTrue(self.is_success_message_present(text='ZAP Scan processed a total of 4 findings'))

    @on_exception_html_source_logger
    def test_delete_finding(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        self.goto_all_findings_list(driver)
        driver.find_element(By.LINK_TEXT, 'App Vulnerable to XSS').click()
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Delete Finding').click()
        driver.switch_to.alert.accept()
        self.assertTrue(self.is_text_present_on_page(text='Finding deleted successfully'))

    def test_list_components(self):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_component_overview(driver)
        self.assertTrue(self.is_element_by_css_selector_present('table'))

def add_finding_tests_to_suite(suite, jira=False, github=False, block_execution=False):
    if False:
        return 10
    suite.addTest(BaseTestCase('test_login'))
    set_suite_settings(suite, jira=jira, github=github, block_execution=block_execution)
    suite.addTest(BaseTestCase('delete_finding_template_if_exists'))
    suite.addTest(ProductTest('test_create_product'))
    suite.addTest(ProductTest('test_add_product_finding'))
    suite.addTest(UserTest('test_create_user_with_writer_global_role'))
    suite.addTest(FindingTest('test_list_findings_all'))
    suite.addTest(FindingTest('test_list_findings_open'))
    suite.addTest(FindingTest('test_quick_report'))
    suite.addTest(FindingTest('test_csv_export'))
    suite.addTest(FindingTest('test_excel_export'))
    suite.addTest(FindingTest('test_list_components'))
    suite.addTest(FindingTest('test_edit_finding'))
    suite.addTest(FindingTest('test_add_note_to_finding'))
    suite.addTest(FindingTest('test_add_image'))
    suite.addTest(FindingTest('test_delete_image'))
    suite.addTest(FindingTest('test_mark_finding_for_review'))
    suite.addTest(FindingTest('test_clear_review_from_finding'))
    suite.addTest(FindingTest('test_close_finding'))
    suite.addTest(FindingTest('test_list_findings_closed'))
    suite.addTest(FindingTest('test_open_finding'))
    suite.addTest(ProductTest('test_enable_simple_risk_acceptance'))
    suite.addTest(FindingTest('test_simple_accept_finding'))
    suite.addTest(FindingTest('test_list_findings_accepted'))
    suite.addTest(FindingTest('test_list_findings_all'))
    suite.addTest(FindingTest('test_unaccept_finding'))
    suite.addTest(FindingTest('test_make_finding_a_template'))
    suite.addTest(FindingTest('test_apply_template_to_a_finding'))
    suite.addTest(FindingTest('test_create_finding_from_template'))
    suite.addTest(FindingTest('test_import_scan_result'))
    suite.addTest(FindingTest('test_delete_finding'))
    suite.addTest(FindingTest('test_delete_finding_template'))
    suite.addTest(ProductTest('test_delete_product'))
    suite.addTest(UserTest('test_user_with_writer_role_delete'))
    return suite

def suite():
    if False:
        i = 10
        return i + 15
    suite = unittest.TestSuite()
    add_finding_tests_to_suite(suite, jira=False, github=False, block_execution=False)
    add_finding_tests_to_suite(suite, jira=True, github=True, block_execution=True)
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)