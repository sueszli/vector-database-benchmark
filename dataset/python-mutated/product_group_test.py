import unittest
import sys
from base_test_class import BaseTestCase
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from group_test import GroupTest
from product_test import ProductTest

class ProductGroupTest(BaseTestCase):

    def test_group_add_product_group(self):
        if False:
            i = 10
            return i + 15
        driver = self.navigate_to_group_view()
        driver.find_element(By.ID, 'dropdownMenuAddProductGroup').click()
        driver.find_element(By.ID, 'addProductGroup').click()
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'id_products')))
        except TimeoutException:
            self.fail('Timed out waiting for products dropdown to initialize ')
        driver.execute_script("document.getElementsByName('products')[0].style.display = 'inline'")
        element = driver.find_element(By.XPATH, "//select[@name='products']")
        product_option = element.find_elements(By.TAG_NAME, 'option')[0]
        Select(element).select_by_value(product_option.get_attribute('value'))
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Reader')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product groups added successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'member_product')[0].text, 'QA Test')
        self.assertEqual(driver.find_elements(By.NAME, 'member_product_role')[0].text, 'Reader')

    def test_group_edit_product_group(self):
        if False:
            return 10
        driver = self.navigate_to_group_view()
        driver.find_elements(By.NAME, 'dropdownManageProductGroup')[0].click()
        driver.find_elements(By.NAME, 'editProductGroup')[0].click()
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Owner')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product group updated successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'member_product')[0].text, 'QA Test')
        self.assertEqual(driver.find_elements(By.NAME, 'member_product_role')[0].text, 'Owner')

    def test_group_delete_product_group(self):
        if False:
            return 10
        driver = self.navigate_to_group_view()
        driver.find_elements(By.NAME, 'dropdownManageProductGroup')[0].click()
        driver.find_elements(By.NAME, 'deleteProductGroup')[0].click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Product group deleted successfully.'))
        self.assertFalse(driver.find_elements(By.NAME, 'member_product'))

    def test_product_add_product_group(self):
        if False:
            return 10
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_element(By.ID, 'dropdownMenuAddProductGroup').click()
        driver.find_element(By.ID, 'addProductGroup').click()
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'id_groups')))
        except TimeoutException:
            self.fail('Timed out waiting for groups dropdown to initialize ')
        driver.execute_script("document.getElementsByName('groups')[0].style.display = 'inline'")
        element = driver.find_element(By.XPATH, "//select[@name='groups']")
        group_option = element.find_elements(By.TAG_NAME, 'option')[0]
        Select(element).select_by_value(group_option.get_attribute('value'))
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Reader')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product groups added successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'group_name')[0].text, 'Group Name')
        self.assertEqual(driver.find_elements(By.NAME, 'group_role')[0].text, 'Reader')

    def test_product_edit_product_group(self):
        if False:
            print('Hello World!')
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_elements(By.NAME, 'dropdownManageProductGroup')[0].click()
        driver.find_elements(By.NAME, 'editProductGroup')[0].click()
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Maintainer')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product group updated successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'group_name')[0].text, 'Group Name')
        self.assertEqual(driver.find_elements(By.NAME, 'group_role')[0].text, 'Maintainer')

    def test_product_delete_product_group(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        self.goto_product_overview(driver)
        driver.find_element(By.LINK_TEXT, 'QA Test').click()
        driver.find_elements(By.NAME, 'dropdownManageProductGroup')[0].click()
        driver.find_elements(By.NAME, 'deleteProductGroup')[0].click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Product group deleted successfully.'))
        self.assertFalse(driver.find_elements(By.NAME, 'group_name'))

    def navigate_to_group_view(self):
        if False:
            return 10
        driver = self.driver
        driver.get(self.base_url + 'group')
        driver.find_element(By.ID, 'show-filters').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Group Name')
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-sm.btn-secondary').click()
        driver.find_element(By.ID, 'dropdownMenuGroup').click()
        driver.find_element(By.ID, 'viewGroup').click()
        return driver

def suite():
    if False:
        while True:
            i = 10
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(GroupTest('test_create_group'))
    suite.addTest(ProductTest('test_create_product'))
    suite.addTest(ProductGroupTest('test_group_add_product_group'))
    suite.addTest(ProductGroupTest('test_group_edit_product_group'))
    suite.addTest(ProductGroupTest('test_group_delete_product_group'))
    suite.addTest(ProductGroupTest('test_product_add_product_group'))
    suite.addTest(ProductGroupTest('test_product_edit_product_group'))
    suite.addTest(ProductGroupTest('test_product_delete_product_group'))
    suite.addTest(GroupTest('test_group_edit_name_and_global_role'))
    suite.addTest(GroupTest('test_group_delete'))
    suite.addTest(ProductTest('test_delete_product'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)