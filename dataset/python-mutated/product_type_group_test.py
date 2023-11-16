import unittest
import sys
from base_test_class import BaseTestCase
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from group_test import GroupTest

class ProductTypeGroupTest(BaseTestCase):

    def test_group_add_product_type_group(self):
        if False:
            while True:
                i = 10
        driver = self.navigate_to_group_view()
        driver.find_element(By.ID, 'dropdownMenuAddProductTypeGroup').click()
        driver.find_element(By.ID, 'addProductTypeGroup').click()
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'id_product_types')))
        except TimeoutException:
            self.fail('Timed out waiting for product types dropdown to initialize ')
        driver.execute_script("document.getElementsByName('product_types')[0].style.display = 'inline'")
        element = driver.find_element(By.XPATH, "//select[@name='product_types']")
        product_type_option = element.find_elements(By.TAG_NAME, 'option')[0]
        Select(element).select_by_value(product_type_option.get_attribute('value'))
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Reader')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product type groups added successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'member_product_type')[0].text, 'Research and Development')
        self.assertEqual(driver.find_elements(By.NAME, 'member_product_type_role')[0].text, 'Reader')

    def test_group_edit_product_type_group(self):
        if False:
            print('Hello World!')
        driver = self.navigate_to_group_view()
        driver.find_elements(By.NAME, 'dropdownManageProductTypeGroup')[0].click()
        driver.find_elements(By.NAME, 'editProductTypeGroup')[0].click()
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Owner')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product type group updated successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'member_product_type')[0].text, 'Research and Development')
        self.assertEqual(driver.find_elements(By.NAME, 'member_product_type_role')[0].text, 'Owner')

    def test_group_delete_product_type_group(self):
        if False:
            while True:
                i = 10
        driver = self.navigate_to_group_view()
        driver.find_elements(By.NAME, 'dropdownManageProductTypeGroup')[0].click()
        driver.find_elements(By.NAME, 'deleteProductTypeGroup')[0].click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Product type group deleted successfully.'))
        self.assertFalse(driver.find_elements(By.NAME, 'member_product_type'))

    def test_product_type_add_product_type_group(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'View').click()
        driver.find_element(By.ID, 'dropdownMenuAddProductTypeGroup').click()
        driver.find_element(By.ID, 'addProductTypeGroup').click()
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
        self.assertTrue(self.is_success_message_present(text='Product type groups added successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'product_type_group_group')[0].text, 'Group Name')
        self.assertEqual(driver.find_elements(By.NAME, 'product_type_group_role')[0].text, 'Reader')

    def test_product_type_edit_product_type_group(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'View').click()
        driver.find_elements(By.NAME, 'dropdownManageProductTypeGroup')[0].click()
        driver.find_elements(By.NAME, 'editProductTypeGroup')[0].click()
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Maintainer')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product type group updated successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'product_type_group_group')[0].text, 'Group Name')
        self.assertEqual(driver.find_elements(By.NAME, 'product_type_group_role')[0].text, 'Maintainer')

    def test_product_type_delete_product_type_group(self):
        if False:
            return 10
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'View').click()
        driver.find_elements(By.NAME, 'dropdownManageProductTypeGroup')[0].click()
        driver.find_elements(By.NAME, 'deleteProductTypeGroup')[0].click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Product type group deleted successfully.'))
        self.assertFalse(driver.find_elements(By.NAME, 'product_type_group_group'))

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
        print('Hello World!')
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(GroupTest('test_create_group'))
    suite.addTest(ProductTypeGroupTest('test_group_add_product_type_group'))
    suite.addTest(ProductTypeGroupTest('test_group_edit_product_type_group'))
    suite.addTest(ProductTypeGroupTest('test_group_delete_product_type_group'))
    suite.addTest(ProductTypeGroupTest('test_product_type_add_product_type_group'))
    suite.addTest(ProductTypeGroupTest('test_product_type_edit_product_type_group'))
    suite.addTest(ProductTypeGroupTest('test_product_type_delete_product_type_group'))
    suite.addTest(GroupTest('test_group_edit_name_and_global_role'))
    suite.addTest(GroupTest('test_group_delete'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)