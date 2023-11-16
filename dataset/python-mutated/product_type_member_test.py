import unittest
import sys
from base_test_class import BaseTestCase
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from user_test import UserTest

class ProductTypeMemberTest(BaseTestCase):

    def test_user_add_product_type_member(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'user')
        driver.find_element(By.LINK_TEXT, 'propersahm').click()
        if self.is_element_by_id_present('dropdownMenuAddProductTypeMember'):
            driver.find_element(By.ID, 'dropdownMenuAddProductTypeMember').click()
            driver.find_element(By.ID, 'addProductTypeMember').click()
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
            self.assertTrue(self.is_success_message_present(text='Product type members added successfully.'))
            self.assertEqual(driver.find_elements(By.NAME, 'member_product_type')[0].text, 'Research and Development')
            self.assertEqual(driver.find_elements(By.NAME, 'member_product_type_role')[0].text, 'Reader')
        else:
            print('--------------------------------')
            print('test_user_add_product_type_member: Not executed because legacy authorization is active')
            print('--------------------------------')

    def test_user_edit_product_type_member(self):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(self.base_url + 'user')
        driver.find_element(By.LINK_TEXT, 'propersahm').click()
        if self.is_element_by_id_present('dropdownMenuAddProductTypeMember'):
            driver.find_elements(By.NAME, 'dropdownManageProductTypeMember')[0].click()
            driver.find_elements(By.NAME, 'editProductTypeMember')[0].click()
            Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Owner')
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
            self.assertTrue(self.is_success_message_present(text='Product type member updated successfully.'))
            self.assertEqual(driver.find_elements(By.NAME, 'member_product_type')[0].text, 'Research and Development')
            self.assertEqual(driver.find_elements(By.NAME, 'member_product_type_role')[0].text, 'Owner')
        else:
            print('--------------------------------')
            print('test_user_edit_product_type_member: Not executed because legacy authorization is active')
            print('--------------------------------')

    def test_user_delete_product_type_member(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'user')
        driver.find_element(By.LINK_TEXT, 'propersahm').click()
        if self.is_element_by_id_present('dropdownMenuAddProductTypeMember'):
            driver.find_elements(By.NAME, 'dropdownManageProductTypeMember')[0].click()
            driver.find_elements(By.NAME, 'deleteProductTypeMember')[0].click()
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
            self.assertTrue(self.is_success_message_present(text='Product type member deleted successfully.'))
            self.assertFalse(driver.find_elements(By.NAME, 'member_product_type'))
        else:
            print('--------------------------------')
            print('test_user_delete_product_type_member: Not executed because legacy authorization is active')
            print('--------------------------------')

    def test_product_type_add_product_type_member(self):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'View').click()
        if self.is_element_by_id_present('dropdownMenuAddProductTypeMember'):
            driver.find_element(By.ID, 'dropdownMenuAddProductTypeMember').click()
            driver.find_element(By.ID, 'addProductTypeMember').click()
            try:
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'id_users')))
            except TimeoutException:
                self.fail('Timed out waiting for users dropdown to initialize ')
            driver.execute_script("document.getElementsByName('users')[0].style.display = 'inline'")
            element = driver.find_element(By.XPATH, "//select[@name='users']")
            user_option = element.find_elements(By.TAG_NAME, 'option')[0]
            Select(element).select_by_value(user_option.get_attribute('value'))
            Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Reader')
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
            self.assertTrue(self.is_success_message_present(text='Product type members added successfully.'))
            self.assertEqual(driver.find_elements(By.NAME, 'member_user')[1].text, 'Proper Samuel (propersahm)')
            self.assertEqual(driver.find_elements(By.NAME, 'member_role')[1].text, 'Reader')
        else:
            print('--------------------------------')
            print('test_product_type_add_product_type_member: Not executed because legacy authorization is active')
            print('--------------------------------')

    def test_product_type_edit_product_type_member(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'View').click()
        if self.is_element_by_id_present('dropdownMenuAddProductTypeMember'):
            driver.find_elements(By.NAME, 'dropdownManageProductTypeMember')[1].click()
            driver.find_elements(By.NAME, 'editProductTypeMember')[1].click()
            Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Maintainer')
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
            self.assertTrue(self.is_success_message_present(text='Product type member updated successfully.'))
            self.assertEqual(driver.find_elements(By.NAME, 'member_user')[1].text, 'Proper Samuel (propersahm)')
            self.assertEqual(driver.find_elements(By.NAME, 'member_role')[1].text, 'Maintainer')
        else:
            print('--------------------------------')
            print('test_product_type_edit_product_type_member: Not executed because legacy authorization is active')
            print('--------------------------------')

    def test_product_type_delete_product_type_member(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'View').click()
        if self.is_element_by_id_present('dropdownMenuAddProductTypeMember'):
            driver.find_elements(By.NAME, 'dropdownManageProductTypeMember')[1].click()
            driver.find_elements(By.NAME, 'deleteProductTypeMember')[1].click()
            driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
            self.assertTrue(self.is_success_message_present(text='Product type member deleted successfully.'))
            self.assertTrue(len(driver.find_elements(By.NAME, 'member_user')) == 1)
        else:
            print('--------------------------------')
            print('test_product_delete_product_member: Not executed because legacy authorization is active')
            print('--------------------------------')

def suite():
    if False:
        i = 10
        return i + 15
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(BaseTestCase('disable_block_execution'))
    suite.addTest(UserTest('test_create_user'))
    suite.addTest(ProductTypeMemberTest('test_user_add_product_type_member'))
    suite.addTest(ProductTypeMemberTest('test_user_edit_product_type_member'))
    suite.addTest(ProductTypeMemberTest('test_user_delete_product_type_member'))
    suite.addTest(ProductTypeMemberTest('test_product_type_add_product_type_member'))
    suite.addTest(ProductTypeMemberTest('test_product_type_edit_product_type_member'))
    suite.addTest(ProductTypeMemberTest('test_product_type_delete_product_type_member'))
    suite.addTest(UserTest('test_user_delete'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)