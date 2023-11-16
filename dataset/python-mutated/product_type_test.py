import unittest
import sys
from base_test_class import BaseTestCase, on_exception_html_source_logger
from selenium.webdriver.common.by import By

class ProductTypeTest(BaseTestCase):

    @on_exception_html_source_logger
    def test_create_product_type(self):
        if False:
            while True:
                i = 10
        print("\n\nDebug Print Log: testing 'create product type' \n")
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'Add Product Type').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Product test type')
        driver.find_element(By.ID, 'id_critical_product').click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product type added successfully.'))
        self.assertFalse(self.is_error_message_present())

    @on_exception_html_source_logger
    def test_create_product_for_product_type(self):
        if False:
            return 10
        self.delete_product_if_exists('QA Test PT')
        driver = self.driver
        self.goto_product_type_overview(driver)
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Add Product').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('QA Test PT')
        driver.find_element(By.ID, 'id_name').send_keys('\tThis is just a test. Be very afraid.')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product added successfully'))
        self.assertFalse(self.is_error_message_present())

    def test_view_product_type(self):
        if False:
            while True:
                i = 10
        print("\n\nDebug Print Log: testing 'view product type' \n")
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'View').click()
        product_type_text = driver.find_element(By.ID, 'id_heading').text
        self.assertEqual('Product Type Product test type', product_type_text)

    def test_edit_product_type(self):
        if False:
            return 10
        print("\n\nDebug Print Log: testing 'edit product type' \n")
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Edit').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Edited product test type')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Product type updated successfully.'))

    def test_delete_product_type(self):
        if False:
            i = 10
            return i + 15
        print("\n\nDebug Print Log: testing 'delete product type' \n")
        driver = self.driver
        driver.get(self.base_url + 'product/type')
        driver.find_element(By.ID, 'dropdownMenuProductType').click()
        driver.find_element(By.PARTIAL_LINK_TEXT, 'Delete').click()
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Product Type and relationships removed.'))

def suite():
    if False:
        while True:
            i = 10
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(BaseTestCase('disable_block_execution'))
    suite.addTest(ProductTypeTest('test_create_product_type'))
    suite.addTest(ProductTypeTest('test_view_product_type'))
    suite.addTest(ProductTypeTest('test_create_product_for_product_type'))
    suite.addTest(ProductTypeTest('test_edit_product_type'))
    suite.addTest(ProductTypeTest('test_delete_product_type'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)