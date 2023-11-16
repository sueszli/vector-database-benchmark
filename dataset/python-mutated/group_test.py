import unittest
import sys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from base_test_class import BaseTestCase
from user_test import UserTest

class GroupTest(BaseTestCase):

    def test_create_group(self):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(self.base_url + 'group')
        driver.find_element(By.ID, 'dropdownMenu1').click()
        driver.find_element(By.LINK_TEXT, 'New Group').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Group Name')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Group was added successfully.'))

    def test_group_edit_name_and_global_role(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'group')
        driver.find_element(By.ID, 'show-filters').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Group Name')
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-sm.btn-secondary').click()
        driver.find_element(By.ID, 'dropdownMenuGroup').click()
        driver.find_element(By.ID, 'editGroup').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Another Name')
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Reader')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Group saved successfully.'))

    def test_add_group_member(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        driver.get(self.base_url + 'group')
        driver.find_element(By.LINK_TEXT, 'Another Name').click()
        driver.find_element(By.ID, 'dropdownMenuAddGroupMember').click()
        driver.find_element(By.ID, 'addGroupMember').click()
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'id_users')))
        except TimeoutException:
            self.fail('Timed out waiting for products dropdown to initialize ')
        driver.execute_script("document.getElementsByName('users')[0].style.display = 'inline'")
        element = driver.find_element(By.XPATH, "//select[@name='users']")
        user_option = element.find_elements(By.TAG_NAME, 'option')[0]
        Select(element).select_by_value(user_option.get_attribute('value'))
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Reader')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Group members added successfully.'))
        self.assertEqual(driver.find_elements(By.NAME, 'member_user')[1].text, 'Proper Samuel (propersahm)')
        self.assertEqual(driver.find_elements(By.NAME, 'member_role')[1].text, 'Reader')

    def test_edit_group_member(self):
        if False:
            i = 10
            return i + 15
        driver = self.driver
        driver.get(self.base_url + 'group')
        driver.find_element(By.LINK_TEXT, 'Another Name').click()
        driver.find_elements(By.NAME, 'dropdownManageGroupMembers')[1].click()
        driver.find_elements(By.NAME, 'editGroupMember')[1].click()
        Select(driver.find_element(By.ID, 'id_role')).select_by_visible_text('Maintainer')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()
        self.assertTrue(self.is_success_message_present(text='Group member updated successfully'))
        self.assertEqual(driver.find_elements(By.NAME, 'member_user')[1].text, 'Proper Samuel (propersahm)')
        self.assertEqual(driver.find_elements(By.NAME, 'member_role')[1].text, 'Maintainer')

    def test_delete_group_member(self):
        if False:
            while True:
                i = 10
        driver = self.driver
        driver.get(self.base_url + 'group')
        driver.find_element(By.LINK_TEXT, 'Another Name').click()
        driver.find_elements(By.NAME, 'dropdownManageGroupMembers')[1].click()
        driver.find_elements(By.NAME, 'deleteGroupMember')[1].click()
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Group member deleted successfully.'))

    def test_group_delete(self):
        if False:
            print('Hello World!')
        driver = self.driver
        driver.get(self.base_url + 'group')
        driver.find_element(By.ID, 'show-filters').click()
        driver.find_element(By.ID, 'id_name').clear()
        driver.find_element(By.ID, 'id_name').send_keys('Another Name')
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-sm.btn-secondary').click()
        driver.find_element(By.ID, 'dropdownMenuGroup').click()
        driver.find_element(By.ID, 'deleteGroup').click()
        driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-danger').click()
        self.assertTrue(self.is_success_message_present(text='Group and relationships successfully removed.'))

    def test_group_edit_configuration(self):
        if False:
            return 10
        driver = self.driver
        self.login_standard_page()
        with self.assertRaises(NoSuchElementException):
            driver.find_element(By.ID, 'id_group_menu')
        self.login_page()
        driver.get(self.base_url + 'group')
        driver.find_element(By.LINK_TEXT, 'Another Name').click()
        driver.find_element(By.ID, 'id_view_group').click()
        self.login_standard_page()
        driver.find_element(By.ID, 'id_group_menu')
        driver.get(self.base_url + 'group')
        driver.find_element(By.LINK_TEXT, 'Another Name').click()
        self.assertFalse(self.driver.find_element(By.ID, 'id_add_development_environment').is_enabled())

def suite():
    if False:
        print('Hello World!')
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(UserTest('test_create_user'))
    suite.addTest(GroupTest('test_create_group'))
    suite.addTest(GroupTest('test_group_edit_name_and_global_role'))
    suite.addTest(GroupTest('test_add_group_member'))
    suite.addTest(GroupTest('test_group_edit_configuration'))
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(GroupTest('test_edit_group_member'))
    suite.addTest(GroupTest('test_delete_group_member'))
    suite.addTest(GroupTest('test_group_delete'))
    suite.addTest(UserTest('test_user_delete'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)