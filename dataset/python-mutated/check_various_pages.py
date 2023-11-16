import unittest
from base_test_class import BaseTestCase
from selenium.webdriver.common.by import By
import sys

class VariousPagesTest(BaseTestCase):

    def test_user_status(self):
        if False:
            return 10
        driver = self.driver
        driver.get(self.base_url + 'user')

    def test_calendar_status(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.driver
        driver.get(self.base_url + 'calendar')
        driver.find_element(By.CSS_SELECTOR, 'input.btn.btn-primary').click()

def suite():
    if False:
        return 10
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(VariousPagesTest('test_user_status'))
    suite.addTest(VariousPagesTest('test_calendar_status'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)