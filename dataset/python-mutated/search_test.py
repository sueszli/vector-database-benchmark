import unittest
import sys
from base_test_class import BaseTestCase
from selenium.webdriver.common.by import By

class SearchTests(BaseTestCase):

    def test_login(self):
        if False:
            return 10
        driver = self.driver

    def test_search(self):
        if False:
            print('Hello World!')
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('finding')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_vulnerability_id(self):
        if False:
            print('Hello World!')
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('vulnerability_id:CVE-2020-12345')
        driver.find_element(By.ID, 'simple_search_submit').click()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('CVE-2020-12345')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_tag(self):
        if False:
            while True:
                i = 10
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('tag:magento')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_product_tag(self):
        if False:
            return 10
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('product-tag:java')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_engagement_tag(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('engagement-tag:php')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_test_tag(self):
        if False:
            for i in range(10):
                print('nop')
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('test-tag:go')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_tags(self):
        if False:
            print('Hello World!')
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('tags:php')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_product_tags(self):
        if False:
            return 10
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('product-tags:java')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_engagement_tags(self):
        if False:
            i = 10
            return i + 15
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('engagement-tags:php')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_test_tags(self):
        if False:
            print('Hello World!')
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('test-tags:go')
        driver.find_element(By.ID, 'simple_search_submit').click()

    def test_search_id(self):
        if False:
            return 10
        driver = self.goto_some_page()
        driver.find_element(By.ID, 'simple_search').clear()
        driver.find_element(By.ID, 'simple_search').send_keys('id:1')
        driver.find_element(By.ID, 'simple_search_submit').click()

def suite():
    if False:
        print('Hello World!')
    suite = unittest.TestSuite()
    suite.addTest(BaseTestCase('test_login'))
    suite.addTest(BaseTestCase('disable_block_execution'))
    suite.addTest(SearchTests('test_search'))
    suite.addTest(SearchTests('test_search_vulnerability_id'))
    suite.addTest(SearchTests('test_search_tag'))
    suite.addTest(SearchTests('test_search_product_tag'))
    suite.addTest(SearchTests('test_search_engagement_tag'))
    suite.addTest(SearchTests('test_search_test_tag'))
    suite.addTest(SearchTests('test_search_tags'))
    suite.addTest(SearchTests('test_search_product_tags'))
    suite.addTest(SearchTests('test_search_engagement_tags'))
    suite.addTest(SearchTests('test_search_test_tags'))
    suite.addTest(SearchTests('test_search_id'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner(descriptions=True, failfast=True, verbosity=2)
    ret = not runner.run(suite()).wasSuccessful()
    BaseTestCase.tearDownDriver()
    sys.exit(ret)