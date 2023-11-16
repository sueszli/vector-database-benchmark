"""Integration tests for interactive beam."""
import unittest
import pytest
from apache_beam.runners.interactive.testing.integration.screen_diff import BaseTestCase

@pytest.mark.timeout(300)
class DataFramesTest(BaseTestCase):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        kwargs['golden_size'] = (1024, 10000)
        super().__init__(*args, **kwargs)

    def explicit_wait(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions
            from selenium.webdriver.support.ui import WebDriverWait
            WebDriverWait(self.driver, 5).until(expected_conditions.presence_of_element_located((By.ID, 'test-done')))
        except:
            pass

    def test_dataframes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_notebook('dataframes')

@pytest.mark.timeout(300)
class InitSquareCubeTest(BaseTestCase):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['golden_size'] = (1024, 10000)
        super().__init__(*args, **kwargs)

    def test_init_square_cube_notebook(self):
        if False:
            return 10
        self.assert_notebook('init_square_cube')
if __name__ == '__main__':
    unittest.main()