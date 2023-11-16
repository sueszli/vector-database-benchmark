from .base_test_case import BaseTestCase
from .page_objects import Page

class MyTestClass(BaseTestCase):

    def test_boilerplate(self):
        if False:
            while True:
                i = 10
        self.login()
        self.example_method()
        self.assert_element(Page.html)