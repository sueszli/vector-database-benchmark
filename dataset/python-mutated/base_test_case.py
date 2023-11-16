"""Use this as a boilerplate for your test framework.
Define customized library methods in a class like this.
Then have your test classes inherit it.
BaseTestCase inherits SeleniumBase methods from BaseCase."""
from seleniumbase import BaseCase

class BaseTestCase(BaseCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.save_teardown_screenshot()
        if self.has_exception():
            pass
        else:
            pass
        super().tearDown()

    def login(self):
        if False:
            print('Hello World!')
        pass

    def example_method(self):
        if False:
            print('Hello World!')
        pass
'\n# Now you can do something like this in your test files:\n\nfrom base_test_case import BaseTestCase\n\nclass MyTests(BaseTestCase):\n\n    def test_example(self):\n        self.login()\n        self.example_method()\n        self.type("input", "Name")\n        self.click("form button")\n        ...\n'