from helium import *
from selenium.webdriver.common.by import By
from tests.api import BrowserAT

class DragTest(BrowserAT):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.drag_target = self.driver.find_element(By.ID, 'target')

    def get_page(self):
        if False:
            return 10
        return 'test_drag/default.html'

    def test_drag(self):
        if False:
            for i in range(10):
                print('nop')
        print(Text('Drag me.').exists())
        drag('Drag me.', to=self.drag_target)
        self.assertEqual('Success!', self.read_result_from_browser())

    def test_drag_to_point(self):
        if False:
            while True:
                i = 10
        target_loc = self.drag_target.location
        target_size = self.drag_target.size
        target_point = Point(target_loc['x'] + target_size['width'] / 2, target_loc['y'] + target_size['height'] / 2)
        self.assertTrue(Text('Drag me').exists())
        drag('Drag me.', to=target_point)
        self.assertEqual('Success!', self.read_result_from_browser())

class Html5DragIT(BrowserAT):

    def get_page(self):
        if False:
            print('Hello World!')
        return 'test_drag/html5.html'

    def test_html5_drag(self):
        if False:
            for i in range(10):
                print('nop')
        drag('Drag me.', to=self.driver.find_element(By.ID, 'target'))
        self.assertEqual('Success!', self.read_result_from_browser())