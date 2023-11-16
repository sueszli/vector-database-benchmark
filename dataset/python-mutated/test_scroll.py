from helium import scroll_down, scroll_left, scroll_right, scroll_up
from tests.api import BrowserAT

class ScrollTest(BrowserAT):

    def get_page(self):
        if False:
            for i in range(10):
                print('nop')
        return 'test_scroll.html'

    def test_scroll_up_when_at_top_of_page(self):
        if False:
            print('Hello World!')
        scroll_up()
        self.assert_scroll_position_equals(0, 0)

    def test_scroll_down(self):
        if False:
            return 10
        scroll_down()
        self.assert_scroll_position_equals(0, 100)

    def test_scroll_down_then_up(self):
        if False:
            for i in range(10):
                print('nop')
        scroll_down()
        scroll_up()
        self.assert_scroll_position_equals(0, 0)

    def test_scroll_down_then_up_pixels(self):
        if False:
            print('Hello World!')
        scroll_down(175)
        scroll_up(100)
        self.assert_scroll_position_equals(0, 75)

    def test_scroll_left_when_at_start_of_page(self):
        if False:
            return 10
        scroll_left()
        self.assert_scroll_position_equals(0, 0)

    def test_scroll_right(self):
        if False:
            return 10
        scroll_right()
        self.assert_scroll_position_equals(100, 0)

    def test_scroll_right_then_left(self):
        if False:
            print('Hello World!')
        scroll_right()
        scroll_left()
        self.assert_scroll_position_equals(0, 0)

    def test_scroll_right_then_left_pixels(self):
        if False:
            i = 10
            return i + 15
        scroll_right(175)
        scroll_left(100)
        self.assert_scroll_position_equals(75, 0)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.driver.execute_script('window.scrollTo(0, 0);')
        super().tearDown()

    def assert_scroll_position_equals(self, x, y):
        if False:
            i = 10
            return i + 15
        scroll_position_x = self.driver.execute_script('return window.pageXOffset || document.documentElement.scrollLeft || document.body.scrollLeft')
        self.assertEqual(x, scroll_position_x)
        scroll_position_y = self.driver.execute_script('return window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop')
        self.assertEqual(y, scroll_position_y)