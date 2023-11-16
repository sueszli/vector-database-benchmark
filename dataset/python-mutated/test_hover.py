from helium import hover, Config
from helium._impl.util.lang import TemporaryAttrValue
from helium._impl.util.system import is_windows
from tests.api import BrowserAT

class HoverTest(BrowserAT):

    def get_page(self):
        if False:
            i = 10
            return i + 15
        return 'test_hover.html'

    def setUp(self):
        if False:
            print('Hello World!')
        self._move_mouse_cursor_to_origin()
        super().setUp()

    def _move_mouse_cursor_to_origin(self):
        if False:
            for i in range(10):
                print('nop')
        if is_windows():
            from win32api import SetCursorPos
            SetCursorPos((0, 0))

    def test_hover_one(self):
        if False:
            i = 10
            return i + 15
        hover('Dropdown 1')
        result = self.read_result_from_browser()
        self.assertEqual('Dropdown 1', result, 'Got unexpected result %r. Maybe the mouse cursor was over the browser window and interfered with the test?' % result)

    def test_hover_two_consecutively(self):
        if False:
            for i in range(10):
                print('nop')
        hover('Dropdown 2')
        hover('Item C')
        result = self.read_result_from_browser()
        self.assertEqual('Dropdown 2 - Item C', result, 'Got unexpected result %r. Maybe the mouse cursor was over the browser window and interfered with the test?' % result)

    def test_hover_hidden(self):
        if False:
            while True:
                i = 10
        with TemporaryAttrValue(Config, 'implicit_wait_secs', 1):
            try:
                hover('Item C')
            except LookupError:
                pass
            else:
                self.fail("Didn't receive expected LookupError. Maybe the mouse cursor was over the browser window and interfered with the test?")