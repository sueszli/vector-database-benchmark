from helium._impl.util.geom import Rectangle
from selenium.common.exceptions import StaleElementReferenceException, NoSuchFrameException, WebDriverException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from urllib.error import URLError
import sys

class Wrapper:

    def __init__(self, target):
        if False:
            i = 10
            return i + 15
        self.target = target

    def __getattr__(self, item):
        if False:
            i = 10
            return i + 15
        return getattr(self.target, item)

    def unwrap(self):
        if False:
            return 10
        return self.target

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.target)

    def __eq__(self, other):
        if False:
            return 10
        return self.target == other.target

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

class WebDriverWrapper(Wrapper):

    def __init__(self, target):
        if False:
            i = 10
            return i + 15
        super(WebDriverWrapper, self).__init__(target)
        self.last_manipulated_element = None

    def action(self):
        if False:
            return 10
        return ActionChains(self.target)

    def get_distance_to_last_manipulated(self, web_element):
        if False:
            i = 10
            return i + 15
        if not self.last_manipulated_element:
            return 0
        try:
            if hasattr(self.last_manipulated_element, 'location'):
                last_location = self.last_manipulated_element.location
                return last_location.distance_to(web_element.location)
        except StaleElementReferenceException:
            return 0
        else:
            return 0

    def find_elements_by_name(self, name):
        if False:
            print('Hello World!')
        return self.target.find_elements(By.NAME, name) or []

    def find_elements_by_xpath(self, xpath):
        if False:
            i = 10
            return i + 15
        return self.target.find_elements(By.XPATH, xpath) or []

    def find_elements_by_css_selector(self, selector):
        if False:
            i = 10
            return i + 15
        return self.target.find_elements(By.CSS_SELECTOR, selector) or []

    def is_firefox(self):
        if False:
            while True:
                i = 10
        return self.browser_name == 'firefox'

    @property
    def browser_name(self):
        if False:
            i = 10
            return i + 15
        return self.target.capabilities['browserName']

    def is_ie(self):
        if False:
            print('Hello World!')
        return self.browser_name == 'internet explorer'

def _translate_url_errors_caused_by_server_shutdown(f):
    if False:
        while True:
            i = 10

    def f_decorated(*args, **kwargs):
        if False:
            while True:
                i = 10
        try:
            return f(*args, **kwargs)
        except URLError as url_error:
            if _is_caused_by_server_shutdown(url_error):
                raise StaleElementReferenceException('The Selenium server this element belonged to is no longer available.')
            else:
                raise
    return f_decorated

def _is_caused_by_server_shutdown(url_error):
    if False:
        for i in range(10):
            print('nop')
    try:
        CONNECTION_REFUSED = 10061
        return url_error.args[0][0] == CONNECTION_REFUSED
    except (IndexError, TypeError):
        return False

def handle_element_being_in_other_frame(f):
    if False:
        return 10

    def f_decorated(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not self.frame_index:
            return f(self, *args, **kwargs)
        try:
            return f(self, *args, **kwargs)
        except (StaleElementReferenceException, NoSuchElementException) as original_exc:
            try:
                frame_iterator = FrameIterator(self.target.parent)
                frame_iterator.switch_to_frame(self.frame_index)
            except NoSuchFrameException:
                raise original_exc
            else:
                return f(self, *args, **kwargs)
    return f_decorated

class WebElementWrapper:

    def __init__(self, target, frame_index=None):
        if False:
            for i in range(10):
                print('nop')
        self.target = target
        self.frame_index = frame_index
        self._cached_location = None

    @property
    @handle_element_being_in_other_frame
    @_translate_url_errors_caused_by_server_shutdown
    def location(self):
        if False:
            for i in range(10):
                print('nop')
        if self._cached_location is None:
            location = self.target.location
            (x, y) = (location['x'], location['y'])
            size = self.target.size
            (width, height) = (size['width'], size['height'])
            self._cached_location = Rectangle(x, y, width, height)
        return self._cached_location

    def is_displayed(self):
        if False:
            print('Hello World!')
        try:
            return self.target.is_displayed() and self.location.intersects(Rectangle(0, 0, sys.maxsize, sys.maxsize))
        except StaleElementReferenceException:
            return False

    @handle_element_being_in_other_frame
    def get_attribute(self, attr_name):
        if False:
            print('Hello World!')
        return self.target.get_attribute(attr_name)

    @property
    @handle_element_being_in_other_frame
    def text(self):
        if False:
            return 10
        return self.target.text

    @handle_element_being_in_other_frame
    def clear(self):
        if False:
            print('Hello World!')
        self.target.clear()

    @handle_element_being_in_other_frame
    def send_keys(self, keys):
        if False:
            for i in range(10):
                print('nop')
        self.target.send_keys(keys)

    @property
    @handle_element_being_in_other_frame
    def tag_name(self):
        if False:
            i = 10
            return i + 15
        return self.target.tag_name

    def unwrap(self):
        if False:
            return 10
        return self.target

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s>%s</%s>' % (self.tag_name, self.target.text, self.tag_name)

class FrameIterator:

    def __init__(self, driver, start_frame=None):
        if False:
            for i in range(10):
                print('nop')
        if start_frame is None:
            start_frame = []
        self.driver = driver
        self.start_frame = start_frame

    def __iter__(self):
        if False:
            while True:
                i = 10
        yield []
        for new_frame in range(sys.maxsize):
            try:
                self.driver.switch_to.frame(new_frame)
            except WebDriverException:
                break
            else:
                new_start_frame = self.start_frame + [new_frame]
                for result in FrameIterator(self.driver, new_start_frame):
                    yield ([new_frame] + result)
                try:
                    self.switch_to_frame(self.start_frame)
                except NoSuchFrameException:
                    raise FramesChangedWhileIterating()

    def switch_to_frame(self, frame_index_path):
        if False:
            print('Hello World!')
        self.driver.switch_to.default_content()
        for frame_index in frame_index_path:
            self.driver.switch_to.frame(frame_index)

class FramesChangedWhileIterating(Exception):
    pass