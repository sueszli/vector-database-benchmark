from helium._impl.selenium_wrappers import FrameIterator, FramesChangedWhileIterating
from selenium.common.exceptions import NoSuchFrameException
from unittest import TestCase

class FrameIteratorTest(TestCase):

    def test_only_main_frame(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual([[]], list(FrameIterator(StubWebDriver())))

    def test_one_frame(self):
        if False:
            i = 10
            return i + 15
        driver = StubWebDriver(Frame())
        self.assertEqual([[], [0]], list(FrameIterator(driver)))

    def test_two_frames(self):
        if False:
            return 10
        driver = StubWebDriver(Frame(), Frame())
        self.assertEqual([[], [0], [1]], list(FrameIterator(driver)))

    def test_nested_frame(self):
        if False:
            return 10
        driver = StubWebDriver(Frame(Frame()))
        self.assertEqual([[], [0], [0, 0]], list(FrameIterator(driver)))

    def test_complex(self):
        if False:
            return 10
        driver = StubWebDriver(Frame(Frame()), Frame())
        self.assertEqual([[], [0], [0, 0], [1]], list(FrameIterator(driver)))

    def test_disappearing_frame(self):
        if False:
            return 10
        child_frame = Frame()
        first_frame = Frame(child_frame)
        driver = StubWebDriver(first_frame)
        driver.switch_to = TargetLocatorFailingAfterNFrameSwitches(driver, 2)
        with self.assertRaises(FramesChangedWhileIterating):
            list(FrameIterator(driver))

class StubWebDriver:

    def __init__(self, *frames):
        if False:
            print('Hello World!')
        self.frames = list(frames)
        self.switch_to = StubTargetLocator(self)
        self.current_frame = None

class StubTargetLocator:

    def __init__(self, driver):
        if False:
            while True:
                i = 10
        self.driver = driver

    def default_content(self):
        if False:
            for i in range(10):
                print('nop')
        self.driver.current_frame = None

    def frame(self, index):
        if False:
            return 10
        if self.driver.current_frame is None:
            children = self.driver.frames
        else:
            children = self.driver.current_frame.children
        try:
            new_frame = children[index]
        except IndexError:
            raise NoSuchFrameException()
        else:
            self.driver.current_frame = new_frame

class Frame:

    def __init__(self, *children):
        if False:
            print('Hello World!')
        self.children = children

class TargetLocatorFailingAfterNFrameSwitches(StubTargetLocator):

    def __init__(self, driver, num_allowed_frame_switches):
        if False:
            while True:
                i = 10
        super(TargetLocatorFailingAfterNFrameSwitches, self).__init__(driver)
        self.num_allowed_frame_switches = num_allowed_frame_switches

    def frame(self, index):
        if False:
            i = 10
            return i + 15
        if self.num_allowed_frame_switches > 0:
            self.num_allowed_frame_switches -= 1
            return super(TargetLocatorFailingAfterNFrameSwitches, self).frame(index)
        raise NoSuchFrameException()