from helium import start_chrome
from helium._impl.util.system import is_windows
from tests.api import test_browser_name
from tests.api.test_kill_service_at_exit import KillServiceAtExitAT
from tests.api.util import InSubProcess
from unittest import TestCase, skipIf

@skipIf(test_browser_name() != 'chrome', 'Only run this test for Chrome')
class KillServiceAtExitChromeTest(KillServiceAtExitAT, TestCase):

    def get_service_process_names(self):
        if False:
            while True:
                i = 10
        if is_windows():
            return ['chromedriver.exe']
        return ['chromedriver']

    def get_browser_process_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'chrome' + ('.exe' if is_windows() else '')

    def start_browser_in_sub_process(self):
        if False:
            return 10
        with ChromeInSubProcess():
            pass

class ChromeInSubProcess(InSubProcess):

    @classmethod
    def main(cls):
        if False:
            while True:
                i = 10
        start_chrome(headless=True)
        cls.synchronize_with_parent_process()
if __name__ == '__main__':
    ChromeInSubProcess.main()