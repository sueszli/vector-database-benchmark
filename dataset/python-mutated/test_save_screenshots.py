import os
from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class ScreenshotTests(BaseCase):

    def test_save_screenshot(self):
        if False:
            while True:
                i = 10
        self.open('https://seleniumbase.io/demo_page')
        self.save_screenshot('demo_page.png', folder='./downloaded_files')
        self.assert_downloaded_file('demo_page.png')
        print('\n"%s/%s" was saved!' % ('downloaded_files', 'demo_page.png'))

    def test_save_screenshot_to_logs(self):
        if False:
            for i in range(10):
                print('nop')
        self.open('https://seleniumbase.io/demo_page')
        self.save_screenshot_to_logs()
        test_logpath = os.path.join(self.log_path, self.test_id)
        expected_screenshot = os.path.join(test_logpath, '_1_screenshot.png')
        self.assert_true(os.path.exists(expected_screenshot))
        print('\n"%s" was saved!' % expected_screenshot)
        self.open('https://seleniumbase.io/tinymce/')
        self.save_screenshot_to_logs()
        expected_screenshot = os.path.join(test_logpath, '_2_screenshot.png')
        self.assert_true(os.path.exists(expected_screenshot))
        print('"%s" was saved!' % expected_screenshot)
        self.open('https://seleniumbase.io/error_page/')
        self.save_screenshot_to_logs('error_page')
        expected_screenshot = os.path.join(test_logpath, '_3_error_page.png')
        self.assert_true(os.path.exists(expected_screenshot))
        print('"%s" was saved!' % expected_screenshot)
        self.open('https://seleniumbase.io/devices/')
        self.save_screenshot_to_logs('devices')
        expected_screenshot = os.path.join(test_logpath, '_4_devices.png')
        self.assert_true(os.path.exists(expected_screenshot))
        print('"%s" was saved!' % expected_screenshot)