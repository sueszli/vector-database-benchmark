from helium._impl.chromedriver import install_matching_chromedriver
from os import access, X_OK
from tempfile import TemporaryDirectory
from unittest import TestCase

class InstallMatchingChromeDriverTest(TestCase):

    def test_install_matching_chromedriver(self):
        if False:
            print('Hello World!')
        driver_path = self._install_matching_chromedriver()
        self.assertTrue(access(driver_path, X_OK))

    def _install_matching_chromedriver(self):
        if False:
            while True:
                i = 10
        return install_matching_chromedriver(self.temp_dir.name)

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()