import logging
import os
os.environ.setdefault('WDM_LOG', str(logging.NOTSET))
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.driver_cache import DriverCacheManager
import webdriver_manager
if webdriver_manager.__version__ == '4.0.0':

    class DriverCacheManager(DriverCacheManager):

        def __get_metadata_key(self, driver):
            if False:
                while True:
                    i = 10
            super().__get_metadata_key(driver)
            return self._metadata_key

def install_matching_chromedriver(cache_dir=None):
    if False:
        for i in range(10):
            print('nop')
    manager = ChromeDriverManager(cache_manager=DriverCacheManager(cache_dir))
    return manager.install()