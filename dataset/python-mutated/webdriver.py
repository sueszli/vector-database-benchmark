"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..util.dependencies import import_required
import_required('selenium.webdriver', "To use bokeh.io image export functions you need selenium ('conda install selenium' or 'pip install selenium')")
import atexit
import os
from os.path import devnull
from shutil import which
from typing import Literal
from packaging.version import Version
from selenium.webdriver.remote.webdriver import WebDriver
from ..settings import settings
DriverKind = Literal['firefox', 'chromium']
__all__ = ('webdriver_control',)

def create_firefox_webdriver(scale_factor: float=1) -> WebDriver:
    if False:
        i = 10
        return i + 15
    firefox = which('firefox')
    if firefox is None:
        raise RuntimeError('firefox is not installed or not present on PATH')
    geckodriver = which('geckodriver')
    if geckodriver is None:
        raise RuntimeError('geckodriver is not installed or not present on PATH')
    import selenium
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.firefox.webdriver import WebDriver as Firefox
    if Version(selenium.__version__) >= Version('4.11'):
        service = Service()
    else:
        service = Service(log_path=devnull)
    options = Options()
    options.add_argument('--headless')
    options.set_preference('layout.css.devPixelsPerPx', f'{scale_factor}')
    return Firefox(service=service, options=options)

def create_chromium_webdriver(extra_options: list[str] | None=None, scale_factor: float=1) -> WebDriver:
    if False:
        i = 10
        return i + 15
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.webdriver import WebDriver as Chrome
    executable_path = settings.chromedriver_path()
    if executable_path is None:
        for executable in ['chromedriver', 'chromium.chromedriver', 'chromedriver-binary']:
            executable_path = which(executable)
            if executable_path is not None:
                break
        else:
            raise RuntimeError("chromedriver or its variant is not installed or not present on PATH; use BOKEH_CHROMEDRIVER_PATH to specify a customized chromedriver's location")
    service = Service(executable_path)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--hide-scrollbars')
    options.add_argument(f'--force-device-scale-factor={scale_factor}')
    options.add_argument('--force-color-profile=srgb')
    if extra_options:
        for op in extra_options:
            options.add_argument(op)
    if os.getenv('BOKEH_IN_DOCKER') == '1':
        options.add_argument('--no-sandbox')
    return Chrome(service=service, options=options)

def scale_factor_less_than_web_driver_device_pixel_ratio(scale_factor: float, web_driver: WebDriver) -> bool:
    if False:
        i = 10
        return i + 15
    device_pixel_ratio = get_web_driver_device_pixel_ratio(web_driver)
    return device_pixel_ratio >= scale_factor

def get_web_driver_device_pixel_ratio(web_driver: WebDriver) -> float:
    if False:
        for i in range(10):
            print('nop')
    calculate_web_driver_device_pixel_ratio = '        return window.devicePixelRatio\n    '
    device_pixel_ratio: float = web_driver.execute_script(calculate_web_driver_device_pixel_ratio)
    return device_pixel_ratio

def _try_create_firefox_webdriver(scale_factor: float=1) -> WebDriver | None:
    if False:
        return 10
    try:
        return create_firefox_webdriver(scale_factor=scale_factor)
    except Exception:
        return None

def _try_create_chromium_webdriver(scale_factor: float=1) -> WebDriver | None:
    if False:
        for i in range(10):
            print('nop')
    try:
        return create_chromium_webdriver(scale_factor=scale_factor)
    except Exception:
        return None

class _WebdriverState:
    reuse: bool
    kind: DriverKind | None
    current: WebDriver | None
    _drivers: set[WebDriver]

    def __init__(self, *, kind: DriverKind | None=None, reuse: bool=True) -> None:
        if False:
            return 10
        self.kind = kind
        self.reuse = reuse
        self.current = None
        self._drivers = set()

    def terminate(self, driver: WebDriver) -> None:
        if False:
            while True:
                i = 10
        self._drivers.remove(driver)
        driver.quit()

    def reset(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.current is not None:
            self.terminate(self.current)
            self.current = None

    def get(self, scale_factor: float=1) -> WebDriver:
        if False:
            print('Hello World!')
        if not self.reuse or self.current is None or (not scale_factor_less_than_web_driver_device_pixel_ratio(scale_factor, self.current)):
            self.reset()
            self.current = self.create(scale_factor=scale_factor)
        return self.current

    def create(self, kind: DriverKind | None=None, scale_factor: float=1) -> WebDriver:
        if False:
            while True:
                i = 10
        driver = self._create(kind, scale_factor=scale_factor)
        self._drivers.add(driver)
        return driver

    def _create(self, kind: DriverKind | None, scale_factor: float=1) -> WebDriver:
        if False:
            i = 10
            return i + 15
        driver_kind = kind or self.kind
        if driver_kind is None:
            driver = _try_create_chromium_webdriver(scale_factor=scale_factor)
            if driver is not None:
                self.kind = 'chromium'
                return driver
            driver = _try_create_firefox_webdriver(scale_factor=scale_factor)
            if driver is not None:
                self.kind = 'firefox'
                return driver
            raise RuntimeError("Neither firefox and geckodriver nor a variant of chromium browser and chromedriver are available on system PATH. You can install the former with 'conda install -c conda-forge firefox geckodriver'.")
        elif driver_kind == 'chromium':
            return create_chromium_webdriver(scale_factor=scale_factor)
        elif driver_kind == 'firefox':
            return create_firefox_webdriver(scale_factor=scale_factor)
        else:
            raise ValueError(f"'{driver_kind}' is not a recognized webdriver kind")

    def cleanup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.reset()
        for driver in list(self._drivers):
            self.terminate(driver)
webdriver_control = _WebdriverState()
atexit.register(lambda : webdriver_control.cleanup())