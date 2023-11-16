from __future__ import annotations
import logging
from typing import Any, Callable, TYPE_CHECKING
from urllib.parse import urlparse
from flask import current_app, Flask, request, Response, session
from flask_login import login_user
from selenium.webdriver.remote.webdriver import WebDriver
from werkzeug.http import parse_cookie
from superset.utils.class_utils import load_class_from_name
from superset.utils.urls import headless_url
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from flask_appbuilder.security.sqla.models import User
    try:
        from playwright.sync_api import BrowserContext
    except ModuleNotFoundError:
        BrowserContext = Any

class MachineAuthProvider:

    def __init__(self, auth_webdriver_func_override: Callable[[WebDriver | BrowserContext, User], WebDriver | BrowserContext] | None=None):
        if False:
            return 10
        self._auth_webdriver_func_override = auth_webdriver_func_override

    def authenticate_webdriver(self, driver: WebDriver, user: User) -> WebDriver:
        if False:
            for i in range(10):
                print('nop')
        '\n        Default AuthDriverFuncType type that sets a session cookie flask-login style\n        :return: The WebDriver passed in (fluent)\n        '
        if self._auth_webdriver_func_override:
            return self._auth_webdriver_func_override(driver, user)
        driver.get(headless_url('/login/'))
        cookies = self.get_cookies(user)
        for (cookie_name, cookie_val) in cookies.items():
            driver.add_cookie({'name': cookie_name, 'value': cookie_val})
        return driver

    def authenticate_browser_context(self, browser_context: BrowserContext, user: User) -> BrowserContext:
        if False:
            i = 10
            return i + 15
        if self._auth_webdriver_func_override:
            return self._auth_webdriver_func_override(browser_context, user)
        url = urlparse(current_app.config['WEBDRIVER_BASEURL'])
        page = browser_context.new_page()
        page.goto(headless_url('/login/'))
        cookies = self.get_cookies(user)
        browser_context.clear_cookies()
        browser_context.add_cookies([{'name': cookie_name, 'value': cookie_val, 'domain': url.netloc, 'path': '/', 'sameSite': 'Lax', 'httpOnly': True} for (cookie_name, cookie_val) in cookies.items()])
        return browser_context

    def get_cookies(self, user: User | None) -> dict[str, str]:
        if False:
            return 10
        if user:
            cookies = self.get_auth_cookies(user)
        elif request.cookies:
            cookies = request.cookies
        else:
            cookies = {}
        return cookies

    @staticmethod
    def get_auth_cookies(user: User) -> dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        with current_app.test_request_context('/login'):
            login_user(user)
            response = Response()
            current_app.session_interface.save_session(current_app, session, response)
        cookies = {}
        for (name, value) in response.headers:
            if name.lower() == 'set-cookie':
                cookie = parse_cookie(value)
                cookie_tuple = list(cookie.items())[0]
                cookies[cookie_tuple[0]] = cookie_tuple[1]
        return cookies

class MachineAuthProviderFactory:

    def __init__(self) -> None:
        if False:
            return 10
        self._auth_provider: MachineAuthProvider | None = None

    def init_app(self, app: Flask) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._auth_provider = load_class_from_name(app.config['MACHINE_AUTH_PROVIDER_CLASS'])(app.config['WEBDRIVER_AUTH_FUNC'])

    @property
    def instance(self) -> MachineAuthProvider:
        if False:
            while True:
                i = 10
        return self._auth_provider