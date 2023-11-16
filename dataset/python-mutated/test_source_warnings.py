import os
import shutil
import pytest
from selenium import webdriver
from tests.functional.app_navigators.source_app_nav import SourceAppNavigator
from tests.functional.web_drivers import _FIREFOX_PATH

@pytest.fixture()
def orbot_web_driver(sd_servers):
    if False:
        while True:
            i = 10
    orbot_user_agent = 'Mozilla/5.0 (Android; Mobile; rv:52.0) Gecko/20100101 Firefox/52.0'
    f_profile_path2 = '/tmp/testprofile2'
    if os.path.exists(f_profile_path2):
        shutil.rmtree(f_profile_path2)
    os.mkdir(f_profile_path2)
    profile = webdriver.FirefoxProfile(f_profile_path2)
    profile.set_preference('general.useragent.override', orbot_user_agent)
    if sd_servers.journalist_app_base_url.find('.onion') != -1:
        profile.set_preference('network.proxy.type', 1)
        profile.set_preference('network.proxy.socks', '127.0.0.1')
        profile.set_preference('network.proxy.socks_port', 9150)
        profile.set_preference('network.proxy.socks_version', 5)
        profile.set_preference('network.proxy.socks_remote_dns', True)
        profile.set_preference('network.dns.blockDotOnion', False)
    profile.update_preferences()
    orbot_web_driver = webdriver.Firefox(firefox_binary=_FIREFOX_PATH, firefox_profile=profile)
    try:
        driver_user_agent = orbot_web_driver.execute_script('return navigator.userAgent')
        assert driver_user_agent == orbot_user_agent
        yield orbot_web_driver
    finally:
        orbot_web_driver.quit()

class TestSourceAppBrowserWarnings:

    def test_warning_appears_if_tor_browser_not_in_use(self, sd_servers, firefox_web_driver):
        if False:
            for i in range(10):
                print('nop')
        navigator = SourceAppNavigator(source_app_base_url=sd_servers.source_app_base_url, web_driver=firefox_web_driver)
        navigator.source_visits_source_homepage()
        warning_banner = navigator.driver.find_element_by_id('browser-tb')
        assert warning_banner.is_displayed()
        assert 'It is recommended to use Tor Browser' in warning_banner.text
        warning_dismiss_button = navigator.driver.find_element_by_id('browser-tb-close')
        warning_dismiss_button.click()

        def warning_banner_is_hidden():
            if False:
                for i in range(10):
                    print('nop')
            assert warning_banner.is_displayed() is False
        navigator.nav_helper.wait_for(warning_banner_is_hidden)

    def test_warning_appears_if_orbot_is_used(self, sd_servers, orbot_web_driver):
        if False:
            print('Hello World!')
        navigator = SourceAppNavigator(source_app_base_url=sd_servers.source_app_base_url, web_driver=orbot_web_driver)
        navigator.source_visits_source_homepage()
        warning_banner = navigator.driver.find_element_by_id('browser-android')
        assert warning_banner.is_displayed()
        assert 'use the desktop version of Tor Browser' in warning_banner.text
        warning_dismiss_button = navigator.driver.find_element_by_id('browser-android-close')
        warning_dismiss_button.click()

        def warning_banner_is_hidden():
            if False:
                return 10
            assert warning_banner.is_displayed() is False
        navigator.nav_helper.wait_for(warning_banner_is_hidden)

    def test_warning_high_security(self, sd_servers, tor_browser_web_driver):
        if False:
            print('Hello World!')
        navigator = SourceAppNavigator(source_app_base_url=sd_servers.source_app_base_url, web_driver=tor_browser_web_driver)
        navigator.source_visits_source_homepage()
        banner = navigator.driver.find_element_by_id('browser-security-level')
        assert banner.is_displayed()
        assert 'Security Level is too low' in banner.text