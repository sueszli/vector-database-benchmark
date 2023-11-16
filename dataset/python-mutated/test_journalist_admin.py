import logging
import time
from pathlib import Path
from typing import Callable
import pytest
from selenium.common.exceptions import TimeoutException
from selenium.webdriver import ActionChains
from tests.functional.app_navigators.journalist_app_nav import JournalistAppNavigator
from tests.functional.pageslayout.utils import list_locales, save_static_data

@pytest.mark.parametrize('locale', list_locales())
@pytest.mark.pagelayout()
class TestAdminLayoutAddAndEditUser:

    def test_admin_adds_user_hotp_and_edits_hotp(self, locale, sd_servers_with_clean_state, firefox_web_driver):
        if False:
            i = 10
            return i + 15
        assert sd_servers_with_clean_state.journalist_is_admin
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_clean_state.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_clean_state.journalist_username, password=sd_servers_with_clean_state.journalist_password, otp_secret=sd_servers_with_clean_state.journalist_otp_secret)
        journ_app_nav.admin_visits_admin_interface()
        save_static_data(journ_app_nav.driver, locale, 'journalist-admin_interface_index')

        def screenshot_of_add_user_hotp_form() -> None:
            if False:
                while True:
                    i = 10
            save_static_data(journ_app_nav.driver, locale, 'journalist-admin_add_user_hotp')

        def screenshot_of_journalist_new_user_two_factor_hotp() -> None:
            if False:
                while True:
                    i = 10
            save_static_data(journ_app_nav.driver, locale, 'journalist-admin_new_user_two_factor_hotp')
        result = journ_app_nav.admin_creates_a_user(hotp_secret='c4 26 43 52 69 13 02 49 9f 6a a5 33 96 46 d9 05 42 a3 4f ae', callback_before_submitting_add_user_step=screenshot_of_add_user_hotp_form, callback_before_submitting_2fa_step=screenshot_of_journalist_new_user_two_factor_hotp)
        (new_user_username, new_user_pw, new_user_otp_secret) = result
        save_static_data(journ_app_nav.driver, locale, 'journalist-admin')
        journ_app_nav.admin_visits_user_edit_page(username_of_journalist_to_edit=new_user_username)
        save_static_data(journ_app_nav.driver, locale, 'journalist-edit_account_admin')
        save_static_data(journ_app_nav.driver, locale, 'journalist-admin_edit_hotp_secret')

        def _admin_visits_reset_2fa_hotp_step() -> None:
            if False:
                print('Hello World!')
            hotp_reset_button = journ_app_nav.driver.find_elements_by_id('reset-two-factor-hotp')[0]
            hotp_reset_button.location_once_scrolled_into_view
            ActionChains(journ_app_nav.driver).move_to_element(hotp_reset_button).perform()
            time.sleep(1)
            tip_opacity = journ_app_nav.driver.find_elements_by_css_selector('#button-reset-two-factor-hotp span.tooltip')[0].value_of_css_property('opacity')
            tip_text = journ_app_nav.driver.find_elements_by_css_selector('#button-reset-two-factor-hotp span.tooltip')[0].text
            assert tip_opacity == '1'
            if not journ_app_nav.accept_languages:
                assert tip_text == 'Reset two-factor authentication for security keys, like a YubiKey'
            journ_app_nav.nav_helper.safe_click_by_id('button-reset-two-factor-hotp')
        self._retry_2fa_pop_ups(journ_app_nav, _admin_visits_reset_2fa_hotp_step, 'reset-two-factor-hotp')
        journ_app_nav.nav_helper.wait_for(lambda : journ_app_nav.driver.find_element_by_css_selector('input[name="otp_secret"]'))

    def test_admin_adds_user_totp_and_edits_totp(self, locale, sd_servers_with_clean_state, firefox_web_driver):
        if False:
            print('Hello World!')
        assert sd_servers_with_clean_state.journalist_is_admin
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_clean_state.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_clean_state.journalist_username, password=sd_servers_with_clean_state.journalist_password, otp_secret=sd_servers_with_clean_state.journalist_otp_secret)
        journ_app_nav.admin_visits_admin_interface()

        def screenshot_of_add_user_totp_form() -> None:
            if False:
                return 10
            save_static_data(journ_app_nav.driver, locale, 'journalist-admin_add_user_totp')

        def screenshot_of_journalist_new_user_two_factor_totp() -> None:
            if False:
                while True:
                    i = 10
            save_static_data(journ_app_nav.driver, locale, 'journalist-admin_new_user_two_factor_totp')
        result = journ_app_nav.admin_creates_a_user(callback_before_submitting_add_user_step=screenshot_of_add_user_totp_form, callback_before_submitting_2fa_step=screenshot_of_journalist_new_user_two_factor_totp)
        (new_user_username, new_user_pw, new_user_otp_secret) = result
        journ_app_nav.admin_visits_user_edit_page(username_of_journalist_to_edit=new_user_username)

        def _admin_visits_reset_2fa_totp_step() -> None:
            if False:
                i = 10
                return i + 15
            totp_reset_button = journ_app_nav.driver.find_elements_by_id('reset-two-factor-totp')[0]
            assert '/admin/reset-2fa-totp' in totp_reset_button.get_attribute('action')
            totp_reset_button = journ_app_nav.driver.find_elements_by_css_selector('#button-reset-two-factor-totp')[0]
            totp_reset_button.location_once_scrolled_into_view
            ActionChains(journ_app_nav.driver).move_to_element(totp_reset_button).perform()
            time.sleep(1)
            tip_opacity = journ_app_nav.driver.find_elements_by_css_selector('#button-reset-two-factor-totp span.tooltip')[0].value_of_css_property('opacity')
            tip_text = journ_app_nav.driver.find_elements_by_css_selector('#button-reset-two-factor-totp span.tooltip')[0].text
            assert tip_opacity == '1'
            if not journ_app_nav.accept_languages:
                expected_text = 'Reset two-factor authentication for mobile apps, such as FreeOTP'
                assert tip_text == expected_text
            journ_app_nav.nav_helper.safe_click_by_id('button-reset-two-factor-totp')
        self._retry_2fa_pop_ups(journ_app_nav, _admin_visits_reset_2fa_totp_step, 'reset-two-factor-totp')
        save_static_data(journ_app_nav.driver, locale, 'journalist-admin_edit_totp_secret')

    @staticmethod
    def _retry_2fa_pop_ups(journ_app_nav: JournalistAppNavigator, navigation_step: Callable, button_to_click: str) -> None:
        if False:
            while True:
                i = 10
        'Clicking on Selenium alerts can be flaky. We need to retry them if they timeout.'
        for i in range(15):
            try:
                try:
                    journ_app_nav.nav_helper.wait_for(lambda : journ_app_nav.driver.find_elements_by_id(button_to_click)[0])
                except IndexError:
                    journ_app_nav.nav_helper.alert_wait()
                    journ_app_nav.nav_helper.alert_accept()
                    break
                navigation_step()
                journ_app_nav.nav_helper.alert_wait()
                journ_app_nav.nav_helper.alert_accept()
                break
            except TimeoutException:
                logging.info('Selenium has failed to click; retrying.')

@pytest.mark.parametrize('locale', list_locales())
@pytest.mark.pagelayout()
class TestAdminLayoutEditConfig:

    def test_admin_changes_logo(self, locale, sd_servers_with_clean_state, firefox_web_driver):
        if False:
            i = 10
            return i + 15
        assert sd_servers_with_clean_state.journalist_is_admin
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_clean_state.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_clean_state.journalist_username, password=sd_servers_with_clean_state.journalist_password, otp_secret=sd_servers_with_clean_state.journalist_otp_secret)
        journ_app_nav.admin_visits_admin_interface()
        journ_app_nav.admin_visits_system_config_page()
        save_static_data(journ_app_nav.driver, locale, 'journalist-admin_system_config_page')
        current_file_path = Path(__file__).absolute().parent
        logo_path = current_file_path / '..' / '..' / '..' / 'static' / 'i' / 'logo.png'
        assert logo_path.is_file()
        journ_app_nav.nav_helper.safe_send_keys_by_id('logo-upload', str(logo_path))
        journ_app_nav.nav_helper.safe_click_by_id('submit-logo-update')

        def updated_image() -> None:
            if False:
                print('Hello World!')
            flash_msg = journ_app_nav.driver.find_element_by_css_selector('.flash')
            assert 'Image updated.' in flash_msg.text
        journ_app_nav.nav_helper.wait_for(updated_image, timeout=20)
        save_static_data(journ_app_nav.driver, locale, 'journalist-admin_changes_logo_image')

    def test_ossec_alert_button(self, locale, sd_servers, firefox_web_driver):
        if False:
            return 10
        assert sd_servers.journalist_is_admin
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers.journalist_username, password=sd_servers.journalist_password, otp_secret=sd_servers.journalist_otp_secret)
        journ_app_nav.admin_visits_admin_interface()
        journ_app_nav.admin_visits_system_config_page()
        alert_button = journ_app_nav.driver.find_element_by_id('test-ossec-alert')
        alert_button.click()

        def test_alert_sent():
            if False:
                return 10
            flash_msg = journ_app_nav.driver.find_element_by_css_selector('.flash')
            assert 'Test alert sent. Please check your email.' in flash_msg.text
        journ_app_nav.nav_helper.wait_for(test_alert_sent)
        save_static_data(journ_app_nav.driver, locale, 'journalist-admin_ossec_alert_button')