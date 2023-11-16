import base64
import gzip
from binascii import unhexlify
from random import randint
from typing import Callable, Dict, Iterable, Optional, Tuple
import requests
import two_factor
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from tests import utils
from tests.functional.app_navigators._nav_helper import NavigationHelper
from tests.functional.tor_utils import proxies_for_url

class JournalistAppNavigator:
    """Helper functions to navigate the journalist app when implementing functional/selenium tests.

    Only logic that needs to be shared across multiple tests within different files should be
    added to this class, in order to keep this class as small as possible.
    """

    def __init__(self, journalist_app_base_url: str, web_driver: WebDriver, accept_languages: Optional[str]=None) -> None:
        if False:
            return 10
        self._journalist_app_base_url = journalist_app_base_url
        self.nav_helper = NavigationHelper(web_driver)
        self.driver = web_driver
        self.accept_languages = accept_languages

    def is_on_journalist_homepage(self) -> WebElement:
        if False:
            for i in range(10):
                print('nop')
        return self.nav_helper.wait_for(lambda : self.driver.find_element_by_css_selector('div.journalist-view-all'))

    def journalist_goes_to_login_page_and_enters_credentials(self, username: str, password: str, otp_secret: str, should_submit_login_form: bool) -> None:
        if False:
            print('Hello World!')
        self.driver.get(f'{self._journalist_app_base_url}/login')
        self.nav_helper.safe_send_keys_by_css_selector('input[name="username"]', username)
        self.nav_helper.safe_send_keys_by_css_selector('input[name="password"]', password)
        otp = two_factor.TOTP(otp_secret)
        self.nav_helper.safe_send_keys_by_css_selector('input[name="token"]', otp.now())
        if should_submit_login_form:
            self.nav_helper.safe_click_by_css_selector('button[type="submit"]')

    def journalist_logs_in(self, username: str, password: str, otp_secret: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.journalist_goes_to_login_page_and_enters_credentials(username=username, password=password, otp_secret=otp_secret, should_submit_login_form=True)
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_id('link-logout'))
        assert self.is_on_journalist_homepage()

    def journalist_checks_messages(self) -> None:
        if False:
            i = 10
            return i + 15
        self.driver.get(self._journalist_app_base_url)
        collections_count = self.count_sources_on_index_page()
        assert collections_count == 1
        if not self.accept_languages:
            unread_span = self.driver.find_element_by_css_selector('tr.unread')
            assert '1 unread' in unread_span.text

    @staticmethod
    def _download_content_at_url(url: str, cookies: Dict[str, str]) -> bytes:
        if False:
            print('Hello World!')
        r = requests.get(url, cookies=cookies, proxies=proxies_for_url(url), stream=True)
        if r.status_code != 200:
            raise Exception('Failed to download the data.')
        data = b''
        for chunk in r.iter_content(1024):
            data += chunk
        return data

    def journalist_downloads_first_message(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self.journalist_selects_the_first_source()
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_css_selector('table#submissions'))
        submissions = self.driver.find_elements_by_css_selector('#submissions a')
        assert len(submissions) == 1
        file_url = submissions[0].get_attribute('href')

        def cookie_string_from_selenium_cookies(cookies: Iterable[Dict[str, str]]) -> Dict[str, str]:
            if False:
                for i in range(10):
                    print('nop')
            result = {}
            for cookie in cookies:
                result[cookie['name']] = cookie['value']
            return result
        cks = cookie_string_from_selenium_cookies(self.driver.get_cookies())
        raw_content = self._download_content_at_url(file_url, cks)
        decryption_result = utils.decrypt_as_journalist(raw_content)
        if file_url.endswith('.gz.gpg'):
            decrypted_message = gzip.decompress(decryption_result)
        else:
            decrypted_message = decryption_result
        return decrypted_message.decode()

    def journalist_selects_the_first_source(self) -> None:
        if False:
            print('Hello World!')
        self.driver.find_element_by_css_selector('#un-starred-source-link-1').click()

    def journalist_composes_reply_to_source(self, reply_content: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_id('reply-text-field'))
        self.nav_helper.safe_send_keys_by_id('reply-text-field', reply_content)

    def journalist_sends_reply_to_source(self, reply_content: str='Thanks for the documents. Can you submit more? éè') -> None:
        if False:
            print('Hello World!')
        self.journalist_composes_reply_to_source(reply_content=reply_content)
        self.driver.find_element_by_id('reply-button').click()

        def reply_stored() -> None:
            if False:
                i = 10
                return i + 15
            if not self.accept_languages:
                assert 'The source will receive your reply' in self.driver.page_source
        self.nav_helper.wait_for(reply_stored)

    def journalist_visits_col(self) -> None:
        if False:
            while True:
                i = 10
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_css_selector('table#collections'))
        self.nav_helper.safe_click_by_id('un-starred-source-link-1')
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_css_selector('table#submissions'))

    def journalist_selects_first_doc(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.nav_helper.safe_click_by_css_selector('input[type="checkbox"][name="doc_names_selected"]')
        self.nav_helper.wait_for(lambda : expected_conditions.element_located_to_be_selected((By.CSS_SELECTOR, 'input[type="checkbox"][name="doc_names_selected"]')))
        assert self.driver.find_element_by_css_selector('input[type="checkbox"][name="doc_names_selected"]').is_selected()

    def journalist_clicks_delete_selected_link(self) -> None:
        if False:
            while True:
                i = 10
        self.nav_helper.safe_click_by_css_selector('a#delete-selected-link')
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_id('delete-selected-confirmation-modal'))

    def journalist_clicks_delete_all_and_sees_confirmation(self) -> None:
        if False:
            return 10
        self.nav_helper.safe_click_all_by_css_selector('[name=doc_names_selected]')
        self.nav_helper.safe_click_by_css_selector('a#delete-selected-link')

    def journalist_confirms_delete_selected(self) -> None:
        if False:
            print('Hello World!')
        self.nav_helper.wait_for(lambda : expected_conditions.element_to_be_clickable((By.ID, 'delete-selected')))
        confirm_btn = self.driver.find_element_by_id('delete-selected')
        confirm_btn.location_once_scrolled_into_view
        ActionChains(self.driver).move_to_element(confirm_btn).click().perform()

    def get_submission_checkboxes_on_current_page(self):
        if False:
            while True:
                i = 10
        return self.driver.find_elements_by_name('doc_names_selected')

    def count_submissions_on_current_page(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.get_submission_checkboxes_on_current_page())

    def get_sources_on_index_page(self):
        if False:
            print('Hello World!')
        assert self.is_on_journalist_homepage()
        return self.driver.find_elements_by_class_name('code-name')

    def count_sources_on_index_page(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.get_sources_on_index_page())

    def journalist_confirm_delete_selected(self) -> None:
        if False:
            while True:
                i = 10
        self.nav_helper.wait_for(lambda : expected_conditions.element_to_be_clickable((By.ID, 'delete-selected')))
        confirm_btn = self.driver.find_element_by_id('delete-selected')
        confirm_btn.location_once_scrolled_into_view
        ActionChains(self.driver).move_to_element(confirm_btn).click().perform()

    def journalist_sees_link_to_admin_page(self) -> bool:
        if False:
            return 10
        try:
            self.driver.find_element_by_id('link-admin-index')
            return True
        except NoSuchElementException:
            return False

    def admin_visits_admin_interface(self) -> None:
        if False:
            i = 10
            return i + 15
        self.nav_helper.safe_click_by_id('link-admin-index')
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_id('add-user'))

    def admin_creates_a_user(self, username: Optional[str]=None, hotp_secret: Optional[str]=None, is_admin: bool=False, callback_before_submitting_add_user_step: Optional[Callable[[], None]]=None, callback_before_submitting_2fa_step: Optional[Callable[[], None]]=None) -> Optional[Tuple[str, str, str]]:
        if False:
            for i in range(10):
                print('nop')
        self.nav_helper.safe_click_by_id('add-user')
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_id('username'))
        if not self.accept_languages:
            btns = self.driver.find_elements_by_tag_name('button')
            assert 'ADD USER' in [el.text for el in btns]
        password = self.driver.find_element_by_css_selector('#password').text.strip()
        if not username:
            final_username = f'journalist{str(randint(1000, 1000000))}'
        else:
            final_username = username
        self.nav_helper.safe_send_keys_by_css_selector('input[name="username"]', final_username)
        if hotp_secret:
            self.nav_helper.safe_click_all_by_css_selector('input[name="is_hotp"]')
            self.nav_helper.safe_send_keys_by_css_selector('input[name="otp_secret"]', hotp_secret)
        if is_admin:
            self.nav_helper.safe_click_by_css_selector('input[name="is_admin"]')
        if callback_before_submitting_add_user_step:
            callback_before_submitting_add_user_step()
        self.nav_helper.safe_click_by_css_selector('button[type=submit]')
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_id('check-token'))
        if self.accept_languages in [None, 'en-US']:
            expected_title = 'Enable YubiKey (OATH-HOTP)' if hotp_secret else 'Enable FreeOTP'
            h1s = [h1.text for h1 in self.driver.find_elements_by_tag_name('h1')]
            assert expected_title in h1s
        if hotp_secret:
            otp_secret = hotp_secret
            hotp_secret_as_hex = unhexlify(hotp_secret.replace(' ', ''))
            hotp_secret_as_b32 = base64.b32encode(hotp_secret_as_hex).decode('ascii')
            hotp = two_factor.HOTP(hotp_secret_as_b32)
            current_2fa_code = hotp.generate(0)
        else:
            otp_secret = self.driver.find_element_by_css_selector('#shared-secret').text.strip().replace(' ', '')
            totp = two_factor.TOTP(otp_secret)
            current_2fa_code = totp.now()
        self.nav_helper.safe_send_keys_by_css_selector('input[name="token"]', current_2fa_code)
        if callback_before_submitting_2fa_step:
            callback_before_submitting_2fa_step()
        self.nav_helper.safe_click_by_css_selector('button[type=submit]')

        def user_token_added():
            if False:
                return 10
            if not self.accept_languages:
                flash_msg = self.driver.find_elements_by_css_selector('.flash')
                expected_msg = f'The two-factor code for user "{final_username}" was verified successfully.'
                assert expected_msg in [el.text for el in flash_msg]
        self.nav_helper.wait_for(user_token_added)
        return (final_username, password, otp_secret)

    def journalist_logs_out(self) -> None:
        if False:
            i = 10
            return i + 15
        self.nav_helper.safe_click_by_id('link-logout')
        self.nav_helper.wait_for(lambda : self.driver.find_element_by_css_selector('.login-form'))

        def login_page():
            if False:
                for i in range(10):
                    print('nop')
            assert 'Log in to access the journalist interface' in self.driver.page_source
        self.nav_helper.wait_for(login_page)

    def admin_visits_system_config_page(self):
        if False:
            for i in range(10):
                print('nop')
        self.nav_helper.safe_click_by_id('update-instance-config')

        def config_page_loaded():
            if False:
                for i in range(10):
                    print('nop')
            assert self.driver.find_element_by_id('test-ossec-alert')
        self.nav_helper.wait_for(config_page_loaded)

    def journalist_visits_edit_account(self):
        if False:
            print('Hello World!')
        self.nav_helper.safe_click_by_id('link-edit-account')

    def admin_visits_user_edit_page(self, username_of_journalist_to_edit: str) -> None:
        if False:
            print('Hello World!')
        selector = f'a.edit-user[data-username="{username_of_journalist_to_edit}"]'
        new_user_edit_links = self.driver.find_elements_by_css_selector(selector)
        assert len(new_user_edit_links) == 1
        new_user_edit_links[0].click()

        def can_edit_user():
            if False:
                return 10
            h = self.driver.find_elements_by_tag_name('h1')[0]
            assert f'Edit user "{username_of_journalist_to_edit}"' == h.text
        self.nav_helper.wait_for(can_edit_user)