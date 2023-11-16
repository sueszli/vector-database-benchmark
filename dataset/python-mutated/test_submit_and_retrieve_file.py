import pytest
from selenium.common.exceptions import NoSuchElementException
from tests.functional.app_navigators.journalist_app_nav import JournalistAppNavigator
from tests.functional.app_navigators.source_app_nav import SourceAppNavigator
from tests.functional.pageslayout.utils import list_locales, save_static_data

@pytest.mark.parametrize('locale', list_locales())
@pytest.mark.pagelayout()
class TestSubmitAndRetrieveFile:

    def test_submit_and_retrieve_happy_path(self, locale, sd_servers_with_clean_state, tor_browser_web_driver, firefox_web_driver):
        if False:
            i = 10
            return i + 15
        locale_with_commas = locale.replace('_', '-')
        source_app_nav = SourceAppNavigator(source_app_base_url=sd_servers_with_clean_state.source_app_base_url, web_driver=tor_browser_web_driver, accept_languages=locale_with_commas)
        source_app_nav.source_visits_source_homepage()
        source_app_nav.source_clicks_submit_documents_on_homepage()
        source_app_nav.source_continues_to_submit_page()
        source_codename = source_app_nav.source_retrieves_codename_from_hint()
        submitted_content = 'Confidential file with some international characters: éèö'
        source_app_nav.source_submits_a_file(file_content=submitted_content)
        source_app_nav.source_logs_out()
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_clean_state.journalist_app_base_url, web_driver=firefox_web_driver)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_clean_state.journalist_username, password=sd_servers_with_clean_state.journalist_password, otp_secret=sd_servers_with_clean_state.journalist_otp_secret)
        journ_app_nav.journalist_checks_messages()
        self._journalist_stars_and_unstars_single_message(journ_app_nav)
        retrieved_message = journ_app_nav.journalist_downloads_first_message()
        assert retrieved_message == submitted_content
        journ_app_nav.journalist_sends_reply_to_source()
        source_app_nav.source_visits_source_homepage()
        source_app_nav.source_chooses_to_login()
        source_app_nav.source_proceeds_to_login(codename=source_codename)
        save_static_data(source_app_nav.driver, locale, 'source-checks_for_reply')
        self._source_deletes_journalist_reply(source_app_nav)
        save_static_data(source_app_nav.driver, locale, 'source-deletes_reply')

    @staticmethod
    def _source_deletes_journalist_reply(navigator: SourceAppNavigator) -> None:
        if False:
            while True:
                i = 10
        reply_filename_element = navigator.driver.find_element_by_name('reply_filename')
        reply_filename = reply_filename_element.get_attribute('value')
        confirm_dialog_id = f'confirm-delete-{reply_filename}'
        navigator.nav_helper.safe_click_by_css_selector(f"a[href='#{confirm_dialog_id}']")

        def confirm_displayed():
            if False:
                print('Hello World!')
            confirm_dialog = navigator.driver.find_element_by_id(confirm_dialog_id)
            confirm_dialog.location_once_scrolled_into_view
            assert confirm_dialog.is_displayed()
        navigator.nav_helper.wait_for(confirm_displayed)
        navigator.nav_helper.safe_click_by_css_selector('#{} button'.format(confirm_dialog_id.replace('.', '\\.')))

        def reply_deleted():
            if False:
                return 10
            if not navigator.accept_languages:
                notification = navigator.driver.find_element_by_class_name('success')
                assert 'Reply deleted' in notification.text
        navigator.nav_helper.wait_for(reply_deleted)

    @staticmethod
    def _journalist_stars_and_unstars_single_message(journ_app_nav: JournalistAppNavigator) -> None:
        if False:
            return 10
        with pytest.raises(NoSuchElementException):
            journ_app_nav.driver.find_element_by_id('starred-source-link-1')
        journ_app_nav.driver.find_element_by_class_name('button-star').click()

        def message_starred():
            if False:
                for i in range(10):
                    print('nop')
            starred = journ_app_nav.driver.find_elements_by_id('starred-source-link-1')
            assert len(starred) == 1
        journ_app_nav.nav_helper.wait_for(message_starred)
        journ_app_nav.driver.find_element_by_class_name('button-star').click()

        def message_unstarred():
            if False:
                return 10
            with pytest.raises(NoSuchElementException):
                journ_app_nav.driver.find_element_by_id('starred-source-link-1')
        journ_app_nav.nav_helper.wait_for(message_unstarred)