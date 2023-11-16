import pytest
from tests.functional.app_navigators.journalist_app_nav import JournalistAppNavigator
from tests.functional.pageslayout.utils import list_locales, save_static_data

@pytest.mark.parametrize('locale', list_locales())
@pytest.mark.pagelayout()
class TestJournalistLayoutDelete:

    def test_delete_none(self, locale, sd_servers_with_submitted_file, firefox_web_driver):
        if False:
            while True:
                i = 10
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_submitted_file.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_submitted_file.journalist_username, password=sd_servers_with_submitted_file.journalist_password, otp_secret=sd_servers_with_submitted_file.journalist_otp_secret)
        journ_app_nav.journalist_visits_col()
        journ_app_nav.journalist_clicks_delete_selected_link()
        journ_app_nav.journalist_confirm_delete_selected()
        save_static_data(journ_app_nav.driver, locale, 'journalist-delete_none')

    def test_delete_one_confirmation(self, locale, sd_servers_with_submitted_file, firefox_web_driver):
        if False:
            i = 10
            return i + 15
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_submitted_file.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_submitted_file.journalist_username, password=sd_servers_with_submitted_file.journalist_password, otp_secret=sd_servers_with_submitted_file.journalist_otp_secret)
        journ_app_nav.journalist_visits_col()
        journ_app_nav.journalist_selects_first_doc()
        journ_app_nav.journalist_clicks_delete_selected_link()
        save_static_data(journ_app_nav.driver, locale, 'journalist-delete_one_confirmation')

    def test_delete_all_confirmation(self, locale, sd_servers_with_submitted_file, firefox_web_driver):
        if False:
            print('Hello World!')
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_submitted_file.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_submitted_file.journalist_username, password=sd_servers_with_submitted_file.journalist_password, otp_secret=sd_servers_with_submitted_file.journalist_otp_secret)
        journ_app_nav.journalist_visits_col()
        journ_app_nav.journalist_clicks_delete_all_and_sees_confirmation()
        save_static_data(journ_app_nav.driver, locale, 'journalist-delete_all_confirmation')

    def test_delete_one(self, locale, sd_servers_with_submitted_file, firefox_web_driver):
        if False:
            for i in range(10):
                print('nop')
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_submitted_file.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_submitted_file.journalist_username, password=sd_servers_with_submitted_file.journalist_password, otp_secret=sd_servers_with_submitted_file.journalist_otp_secret)
        journ_app_nav.journalist_visits_col()
        journ_app_nav.journalist_selects_first_doc()
        journ_app_nav.journalist_clicks_delete_selected_link()
        journ_app_nav.nav_helper.safe_click_by_id('delete-selected')
        save_static_data(journ_app_nav.driver, locale, 'journalist-delete_one')

    def test_delete_all(self, locale, sd_servers_with_submitted_file, firefox_web_driver):
        if False:
            for i in range(10):
                print('nop')
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_submitted_file.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_submitted_file.journalist_username, password=sd_servers_with_submitted_file.journalist_password, otp_secret=sd_servers_with_submitted_file.journalist_otp_secret)
        journ_app_nav.journalist_visits_col()
        journ_app_nav.journalist_clicks_delete_all_and_sees_confirmation()
        journ_app_nav.journalist_confirms_delete_selected()
        save_static_data(journ_app_nav.driver, locale, 'journalist-delete_all')