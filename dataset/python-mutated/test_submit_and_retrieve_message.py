from tests.functional.app_navigators.journalist_app_nav import JournalistAppNavigator
from tests.functional.app_navigators.source_app_nav import SourceAppNavigator

class TestSubmitAndRetrieveMessage:

    def test_submit_and_retrieve_happy_path(self, sd_servers_with_clean_state, tor_browser_web_driver, firefox_web_driver):
        if False:
            while True:
                i = 10
        source_app_nav = SourceAppNavigator(source_app_base_url=sd_servers_with_clean_state.source_app_base_url, web_driver=tor_browser_web_driver)
        source_app_nav.source_visits_source_homepage()
        source_app_nav.source_clicks_submit_documents_on_homepage()
        source_app_nav.source_continues_to_submit_page()
        submitted_message = 'Confidential message with some international characters: éèö'
        source_app_nav.source_submits_a_message(message=submitted_message)
        source_app_nav.source_logs_out()
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_clean_state.journalist_app_base_url, web_driver=firefox_web_driver)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_clean_state.journalist_username, password=sd_servers_with_clean_state.journalist_password, otp_secret=sd_servers_with_clean_state.journalist_otp_secret)
        journ_app_nav.journalist_checks_messages()
        retrieved_message = journ_app_nav.journalist_downloads_first_message()
        assert retrieved_message == submitted_message