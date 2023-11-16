import pytest
from tbselenium.utils import disable_js
from tests.functional.app_navigators.source_app_nav import SourceAppNavigator
from tests.functional.pageslayout.utils import list_locales, save_static_data

@pytest.mark.parametrize('locale', list_locales())
@pytest.mark.pagelayout()
class TestSourceLayoutTorBrowser:

    def test_index_and_logout(self, locale, sd_servers):
        if False:
            print('Hello World!')
        locale_with_commas = locale.replace('_', '-')
        with SourceAppNavigator.using_tor_browser_web_driver(source_app_base_url=sd_servers.source_app_base_url, accept_languages=locale_with_commas) as navigator:
            disable_js(navigator.driver)
            navigator.source_visits_source_homepage()
            save_static_data(navigator.driver, locale, 'source-index')
            navigator.source_clicks_submit_documents_on_homepage()
            navigator.source_continues_to_submit_page()
            navigator.source_logs_out()
            save_static_data(navigator.driver, locale, 'source-logout_page')