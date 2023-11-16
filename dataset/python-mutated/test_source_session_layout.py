import time
from pathlib import Path
from typing import Generator, Tuple
import pytest
from tests.factories import SecureDropConfigFactory
from tests.functional.app_navigators.source_app_nav import SourceAppNavigator
from tests.functional.conftest import SdServersFixtureResult, spawn_sd_servers
from tests.functional.pageslayout.utils import list_locales, save_static_data
SESSION_EXPIRATION_SECONDS = 3

@pytest.fixture(scope='session')
def sd_servers_with_short_timeout(setup_journalist_key_and_gpg_folder: Tuple[str, Path], setup_rqworker: Tuple[str, str]) -> Generator[SdServersFixtureResult, None, None]:
    if False:
        return 10
    'Spawn the source and journalist apps as separate processes with a short session timeout.'
    (journalist_key_fingerprint, gpg_key_dir) = setup_journalist_key_and_gpg_folder
    (worker_name, _) = setup_rqworker
    config_with_short_timeout = SecureDropConfigFactory.create(SESSION_EXPIRATION_MINUTES=SESSION_EXPIRATION_SECONDS / 60, SECUREDROP_DATA_ROOT=Path('/tmp/sd-tests/functional-session-timeout'), GPG_KEY_DIR=gpg_key_dir, JOURNALIST_KEY=journalist_key_fingerprint, RQ_WORKER_NAME=worker_name)
    with spawn_sd_servers(config_to_use=config_with_short_timeout) as sd_servers_result:
        yield sd_servers_result

@pytest.mark.parametrize('locale', list_locales())
@pytest.mark.pagelayout()
class TestSourceAppSessionTimeout:

    def test_source_session_timeout(self, locale, sd_servers_with_short_timeout):
        if False:
            i = 10
            return i + 15
        locale_with_commas = locale.replace('_', '-')
        with SourceAppNavigator.using_tor_browser_web_driver(source_app_base_url=sd_servers_with_short_timeout.source_app_base_url, accept_languages=locale_with_commas) as navigator:
            navigator.source_visits_source_homepage()
            navigator.source_clicks_submit_documents_on_homepage()
            navigator.source_continues_to_submit_page()
            time.sleep(SESSION_EXPIRATION_SECONDS + 1)
            navigator.driver.refresh()
            notification = navigator.driver.find_element_by_class_name('error')
            assert notification.text
            if locale == 'en_US':
                expected_text = 'You were logged out due to inactivity.'
                assert expected_text in notification.text
            save_static_data(navigator.driver, locale, 'source-session_timeout')