from pathlib import Path
from typing import Generator, Tuple
from uuid import uuid4
import pytest
from sdconfig import SecureDropConfig
from tests.factories import SecureDropConfigFactory
from tests.functional.app_navigators.journalist_app_nav import JournalistAppNavigator
from tests.functional.conftest import SdServersFixtureResult, spawn_sd_servers
from tests.functional.pageslayout.utils import list_locales, save_static_data

def _create_source_and_submission_and_delete_source_key(config_in_use: SecureDropConfig) -> None:
    if False:
        i = 10
        return i + 15
    from encryption import EncryptionManager
    from tests.functional.conftest import create_source_and_submission
    (source_user, _) = create_source_and_submission(config_in_use)
    EncryptionManager.get_default().delete_source_key_pair(source_user.filesystem_id)

@pytest.fixture()
def sd_servers_with_deleted_source_key(setup_journalist_key_and_gpg_folder: Tuple[str, Path], setup_rqworker: Tuple[str, Path]) -> Generator[SdServersFixtureResult, None, None]:
    if False:
        while True:
            i = 10
    'Same as sd_servers but spawns the apps with a source whose key was deleted.\n\n    Slower than sd_servers as it is function-scoped.\n    '
    (journalist_key_fingerprint, gpg_key_dir) = setup_journalist_key_and_gpg_folder
    (worker_name, _) = setup_rqworker
    default_config = SecureDropConfigFactory.create(SECUREDROP_DATA_ROOT=Path(f'/tmp/sd-tests/functional-with-deleted-source-key-{uuid4()}'), GPG_KEY_DIR=gpg_key_dir, JOURNALIST_KEY=journalist_key_fingerprint, RQ_WORKER_NAME=worker_name)
    with spawn_sd_servers(config_to_use=default_config, journalist_app_setup_callback=_create_source_and_submission_and_delete_source_key) as sd_servers_result:
        yield sd_servers_result

@pytest.mark.parametrize('locale', list_locales())
@pytest.mark.pagelayout()
class TestJournalistLayoutCol:

    def test_col_with_and_without_documents(self, locale, sd_servers_with_submitted_file, firefox_web_driver):
        if False:
            while True:
                i = 10
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_submitted_file.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_submitted_file.journalist_username, password=sd_servers_with_submitted_file.journalist_password, otp_secret=sd_servers_with_submitted_file.journalist_otp_secret)
        journ_app_nav.journalist_visits_col()
        save_static_data(journ_app_nav.driver, locale, 'journalist-col')
        save_static_data(journ_app_nav.driver, locale, 'journalist-col_javascript')
        journ_app_nav.journalist_clicks_delete_all_and_sees_confirmation()
        journ_app_nav.journalist_confirms_delete_selected()

        def submission_deleted() -> None:
            if False:
                return 10
            submissions_after_confirming_count = journ_app_nav.count_submissions_on_current_page()
            assert submissions_after_confirming_count == 0
        journ_app_nav.nav_helper.wait_for(submission_deleted)
        save_static_data(journ_app_nav.driver, locale, 'journalist-col_no_document')

    def test_col_has_no_key(self, locale, sd_servers_with_deleted_source_key, firefox_web_driver):
        if False:
            while True:
                i = 10
        locale_with_commas = locale.replace('_', '-')
        journ_app_nav = JournalistAppNavigator(journalist_app_base_url=sd_servers_with_deleted_source_key.journalist_app_base_url, web_driver=firefox_web_driver, accept_languages=locale_with_commas)
        journ_app_nav.journalist_logs_in(username=sd_servers_with_deleted_source_key.journalist_username, password=sd_servers_with_deleted_source_key.journalist_password, otp_secret=sd_servers_with_deleted_source_key.journalist_otp_secret)
        journ_app_nav.journalist_visits_col()
        save_static_data(journ_app_nav.driver, locale, 'journalist-col_has_no_key')