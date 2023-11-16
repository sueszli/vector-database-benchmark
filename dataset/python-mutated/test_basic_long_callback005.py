import sys
from multiprocessing import Lock
import pytest
from tests.integration.long_callback.utils import setup_long_callback_app

@pytest.mark.skipif(sys.version_info < (3, 7), reason='Python 3.6 long callbacks tests hangs up')
@pytest.mark.skip(reason='Timeout often')
def test_lcbc005_long_callback_caching(dash_duo, manager):
    if False:
        i = 10
        return i + 15
    lock = Lock()
    with setup_long_callback_app(manager, 'app5') as app:
        dash_duo.start_server(app)
        dash_duo.wait_for_text_to_equal('#status', 'Progress 2/4', 15)
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 15)
        dash_duo.wait_for_text_to_equal('#result', "Result for 'AAA'", 8)
        input_ = dash_duo.find_element('#input')
        dash_duo.clear_input(input_)
        for key in 'BBB':
            with lock:
                input_.send_keys(key)
        dash_duo.find_element('#run-button').click()
        dash_duo.wait_for_text_to_equal('#status', 'Progress 2/4', 20)
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 12)
        dash_duo.wait_for_text_to_equal('#result', "Result for 'BBB'", 8)
        input_ = dash_duo.find_element('#input')
        dash_duo.clear_input(input_)
        for key in 'AAA':
            with lock:
                input_.send_keys(key)
        dash_duo.find_element('#run-button').click()
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 8)
        dash_duo.wait_for_text_to_equal('#result', "Result for 'AAA'", 8)
        input_ = dash_duo.find_element('#input')
        dash_duo.clear_input(input_)
        for key in 'BBB':
            with lock:
                input_.send_keys(key)
        dash_duo.find_element('#run-button').click()
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 8)
        dash_duo.wait_for_text_to_equal('#result', "Result for 'BBB'", 8)
        input_ = dash_duo.find_element('#input')
        dash_duo.clear_input(input_)
        for key in 'AAA':
            with lock:
                input_.send_keys(key)
        app._cache_key.value = 1
        dash_duo.find_element('#run-button').click()
        dash_duo.wait_for_text_to_equal('#status', 'Progress 2/4', 20)
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 12)
        dash_duo.wait_for_text_to_equal('#result', "Result for 'AAA'", 8)
        assert not dash_duo.redux_state_is_loading
        assert dash_duo.get_logs() == []