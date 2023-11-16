import sys
from multiprocessing import Lock
import pytest
from flaky import flaky
from tests.integration.long_callback.utils import setup_long_callback_app

@pytest.mark.skipif(sys.version_info < (3, 7), reason='Python 3.6 long callbacks tests hangs up')
@flaky(max_runs=3)
def test_lcbc003_long_callback_running_cancel(dash_duo, manager):
    if False:
        print('Hello World!')
    lock = Lock()
    with setup_long_callback_app(manager, 'app3') as app:
        dash_duo.start_server(app)
        dash_duo.wait_for_text_to_equal('#result', 'No results', 15)
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 6)
        dash_duo.find_element('#run-button').click()
        dash_duo.wait_for_text_to_equal('#result', "Processed 'initial value'", 15)
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 6)
        input_ = dash_duo.find_element('#input')
        dash_duo.clear_input(input_)
        for key in 'hello world':
            with lock:
                input_.send_keys(key)
        dash_duo.find_element('#run-button').click()
        dash_duo.wait_for_text_to_equal('#status', 'Running', 8)
        dash_duo.find_element('#cancel-button').click()
        dash_duo.wait_for_text_to_equal('#result', "Processed 'initial value'", 12)
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 8)
        dash_duo.find_element('#run-button').click()
        dash_duo.wait_for_text_to_equal('#status', 'Running', 8)
        dash_duo.wait_for_text_to_equal('#result', "Processed 'hello world'", 8)
        dash_duo.wait_for_text_to_equal('#status', 'Finished', 8)
    assert not dash_duo.redux_state_is_loading
    assert dash_duo.get_logs() == []