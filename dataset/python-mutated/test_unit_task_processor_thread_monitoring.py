import json
import logging
from threading import Thread
from unittest.mock import mock_open, patch
from task_processor.thread_monitoring import UNHEALTHY_THREADS_FILE_PATH, clear_unhealthy_threads, get_unhealthy_thread_names, write_unhealthy_threads

def test_clear_unhealthy_threads(mocker):
    if False:
        i = 10
        return i + 15
    mocked_os = mocker.patch('task_processor.thread_monitoring.os')

    def os_path_side_effect(file_path):
        if False:
            print('Hello World!')
        return file_path == UNHEALTHY_THREADS_FILE_PATH
    mocked_os.path.exists.side_effect = os_path_side_effect
    clear_unhealthy_threads()
    mocked_os.remove.assert_called_once_with(UNHEALTHY_THREADS_FILE_PATH)

def test_write_unhealthy_threads(caplog, settings):
    if False:
        while True:
            i = 10
    task_processor_logger = logging.getLogger('task_processor')
    task_processor_logger.propagate = True
    threads = [Thread(target=lambda : None)]
    with patch('builtins.open', mock_open()) as mocked_open:
        write_unhealthy_threads(threads)
    mocked_open.assert_called_once_with(UNHEALTHY_THREADS_FILE_PATH, 'w+')
    mocked_open.return_value.write.assert_called_once_with(json.dumps([t.name for t in threads]))
    assert len(caplog.records) == 1
    assert caplog.record_tuples[0][1] == 30
    assert caplog.record_tuples[0][2] == 'Writing unhealthy threads: %s' % [t.name for t in threads]

def test_get_unhealthy_thread_names_returns_empty_list_if_file_does_not_exist(mocker):
    if False:
        print('Hello World!')
    mocked_os = mocker.patch('task_processor.thread_monitoring.os')
    mocked_os.path.exists.return_value = False
    unhealthy_thread_names = get_unhealthy_thread_names()
    assert unhealthy_thread_names == []

def test_get_unhealthy_thread_names(mocker):
    if False:
        return 10
    mocked_os = mocker.patch('task_processor.thread_monitoring.os')

    def os_path_side_effect(file_path):
        if False:
            print('Hello World!')
        return file_path == UNHEALTHY_THREADS_FILE_PATH
    mocked_os.path.exists.side_effect = os_path_side_effect
    expected_unhealthy_thread_names = ['Thread-1', 'Thread-2']
    with patch('builtins.open', mock_open(read_data=json.dumps(expected_unhealthy_thread_names))) as mocked_open:
        unhealthy_thread_names = get_unhealthy_thread_names()
    mocked_open.assert_called_once_with(UNHEALTHY_THREADS_FILE_PATH, 'r')
    assert unhealthy_thread_names == expected_unhealthy_thread_names