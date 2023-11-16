import pytest
from pytest_lazyfixture import lazy_fixture
from sse.sse_service import send_environment_update_message_for_environment, send_environment_update_message_for_project

def test_send_environment_update_message_for_project_schedules_task_correctly(mocker, sse_enabled_settings, realtime_enabled_project):
    if False:
        while True:
            i = 10
    mocked_tasks = mocker.patch('sse.sse_service.tasks', autospec=True)
    send_environment_update_message_for_project(realtime_enabled_project)
    mocked_tasks.send_environment_update_message_for_project.delay.assert_called_once_with(args=(realtime_enabled_project.id,))

@pytest.mark.parametrize('test_settings, test_project', [(lazy_fixture('sse_enabled_settings'), lazy_fixture('project')), (lazy_fixture('sse_disabled_settings'), lazy_fixture('realtime_enabled_project'))])
def test_send_environment_update_message_for_project_exits_early_without_scheduling_task(mocker, test_settings, test_project):
    if False:
        return 10
    mocked_tasks = mocker.patch('sse.sse_service.tasks', autospec=True)
    send_environment_update_message_for_project(test_project)
    mocked_tasks.send_environment_update_message_for_project.delay.assert_not_called()

@pytest.mark.parametrize('test_settings, test_environment ', [(lazy_fixture('sse_enabled_settings'), lazy_fixture('environment')), (lazy_fixture('sse_disabled_settings'), lazy_fixture('realtime_enabled_project_environment_one'))])
def test_send_environment_update_message_for_environment_exits_early_without_scheduling_task(mocker, test_settings, test_environment):
    if False:
        return 10
    mocked_tasks = mocker.patch('sse.sse_service.tasks', autospec=True)
    send_environment_update_message_for_environment(test_environment)
    mocked_tasks.send_environment_update_message.delay.assert_not_called()

def test_send_environment_update_message_for_environment_schedules_task_correctly(mocker, sse_enabled_settings, realtime_enabled_project_environment_one):
    if False:
        return 10
    mocked_tasks = mocker.patch('sse.sse_service.tasks', autospec=True)
    send_environment_update_message_for_environment(realtime_enabled_project_environment_one)
    mocked_tasks.send_environment_update_message.delay.assert_called_once_with(args=(realtime_enabled_project_environment_one.api_key, realtime_enabled_project_environment_one.updated_at.isoformat()))