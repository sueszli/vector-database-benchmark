from task_processor.health import is_processor_healthy
from task_processor.models import HealthCheckModel
from task_processor.task_run_method import TaskRunMethod

def test_is_processor_healthy_returns_false_if_task_not_processed(mocker):
    if False:
        return 10
    mocker.patch('task_processor.health.create_health_check_model')
    mocked_health_check_model_class = mocker.patch('task_processor.health.HealthCheckModel')
    mocked_health_check_model_class.objects.filter.return_value.first.return_value = None
    result = is_processor_healthy(max_tries=3)
    assert result is False

def test_is_processor_healthy_returns_true_if_task_processed(db, settings):
    if False:
        i = 10
        return i + 15
    settings.TASK_RUN_METHOD = TaskRunMethod.SYNCHRONOUSLY
    result = is_processor_healthy(max_tries=3)
    assert result is True
    assert not HealthCheckModel.objects.exists()