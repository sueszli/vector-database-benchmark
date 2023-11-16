from audit.constants import FEATURE_STATE_UPDATED_BY_CHANGE_REQUEST_MESSAGE, FEATURE_STATE_WENT_LIVE_MESSAGE
from audit.models import AuditLog
from audit.related_object_type import RelatedObjectType
from audit.tasks import create_audit_log_from_historical_record, create_feature_state_updated_by_change_request_audit_log, create_feature_state_went_live_audit_log, create_segment_priorities_changed_audit_log
from features.models import FeatureSegment
from segments.models import Segment

def test_create_audit_log_from_historical_record_does_nothing_if_no_user_or_api_key(mocker, monkeypatch):
    if False:
        i = 10
        return i + 15
    instance = mocker.MagicMock()
    instance.get_audit_log_author.return_value = None
    history_instance = mocker.MagicMock(history_id=1, instance=instance, master_api_key=None)
    mocked_historical_record_model_class = mocker.MagicMock(name='DummyHistoricalRecordModelClass')
    mocked_historical_record_model_class.objects.get.return_value = history_instance
    mocked_user_model_class = mocker.MagicMock()
    mocker.patch('audit.tasks.get_user_model', return_value=mocked_user_model_class)
    mocked_user_model_class.objects.filter.return_value.first.return_value = None
    mocked_audit_log_model_class = mocker.patch('audit.tasks.AuditLog')
    mocked_audit_log_model_class.get_history_record_model_class.return_value = mocked_historical_record_model_class
    history_record_class_path = f'app.models.{mocked_historical_record_model_class.name}'
    create_audit_log_from_historical_record(history_instance.history_id, 1, history_record_class_path)
    mocked_audit_log_model_class.objects.create.assert_not_called()

def test_create_audit_log_from_historical_record_does_nothing_if_no_log_message(mocker, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    mock_environment = mocker.MagicMock()
    instance = mocker.MagicMock()
    instance.get_audit_log_author.return_value = None
    instance.get_create_log_message.return_value = None
    instance.get_environment_and_project.return_value = (mock_environment, None)
    history_instance = mocker.MagicMock(history_id=1, instance=instance, master_api_key=None, history_type='+')
    history_user = mocker.MagicMock()
    history_user.id = 1
    mocked_historical_record_model_class = mocker.MagicMock(name='DummyHistoricalRecordModelClass')
    mocked_historical_record_model_class.objects.get.return_value = history_instance
    mocked_user_model_class = mocker.MagicMock()
    mocker.patch('audit.tasks.get_user_model', return_value=mocked_user_model_class)
    mocked_user_model_class.objects.filter.return_value.first.return_value = history_user
    mocked_audit_log_model_class = mocker.patch('audit.tasks.AuditLog')
    mocked_audit_log_model_class.get_history_record_model_class.return_value = mocked_historical_record_model_class
    history_record_class_path = f'app.models.{mocked_historical_record_model_class.name}'
    create_audit_log_from_historical_record(history_instance.history_id, history_user.id, history_record_class_path)
    mocked_audit_log_model_class.objects.create.assert_not_called()

def test_create_audit_log_from_historical_record_creates_audit_log_with_correct_fields(mocker, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    log_message = 'a log message'
    related_object_id = 1
    related_object_type = RelatedObjectType.ENVIRONMENT
    mock_environment = mocker.MagicMock()
    instance = mocker.MagicMock()
    instance.get_audit_log_author.return_value = None
    instance.get_create_log_message.return_value = log_message
    instance.get_environment_and_project.return_value = (mock_environment, None)
    instance.get_audit_log_related_object_id.return_value = related_object_id
    instance.get_audit_log_related_object_type.return_value = related_object_type
    instance.get_extra_audit_log_kwargs.return_value = {}
    history_instance = mocker.MagicMock(history_id=1, instance=instance, master_api_key=None, history_type='+')
    history_user = mocker.MagicMock()
    history_user.id = 1
    mocked_historical_record_model_class = mocker.MagicMock(name='DummyHistoricalRecordModelClass')
    mocked_historical_record_model_class.objects.get.return_value = history_instance
    mocked_user_model_class = mocker.MagicMock()
    mocker.patch('audit.tasks.get_user_model', return_value=mocked_user_model_class)
    mocked_user_model_class.objects.filter.return_value.first.return_value = history_user
    mocked_audit_log_model_class = mocker.patch('audit.tasks.AuditLog')
    mocked_audit_log_model_class.get_history_record_model_class.return_value = mocked_historical_record_model_class
    history_record_class_path = f'app.models.{mocked_historical_record_model_class.name}'
    create_audit_log_from_historical_record(history_instance.history_id, history_user.id, history_record_class_path)
    mocked_audit_log_model_class.objects.create.assert_called_once_with(history_record_id=history_instance.history_id, history_record_class_path=history_record_class_path, environment=mock_environment, project=None, author=history_user, related_object_id=related_object_id, related_object_type=related_object_type.name, log=log_message, master_api_key=None)

def test_create_segment_priorities_changed_audit_log(admin_user, feature_segment, feature, environment):
    if False:
        for i in range(10):
            print('nop')
    another_segment = Segment.objects.create(project=environment.project, name='Another Segment')
    another_feature_segment = FeatureSegment.objects.create(feature=feature, environment=environment, segment=another_segment)
    create_segment_priorities_changed_audit_log(previous_id_priority_pairs=[(feature_segment.id, 0), (another_feature_segment.id, 1)], feature_segment_ids=[feature_segment.id, another_feature_segment.id], user_id=admin_user.id)
    assert AuditLog.objects.filter(environment=environment, log=f"Segment overrides re-ordered for feature '{feature.name}'.").exists()

def test_create_feature_state_went_live_audit_log(change_request_feature_state):
    if False:
        while True:
            i = 10
    message = FEATURE_STATE_WENT_LIVE_MESSAGE % (change_request_feature_state.feature.name, change_request_feature_state.change_request.title)
    feature_state_id = change_request_feature_state.id
    create_feature_state_went_live_audit_log(feature_state_id)
    assert AuditLog.objects.filter(related_object_id=feature_state_id, is_system_event=True, log=message).count() == 1

def test_create_feature_state_updated_by_change_request_audit_log(change_request_feature_state):
    if False:
        while True:
            i = 10
    message = FEATURE_STATE_UPDATED_BY_CHANGE_REQUEST_MESSAGE % (change_request_feature_state.feature.name, change_request_feature_state.change_request.title)
    feature_state_id = change_request_feature_state.id
    create_feature_state_updated_by_change_request_audit_log(feature_state_id)
    assert AuditLog.objects.filter(related_object_id=feature_state_id, is_system_event=True, log=message).count() == 1

def test_create_feature_state_updated_by_change_request_audit_log_does_nothing_if_feature_state_deleted(change_request_feature_state):
    if False:
        print('Hello World!')
    change_request_feature_state.delete()
    feature_state_id = change_request_feature_state.id
    create_feature_state_went_live_audit_log(feature_state_id)
    assert AuditLog.objects.filter(related_object_id=feature_state_id, is_system_event=True).count() == 0

def test_create_feature_state_wen_live_audit_log_does_nothing_if_feature_state_deleted(change_request_feature_state):
    if False:
        return 10
    message = FEATURE_STATE_WENT_LIVE_MESSAGE % (change_request_feature_state.feature.name, change_request_feature_state.change_request.title)
    change_request_feature_state.delete()
    feature_state_id = change_request_feature_state.id
    create_feature_state_went_live_audit_log(feature_state_id)
    assert AuditLog.objects.filter(related_object_id=feature_state_id, is_system_event=True, log=message).count() == 0