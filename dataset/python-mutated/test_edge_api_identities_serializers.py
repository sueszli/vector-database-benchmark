from django.utils import timezone
from flag_engine.features.models import FeatureModel, FeatureStateModel
from edge_api.identities.models import EdgeIdentity
from edge_api.identities.serializers import EdgeIdentityFeatureStateSerializer
from environments.identities.serializers import IdentityAllFeatureStatesSerializer
from features.feature_types import STANDARD
from features.models import FeatureState
from util.mappers import map_identity_to_identity_document
from webhooks.constants import WEBHOOK_DATETIME_FORMAT

def test_edge_identity_feature_state_serializer_save_allows_missing_mvfsvs(mocker, identity, feature, admin_user):
    if False:
        while True:
            i = 10
    identity_model = EdgeIdentity.from_identity_document(map_identity_to_identity_document(identity))
    view = mocker.MagicMock(identity=identity_model)
    request = mocker.MagicMock(user=admin_user, master_api_key=None)
    serializer = EdgeIdentityFeatureStateSerializer(data={'feature_state_value': 'foo', 'feature': feature.id}, context={'view': view, 'request': request})
    mock_dynamo_wrapper = mocker.patch('edge_api.identities.serializers.EdgeIdentity.dynamo_wrapper')
    serializer.is_valid(raise_exception=True)
    result = serializer.save()
    assert result
    mock_dynamo_wrapper.put_item.assert_called_once()
    saved_identity_record = mock_dynamo_wrapper.put_item.call_args[0][0]
    assert saved_identity_record['identifier'] == identity.identifier
    assert len(saved_identity_record['identity_features']) == 1
    saved_identity_feature_state = saved_identity_record['identity_features'][0]
    assert saved_identity_feature_state['multivariate_feature_state_values'] == []
    assert saved_identity_feature_state['featurestate_uuid']
    assert saved_identity_feature_state['enabled'] is False
    assert saved_identity_feature_state['feature']['id'] == feature.id

def test_edge_identity_feature_state_serializer_save_calls_webhook_for_new_override(mocker, identity, feature, admin_user):
    if False:
        print('Hello World!')
    identity_model = EdgeIdentity.from_identity_document(map_identity_to_identity_document(identity))
    view = mocker.MagicMock(identity=identity_model)
    request = mocker.MagicMock(user=admin_user, master_api_key=None)
    new_enabled_state = True
    new_value = 'foo'
    serializer = EdgeIdentityFeatureStateSerializer(data={'feature_state_value': new_value, 'enabled': new_enabled_state, 'feature': feature.id}, context={'view': view, 'request': request})
    mocker.patch('edge_api.identities.serializers.EdgeIdentity.dynamo_wrapper')
    mock_call_environment_webhook = mocker.patch('edge_api.identities.serializers.call_environment_webhook_for_feature_state_change')
    now = timezone.now()
    mocker.patch('edge_api.identities.serializers.timezone.now', return_value=now)
    serializer.is_valid(raise_exception=True)
    serializer.save()
    mock_call_environment_webhook.delay.assert_called_once_with(kwargs={'feature_id': feature.id, 'environment_api_key': identity.environment.api_key, 'identity_id': identity.id, 'identity_identifier': identity.identifier, 'changed_by_user_id': admin_user.id, 'new_enabled_state': new_enabled_state, 'new_value': new_value, 'previous_enabled_state': None, 'previous_value': None, 'timestamp': now.strftime(WEBHOOK_DATETIME_FORMAT)})

def test_edge_identity_feature_state_serializer_save_calls_webhook_for_update(mocker, identity, feature, admin_user):
    if False:
        print('Hello World!')
    identity_model = EdgeIdentity.from_identity_document(map_identity_to_identity_document(identity))
    view = mocker.MagicMock(identity=identity_model)
    request = mocker.MagicMock(user=admin_user)
    previous_enabled_state = False
    previous_value = 'foo'
    new_enabled_state = True
    new_value = 'bar'
    instance = FeatureStateModel(feature=FeatureModel(id=feature.id, name=feature.name, type=STANDARD), enabled=previous_enabled_state)
    instance.set_value(previous_value)
    serializer = EdgeIdentityFeatureStateSerializer(instance=instance, data={'feature_state_value': new_value, 'enabled': new_enabled_state, 'feature': feature.id}, context={'view': view, 'request': request})
    mocker.patch('edge_api.identities.serializers.EdgeIdentity.dynamo_wrapper')
    mock_call_environment_webhook = mocker.patch('edge_api.identities.serializers.call_environment_webhook_for_feature_state_change')
    now = timezone.now()
    mocker.patch('edge_api.identities.serializers.timezone.now', return_value=now)
    serializer.is_valid(raise_exception=True)
    serializer.save()
    mock_call_environment_webhook.delay.assert_called_once_with(kwargs={'feature_id': feature.id, 'environment_api_key': identity.environment.api_key, 'identity_id': identity.id, 'identity_identifier': identity.identifier, 'changed_by_user_id': admin_user.id, 'new_enabled_state': new_enabled_state, 'new_value': new_value, 'previous_enabled_state': previous_enabled_state, 'previous_value': previous_value, 'timestamp': now.strftime(WEBHOOK_DATETIME_FORMAT)})

def test_all_feature_states_serializer_get_feature_state_value_uses_mv_values_for_edge(identity, multivariate_feature, environment):
    if False:
        while True:
            i = 10
    identity_document = map_identity_to_identity_document(identity)
    del identity_document['django_id']
    identity_model = EdgeIdentity.from_identity_document(identity_document)
    feature_state = FeatureState.objects.get(feature=multivariate_feature, environment=environment)
    serializer = IdentityAllFeatureStatesSerializer(context={'identity': identity_model, 'environment_api_key': environment.api_key})
    value = serializer.get_feature_state_value(instance=feature_state)
    assert value != multivariate_feature.initial_value