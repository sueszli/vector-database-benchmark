from integrations.amplitude.models import AmplitudeConfiguration
from integrations.common.models import EnvironmentIntegrationModel
from integrations.common.wrapper import AbstractBaseIdentityIntegrationWrapper
from integrations.integration import identify_integrations
from integrations.segment.models import SegmentConfiguration

def test_identify_integrations_amplitude_called(mocker, environment, identity):
    if False:
        print('Hello World!')
    mock_amplitude_wrapper = mocker.patch('integrations.amplitude.amplitude.AmplitudeWrapper.identify_user_async')
    AmplitudeConfiguration.objects.create(api_key='abc-123', environment=environment)
    identify_integrations(identity, identity.get_all_feature_states())
    mock_amplitude_wrapper.assert_called()

def test_identify_integrations_segment_called(mocker, environment, identity):
    if False:
        return 10
    mock_segment_wrapper = mocker.patch('integrations.segment.segment.SegmentWrapper.identify_user_async')
    SegmentConfiguration.objects.create(api_key='abc-123', environment=environment)
    identify_integrations(identity, identity.get_all_feature_states())
    mock_segment_wrapper.assert_called()

def test_identify_integrations_calls_every_integration_in_identity_integrations_dict(mocker, identity):
    if False:
        return 10
    integration_wrapper_a = mocker.MagicMock(autospec=AbstractBaseIdentityIntegrationWrapper)
    integration_wrapper_b = mocker.MagicMock(autospec=AbstractBaseIdentityIntegrationWrapper)
    integration_a_config = mocker.MagicMock(autospec=EnvironmentIntegrationModel, deleted=False)
    integration_b_config = mocker.MagicMock(autospec=EnvironmentIntegrationModel, deleted=False)
    identity.environment.integration_a_config = integration_a_config
    identity.environment.integration_b_config = integration_b_config
    identity_integrations = [{'relation_name': 'integration_a_config', 'wrapper': integration_wrapper_a}, {'relation_name': 'integration_b_config', 'wrapper': integration_wrapper_b}]
    mocker.patch('integrations.integration.IDENTITY_INTEGRATIONS', identity_integrations)
    identify_integrations(identity, identity.get_all_feature_states())
    integration_wrapper_a.assert_called_with(integration_a_config)
    integration_a_mocked_generate_user_data = integration_wrapper_a.return_value.generate_user_data
    integration_a_mocked_generate_user_data.assert_called_with(identity=identity, feature_states=identity.get_all_feature_states(), trait_models=None)
    integration_wrapper_a.return_value.identify_user_async.assert_called_with(data=integration_a_mocked_generate_user_data.return_value)
    integration_wrapper_b.assert_called_with(integration_b_config)
    integration_b_mocked_generate_user_data = integration_wrapper_b.return_value.generate_user_data
    integration_b_mocked_generate_user_data.assert_called_with(identity=identity, feature_states=identity.get_all_feature_states(), trait_models=None)
    integration_wrapper_b.return_value.identify_user_async.assert_called_with(data=integration_b_mocked_generate_user_data.return_value)

def test_identify_integrations_does_not_call_deleted_integrations(mocker, environment, identity):
    if False:
        print('Hello World!')
    mock_segment_wrapper = mocker.patch('integrations.segment.segment.SegmentWrapper.identify_user_async')
    sc = SegmentConfiguration.objects.create(api_key='abc-123', environment=environment)
    sc.delete()
    identify_integrations(identity, identity.get_all_feature_states())
    mock_segment_wrapper.assert_not_called()