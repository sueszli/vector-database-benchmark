from integrations.amplitude.models import AmplitudeConfiguration

def test_amplitude_configuration_save_writes_environment_to_dynamodb(environment, mocker):
    if False:
        return 10
    "\n    Test to verify that AmplitudeConfiguration's base model class works as expected\n    "
    amplitude_config = AmplitudeConfiguration(environment=environment, api_key='api-key', base_url='https://base.url.com')
    mock_environment_model_class = mocker.patch('integrations.common.models.Environment')
    amplitude_config.save()
    mock_environment_model_class.write_environments_to_dynamodb.assert_called_once_with(environment_id=environment.id)

def test_amplitude_configuration_delete_writes_environment_to_dynamodb(environment, mocker):
    if False:
        return 10
    "\n    Test to verify that AmplitudeConfiguration's base model class works as expected\n    "
    mock_environment_model_class = mocker.patch('integrations.common.models.Environment')
    amplitude_config = AmplitudeConfiguration.objects.create(environment=environment, api_key='api-key', base_url='https://base.url.com')
    mock_environment_model_class.reset_mock()
    amplitude_config.delete()
    mock_environment_model_class.write_environments_to_dynamodb.assert_called_once_with(environment_id=environment.id)

def test_amplitude_configuration_update_clears_environment_cache(environment, mocker):
    if False:
        return 10
    mock_environment_cache = mocker.patch('environments.models.environment_cache')
    amplitude_config = AmplitudeConfiguration.objects.create(environment=environment, api_key='api-key', base_url='https://base.url.com')
    amplitude_config.api_key += 'update'
    amplitude_config.save()
    mock_environment_cache.delete.assert_called_once_with(environment.api_key)