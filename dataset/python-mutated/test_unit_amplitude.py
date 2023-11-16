import pytest
from environments.identities.models import Identity
from environments.models import Environment
from features.models import FeatureState
from integrations.amplitude.amplitude import AmplitudeWrapper
from integrations.amplitude.constants import DEFAULT_AMPLITUDE_API_URL
from integrations.amplitude.models import AmplitudeConfiguration

def test_amplitude_initialized_correctly():
    if False:
        return 10
    config = AmplitudeConfiguration(api_key='123key')
    amplitude_wrapper = AmplitudeWrapper(config)
    expected_url = f'{DEFAULT_AMPLITUDE_API_URL}/identify'
    assert amplitude_wrapper.url == expected_url

def test_amplitude_initialized_correctly_with_custom_base_url():
    if False:
        print('Hello World!')
    base_url = 'https://api.eu.amplitude.com'
    config = AmplitudeConfiguration(api_key='123key', base_url=base_url)
    amplitude_wrapper = AmplitudeWrapper(config)
    assert amplitude_wrapper.url == f'{base_url}/identify'

@pytest.mark.django_db
def test_amplitude_when_generate_user_data_with_correct_values_then_success(environment: Environment, feature_state: FeatureState, feature_state_with_value: FeatureState, identity: Identity) -> None:
    if False:
        print('Hello World!')
    api_key = '123key'
    config = AmplitudeConfiguration(api_key=api_key)
    amplitude_wrapper = AmplitudeWrapper(config)
    user_data = amplitude_wrapper.generate_user_data(identity=identity, feature_states=[feature_state, feature_state_with_value])
    feature_properties = {feature_state.feature.name: feature_state.enabled, feature_state_with_value.feature.name: 'foo'}
    expected_user_data = {'user_id': identity.identifier, 'user_properties': feature_properties}
    assert expected_user_data == user_data