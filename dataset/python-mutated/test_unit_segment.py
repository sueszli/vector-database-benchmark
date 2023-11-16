import pytest
from environments.identities.models import Identity
from environments.models import Environment
from features.models import FeatureState
from integrations.segment.models import SegmentConfiguration
from integrations.segment.segment import SegmentWrapper

def test_segment_initialized_correctly():
    if False:
        i = 10
        return i + 15
    api_key = '123key'
    config = SegmentConfiguration(api_key=api_key)
    segment_wrapper = SegmentWrapper(config)
    assert segment_wrapper.analytics.write_key == api_key

@pytest.mark.django_db
def test_segment_when_generate_user_data_with_correct_values_then_success(environment: Environment, feature_state: FeatureState, feature_state_with_value: FeatureState, identity: Identity) -> None:
    if False:
        while True:
            i = 10
    api_key = '123key'
    config = SegmentConfiguration(api_key=api_key)
    segment_wrapper = SegmentWrapper(config)
    user_data = segment_wrapper.generate_user_data(identity=identity, feature_states=[feature_state, feature_state_with_value])
    feature_properties = {feature_state.feature.name: feature_state.enabled, feature_state_with_value.feature.name: 'foo'}
    expected_user_data = {'user_id': identity.identifier, 'traits': feature_properties}
    assert expected_user_data == user_data