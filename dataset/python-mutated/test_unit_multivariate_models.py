from unittest.mock import MagicMock
from features.feature_types import MULTIVARIATE, STANDARD
from features.models import FeatureSegment, FeatureState
from features.multivariate.models import MultivariateFeatureOption, MultivariateFeatureStateValue

def test_multivariate_feature_option_get_create_log_message(feature):
    if False:
        print('Hello World!')
    mvfo = MultivariateFeatureOption.objects.create(feature=feature, string_value='foo')
    history_instance = MagicMock()
    msg = mvfo.get_create_log_message(history_instance)
    assert msg == f"Multivariate option added to feature '{feature.name}'."

def test_multivariate_feature_option_get_delete_log_message_for_valid_feature(feature):
    if False:
        i = 10
        return i + 15
    mvfo = MultivariateFeatureOption.objects.create(feature=feature, string_value='foo')
    history_instance = MagicMock()
    msg = mvfo.get_delete_log_message(history_instance)
    assert msg == f"Multivariate option removed from feature '{feature.name}'."

def test_multivariate_feature_option_get_delete_log_message_for_deleted_feature(feature):
    if False:
        for i in range(10):
            print('nop')
    mvfo = MultivariateFeatureOption.objects.create(feature=feature, string_value='foo')
    mvfo.refresh_from_db()
    feature.delete()
    history_instance = MagicMock()
    msg = mvfo.get_delete_log_message(history_instance)
    assert msg is None

def test_multivariate_feature_state_value_get_update_log_message_environment_default(multivariate_feature, environment):
    if False:
        return 10
    history_instance = MagicMock()
    mvfsv = MultivariateFeatureStateValue.objects.filter(feature_state__environment=environment).first()
    msg = mvfsv.get_update_log_message(history_instance)
    assert msg == f"Multivariate value changed for feature '{multivariate_feature.name}'."

def test_multivariate_feature_state_value_get_update_log_message_identity_override(multivariate_feature, environment, identity):
    if False:
        while True:
            i = 10
    history_instance = MagicMock()
    identity_override = FeatureState.objects.create(feature=multivariate_feature, identity=identity, environment=environment)
    mvfsv = identity_override.multivariate_feature_state_values.create(multivariate_feature_option=multivariate_feature.multivariate_options.first(), percentage_allocation=100)
    msg = mvfsv.get_update_log_message(history_instance)
    assert msg == f"Multivariate value changed for feature '{multivariate_feature.name}' and identity '{identity.identifier}'."

def test_multivariate_feature_state_value_get_update_log_message_segment_override(multivariate_feature, environment, segment):
    if False:
        for i in range(10):
            print('nop')
    history_instance = MagicMock()
    feature_segment = FeatureSegment.objects.create(segment=segment, feature=multivariate_feature, environment=environment)
    segment_override = FeatureState.objects.create(feature=multivariate_feature, feature_segment=feature_segment, environment=environment)
    mvfsv = segment_override.multivariate_feature_state_values.create(multivariate_feature_option=multivariate_feature.multivariate_options.first(), percentage_allocation=100)
    msg = mvfsv.get_update_log_message(history_instance)
    assert msg == f"Multivariate value changed for feature '{multivariate_feature.name}' and segment '{segment.name}'."

def test_deleting_last_mv_option_of_mulitvariate_feature_converts_it_into_standard(multivariate_feature):
    if False:
        return 10
    assert multivariate_feature.type == MULTIVARIATE
    mv_options = multivariate_feature.multivariate_options.all()
    mv_options[0].delete()
    multivariate_feature.refresh_from_db()
    assert multivariate_feature.type == MULTIVARIATE
    for mv_option in mv_options:
        mv_option.delete()
    multivariate_feature.refresh_from_db()
    assert multivariate_feature.type == STANDARD

def test_adding_mv_option_to_standard_feature_converts_it_into_multivariate(feature):
    if False:
        return 10
    assert feature.type == STANDARD
    MultivariateFeatureOption.objects.create(feature=feature, string_value='foo')
    feature.refresh_from_db()
    assert feature.type == MULTIVARIATE