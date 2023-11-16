from django.utils import timezone
from environments.identities.models import Identity
from features.models import Feature, FeatureState

def test_identity_get_all_feature_states_gets_latest_committed_version(environment):
    if False:
        for i in range(10):
            print('nop')
    identity = Identity.objects.create(identifier='identity', environment=environment)
    feature = Feature.objects.create(name='versioned_feature', project=environment.project, default_enabled=False, initial_value='v1')
    now = timezone.now()
    feature_state_v2 = FeatureState.objects.create(feature=feature, version=2, live_from=now, enabled=True, environment=environment)
    feature_state_v2.feature_state_value.string_value = 'v2'
    feature_state_v2.feature_state_value.save()
    not_live_feature_state = FeatureState.objects.create(feature=feature, version=None, live_from=None, enabled=False, environment=environment)
    not_live_feature_state.feature_state_value.string_value = 'v3'
    not_live_feature_state.feature_state_value.save()
    identity_feature_states = identity.get_all_feature_states()
    identity_feature_state = next(filter(lambda fs: fs.feature == feature, identity_feature_states))
    assert identity_feature_state.get_feature_state_value() == 'v2'

def test_get_hash_key_with_use_identity_composite_key_for_hashing_enabled(identity: Identity):
    if False:
        return 10
    assert identity.get_hash_key(use_identity_composite_key_for_hashing=True) == f'{identity.environment.api_key}_{identity.identifier}'

def test_get_hash_key_with_use_identity_composite_key_for_hashing_disabled(identity: Identity):
    if False:
        i = 10
        return i + 15
    assert identity.get_hash_key(use_identity_composite_key_for_hashing=False) == str(identity.id)