import logging
import typing
import rudder_analytics
from environments.identities.models import Identity
from environments.identities.traits.models import Trait
from features.models import FeatureState
from integrations.common.wrapper import AbstractBaseIdentityIntegrationWrapper
from .models import RudderstackConfiguration
logger = logging.getLogger(__name__)

class RudderstackWrapper(AbstractBaseIdentityIntegrationWrapper):

    def __init__(self, config: RudderstackConfiguration):
        if False:
            while True:
                i = 10
        rudder_analytics.write_key = config.api_key
        rudder_analytics.data_plane_url = config.base_url

    def _identify_user(self, user_data: dict) -> None:
        if False:
            return 10
        rudder_analytics.identify(**user_data)

    def generate_user_data(self, identity: Identity, feature_states: typing.List[FeatureState], trait_models: typing.List[Trait]=None) -> dict:
        if False:
            print('Hello World!')
        feature_properties = {}
        for feature_state in feature_states:
            value = feature_state.get_feature_state_value(identity=identity)
            feature_properties[feature_state.feature.name] = value if feature_state.enabled and value else feature_state.enabled
        return {'user_id': identity.identifier, 'traits': feature_properties}