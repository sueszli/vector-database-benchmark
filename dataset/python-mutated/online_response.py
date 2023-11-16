from typing import Any, Dict, List
import pandas as pd
from feast.feature_view import DUMMY_ENTITY_ID
from feast.protos.feast.serving.ServingService_pb2 import GetOnlineFeaturesResponse
from feast.type_map import feast_value_type_to_python_type
TIMESTAMP_POSTFIX: str = '__ts'

class OnlineResponse:
    """
    Defines an online response in feast.
    """

    def __init__(self, online_response_proto: GetOnlineFeaturesResponse):
        if False:
            while True:
                i = 10
        '\n        Construct a native online response from its protobuf version.\n\n        Args:\n        online_response_proto: GetOnlineResponse proto object to construct from.\n        '
        self.proto = online_response_proto
        for (idx, val) in enumerate(self.proto.metadata.feature_names.val):
            if val == DUMMY_ENTITY_ID:
                del self.proto.metadata.feature_names.val[idx]
                del self.proto.results[idx]
                break

    def to_dict(self, include_event_timestamps: bool=False) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts GetOnlineFeaturesResponse features into a dictionary form.\n\n        Args:\n        is_with_event_timestamps: bool Optionally include feature timestamps in the dictionary\n        '
        response: Dict[str, List[Any]] = {}
        for (feature_ref, feature_vector) in zip(self.proto.metadata.feature_names.val, self.proto.results):
            response[feature_ref] = [feast_value_type_to_python_type(v) for v in feature_vector.values]
            if include_event_timestamps:
                timestamp_ref = feature_ref + TIMESTAMP_POSTFIX
                response[timestamp_ref] = [ts.seconds for ts in feature_vector.event_timestamps]
        return response

    def to_df(self, include_event_timestamps: bool=False) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts GetOnlineFeaturesResponse features into Panda dataframe form.\n\n        Args:\n        is_with_event_timestamps: bool Optionally include feature timestamps in the dataframe\n        '
        return pd.DataFrame(self.to_dict(include_event_timestamps))