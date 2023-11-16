from typing import List, NamedTuple
from feast.data_source import DataSource
from feast.entity import Entity
from feast.feature_service import FeatureService
from feast.feature_view import FeatureView
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto
from feast.request_feature_view import RequestFeatureView
from feast.stream_feature_view import StreamFeatureView

class RepoContents(NamedTuple):
    """
    Represents the objects in a Feast feature repo.
    """
    data_sources: List[DataSource]
    feature_views: List[FeatureView]
    on_demand_feature_views: List[OnDemandFeatureView]
    request_feature_views: List[RequestFeatureView]
    stream_feature_views: List[StreamFeatureView]
    entities: List[Entity]
    feature_services: List[FeatureService]

    def to_registry_proto(self) -> RegistryProto:
        if False:
            i = 10
            return i + 15
        registry_proto = RegistryProto()
        registry_proto.data_sources.extend([e.to_proto() for e in self.data_sources])
        registry_proto.entities.extend([e.to_proto() for e in self.entities])
        registry_proto.feature_views.extend([fv.to_proto() for fv in self.feature_views])
        registry_proto.on_demand_feature_views.extend([fv.to_proto() for fv in self.on_demand_feature_views])
        registry_proto.request_feature_views.extend([fv.to_proto() for fv in self.request_feature_views])
        registry_proto.feature_services.extend([fs.to_proto() for fs in self.feature_services])
        registry_proto.stream_feature_views.extend([fv.to_proto() for fv in self.stream_feature_views])
        return registry_proto