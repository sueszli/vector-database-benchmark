import copy
import warnings
from typing import Dict, List, Optional, Type
from feast.base_feature_view import BaseFeatureView
from feast.data_source import RequestSource
from feast.feature_view_projection import FeatureViewProjection
from feast.field import Field
from feast.protos.feast.core.RequestFeatureView_pb2 import RequestFeatureView as RequestFeatureViewProto
from feast.protos.feast.core.RequestFeatureView_pb2 import RequestFeatureViewSpec
from feast.usage import log_exceptions

class RequestFeatureView(BaseFeatureView):
    """
    [Experimental] A RequestFeatureView defines a logical group of features that should
    be available as an input to an on demand feature view at request time.

    Attributes:
        name: The unique name of the request feature view.
        request_source: The request source that specifies the schema and
            features of the request feature view.
        features: The list of features defined as part of this request feature view.
        description: A human-readable description.
        tags: A dictionary of key-value pairs to store arbitrary metadata.
        owner: The owner of the request feature view, typically the email of the primary
            maintainer.
    """
    name: str
    request_source: RequestSource
    features: List[Field]
    description: str
    tags: Dict[str, str]
    owner: str

    @log_exceptions
    def __init__(self, name: str, request_data_source: RequestSource, description: str='', tags: Optional[Dict[str, str]]=None, owner: str=''):
        if False:
            while True:
                i = 10
        '\n        Creates a RequestFeatureView object.\n\n        Args:\n            name: The unique name of the request feature view.\n            request_data_source: The request data source that specifies the schema and\n                features of the request feature view.\n            description (optional): A human-readable description.\n            tags (optional): A dictionary of key-value pairs to store arbitrary metadata.\n            owner (optional): The owner of the request feature view, typically the email\n                of the primary maintainer.\n        '
        warnings.warn('Request feature view is deprecated. Please use request data source instead', DeprecationWarning)
        if isinstance(request_data_source.schema, Dict):
            new_features = [Field(name=name, dtype=dtype) for (name, dtype) in request_data_source.schema.items()]
        else:
            new_features = request_data_source.schema
        super().__init__(name=name, features=new_features, description=description, tags=tags, owner=owner)
        self.request_source = request_data_source

    @property
    def proto_class(self) -> Type[RequestFeatureViewProto]:
        if False:
            for i in range(10):
                print('nop')
        return RequestFeatureViewProto

    def to_proto(self) -> RequestFeatureViewProto:
        if False:
            return 10
        '\n        Converts an request feature view object to its protobuf representation.\n\n        Returns:\n            A RequestFeatureViewProto protobuf.\n        '
        spec = RequestFeatureViewSpec(name=self.name, request_data_source=self.request_source.to_proto(), description=self.description, tags=self.tags, owner=self.owner)
        return RequestFeatureViewProto(spec=spec)

    @classmethod
    def from_proto(cls, request_feature_view_proto: RequestFeatureViewProto):
        if False:
            while True:
                i = 10
        '\n        Creates a request feature view from a protobuf representation.\n\n        Args:\n            request_feature_view_proto: A protobuf representation of an request feature view.\n\n        Returns:\n            A RequestFeatureView object based on the request feature view protobuf.\n        '
        request_feature_view_obj = cls(name=request_feature_view_proto.spec.name, request_data_source=RequestSource.from_proto(request_feature_view_proto.spec.request_data_source), description=request_feature_view_proto.spec.description, tags=dict(request_feature_view_proto.spec.tags), owner=request_feature_view_proto.spec.owner)
        request_feature_view_obj.projection = FeatureViewProjection.from_definition(request_feature_view_obj)
        return request_feature_view_obj

    def __copy__(self):
        if False:
            return 10
        fv = RequestFeatureView(name=self.name, request_data_source=self.request_source)
        fv.projection = copy.copy(self.projection)
        return fv