from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import timestamp_pb2
import proto
__protobuf__ = proto.module(package='google.cloud.networkservices.v1', manifest={'ServiceBinding', 'ListServiceBindingsRequest', 'ListServiceBindingsResponse', 'GetServiceBindingRequest', 'CreateServiceBindingRequest', 'DeleteServiceBindingRequest'})

class ServiceBinding(proto.Message):
    """ServiceBinding is the resource that defines a Service
    Directory Service to be used in a BackendService resource.

    Attributes:
        name (str):
            Required. Name of the ServiceBinding resource. It matches
            pattern
            ``projects/*/locations/global/serviceBindings/service_binding_name``.
        description (str):
            Optional. A free-text description of the
            resource. Max length 1024 characters.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when the resource
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when the resource
            was updated.
        service (str):
            Required. The full service directory service name of the
            format /projects/*/locations/*/namespaces/*/services/*
        labels (MutableMapping[str, str]):
            Optional. Set of label tags associated with
            the ServiceBinding resource.
    """
    name: str = proto.Field(proto.STRING, number=1)
    description: str = proto.Field(proto.STRING, number=2)
    create_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    update_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)
    service: str = proto.Field(proto.STRING, number=5)
    labels: MutableMapping[str, str] = proto.MapField(proto.STRING, proto.STRING, number=7)

class ListServiceBindingsRequest(proto.Message):
    """Request used with the ListServiceBindings method.

    Attributes:
        parent (str):
            Required. The project and location from which the
            ServiceBindings should be listed, specified in the format
            ``projects/*/locations/global``.
        page_size (int):
            Maximum number of ServiceBindings to return
            per call.
        page_token (str):
            The value returned by the last
            ``ListServiceBindingsResponse`` Indicates that this is a
            continuation of a prior ``ListRouters`` call, and that the
            system should return the next page of data.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)

class ListServiceBindingsResponse(proto.Message):
    """Response returned by the ListServiceBindings method.

    Attributes:
        service_bindings (MutableSequence[google.cloud.network_services_v1.types.ServiceBinding]):
            List of ServiceBinding resources.
        next_page_token (str):
            If there might be more results than those appearing in this
            response, then ``next_page_token`` is included. To get the
            next set of results, call this method again using the value
            of ``next_page_token`` as ``page_token``.
    """

    @property
    def raw_page(self):
        if False:
            return 10
        return self
    service_bindings: MutableSequence['ServiceBinding'] = proto.RepeatedField(proto.MESSAGE, number=1, message='ServiceBinding')
    next_page_token: str = proto.Field(proto.STRING, number=2)

class GetServiceBindingRequest(proto.Message):
    """Request used by the GetServiceBinding method.

    Attributes:
        name (str):
            Required. A name of the ServiceBinding to get. Must be in
            the format
            ``projects/*/locations/global/serviceBindings/*``.
    """
    name: str = proto.Field(proto.STRING, number=1)

class CreateServiceBindingRequest(proto.Message):
    """Request used by the ServiceBinding method.

    Attributes:
        parent (str):
            Required. The parent resource of the ServiceBinding. Must be
            in the format ``projects/*/locations/global``.
        service_binding_id (str):
            Required. Short name of the ServiceBinding
            resource to be created.
        service_binding (google.cloud.network_services_v1.types.ServiceBinding):
            Required. ServiceBinding resource to be
            created.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    service_binding_id: str = proto.Field(proto.STRING, number=2)
    service_binding: 'ServiceBinding' = proto.Field(proto.MESSAGE, number=3, message='ServiceBinding')

class DeleteServiceBindingRequest(proto.Message):
    """Request used by the DeleteServiceBinding method.

    Attributes:
        name (str):
            Required. A name of the ServiceBinding to delete. Must be in
            the format
            ``projects/*/locations/global/serviceBindings/*``.
    """
    name: str = proto.Field(proto.STRING, number=1)
__all__ = tuple(sorted(__protobuf__.manifest))