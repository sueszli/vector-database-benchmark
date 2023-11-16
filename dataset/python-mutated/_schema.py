from enum import Enum
from typing import Dict
from azure.core import CaseInsensitiveEnumMeta

class OpenTelemetrySchemaVersion(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    V1_19_0 = '1.19.0'

class OpenTelemetrySchema:
    SUPPORTED_VERSIONS = (OpenTelemetrySchemaVersion.V1_19_0,)
    _ATTRIBUTE_MAPPINGS = {OpenTelemetrySchemaVersion.V1_19_0: {'x-ms-client-request-id': 'az.client_request_id', 'x-ms-request-id': 'az.service_request_id', 'http.user_agent': 'user_agent.original', 'message_bus.destination': 'messaging.destination.name', 'peer.address': 'net.peer.name'}}

    @classmethod
    def get_latest_version(cls) -> OpenTelemetrySchemaVersion:
        if False:
            print('Hello World!')
        return OpenTelemetrySchemaVersion(cls.SUPPORTED_VERSIONS[-1])

    @classmethod
    def get_attribute_mappings(cls, version: OpenTelemetrySchemaVersion) -> Dict[str, str]:
        if False:
            return 10
        return cls._ATTRIBUTE_MAPPINGS[version]

    @classmethod
    def get_schema_url(cls, version: OpenTelemetrySchemaVersion) -> str:
        if False:
            return 10
        return f'https://opentelemetry.io/schemas/{version}'