import uuid
from typing import Optional
from google.protobuf.json_format import MessageToJson
from typeguard import typechecked
from feast.protos.feast.core.Registry_pb2 import ProjectMetadata as ProjectMetadataProto
from feast.usage import log_exceptions

@typechecked
class ProjectMetadata:
    """
    Tracks project level metadata

    Attributes:
        project_name: The registry-scoped unique name of the project.
        project_uuid: The UUID for this project
    """
    project_name: str
    project_uuid: str

    @log_exceptions
    def __init__(self, *args, project_name: Optional[str]=None, project_uuid: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        '\n        Creates an Project metadata object.\n\n        Args:\n            project_name: The registry-scoped unique name of the project.\n            project_uuid: The UUID for this project\n\n        Raises:\n            ValueError: Parameters are specified incorrectly.\n        '
        if not project_name:
            raise ValueError('Project name needs to be specified')
        self.project_name = project_name
        self.project_uuid = project_uuid or f'{uuid.uuid4()}'

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash((self.project_name, self.project_uuid))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, ProjectMetadata):
            raise TypeError('Comparisons should only involve ProjectMetadata class objects.')
        if self.project_name != other.project_name or self.project_uuid != other.project_uuid:
            return False
        return True

    def __str__(self):
        if False:
            while True:
                i = 10
        return str(MessageToJson(self.to_proto()))

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return self.project_name < other.project_name

    @classmethod
    def from_proto(cls, project_metadata_proto: ProjectMetadataProto):
        if False:
            i = 10
            return i + 15
        '\n        Creates project metadata from a protobuf representation.\n\n        Args:\n            project_metadata_proto: A protobuf representation of project metadata.\n\n        Returns:\n            A ProjectMetadata object based on the protobuf.\n        '
        entity = cls(project_name=project_metadata_proto.project, project_uuid=project_metadata_proto.project_uuid)
        return entity

    def to_proto(self) -> ProjectMetadataProto:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a project metadata object to its protobuf representation.\n\n        Returns:\n            An ProjectMetadataProto protobuf.\n        '
        return ProjectMetadataProto(project=self.project_name, project_uuid=self.project_uuid)