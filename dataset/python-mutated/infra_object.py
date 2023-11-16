from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List
from feast.errors import FeastInvalidInfraObjectType
from feast.importer import import_class
from feast.protos.feast.core.DatastoreTable_pb2 import DatastoreTable as DatastoreTableProto
from feast.protos.feast.core.DynamoDBTable_pb2 import DynamoDBTable as DynamoDBTableProto
from feast.protos.feast.core.InfraObject_pb2 import Infra as InfraProto
from feast.protos.feast.core.InfraObject_pb2 import InfraObject as InfraObjectProto
from feast.protos.feast.core.SqliteTable_pb2 import SqliteTable as SqliteTableProto
DATASTORE_INFRA_OBJECT_CLASS_TYPE = 'feast.infra.online_stores.datastore.DatastoreTable'
DYNAMODB_INFRA_OBJECT_CLASS_TYPE = 'feast.infra.online_stores.dynamodb.DynamoDBTable'
SQLITE_INFRA_OBJECT_CLASS_TYPE = 'feast.infra.online_stores.sqlite.SqliteTable'

class InfraObject(ABC):
    """
    Represents a single infrastructure object (e.g. online store table) managed by Feast.
    """

    @abstractmethod
    def __init__(self, name: str):
        if False:
            return 10
        self._name = name

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._name

    @abstractmethod
    def to_infra_object_proto(self) -> InfraObjectProto:
        if False:
            print('Hello World!')
        'Converts an InfraObject to its protobuf representation, wrapped in an InfraObjectProto.'
        pass

    @abstractmethod
    def to_proto(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Converts an InfraObject to its protobuf representation.'
        pass

    def __lt__(self, other) -> bool:
        if False:
            return 10
        return self.name < other.name

    @staticmethod
    @abstractmethod
    def from_infra_object_proto(infra_object_proto: InfraObjectProto) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Returns an InfraObject created from a protobuf representation.\n\n        Args:\n            infra_object_proto: A protobuf representation of an InfraObject.\n\n        Raises:\n            FeastInvalidInfraObjectType: The type of InfraObject could not be identified.\n        '
        if infra_object_proto.infra_object_class_type:
            cls = _get_infra_object_class_from_type(infra_object_proto.infra_object_class_type)
            return cls.from_infra_object_proto(infra_object_proto)
        raise FeastInvalidInfraObjectType()

    @staticmethod
    def from_proto(infra_object_proto: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a protobuf representation of a subclass to an object of that subclass.\n\n        Args:\n            infra_object_proto: A protobuf representation of an InfraObject.\n\n        Raises:\n            FeastInvalidInfraObjectType: The type of InfraObject could not be identified.\n        '
        if isinstance(infra_object_proto, DatastoreTableProto):
            infra_object_class_type = DATASTORE_INFRA_OBJECT_CLASS_TYPE
        elif isinstance(infra_object_proto, DynamoDBTableProto):
            infra_object_class_type = DYNAMODB_INFRA_OBJECT_CLASS_TYPE
        elif isinstance(infra_object_proto, SqliteTableProto):
            infra_object_class_type = SQLITE_INFRA_OBJECT_CLASS_TYPE
        else:
            raise FeastInvalidInfraObjectType()
        cls = _get_infra_object_class_from_type(infra_object_class_type)
        return cls.from_proto(infra_object_proto)

    @abstractmethod
    def update(self):
        if False:
            print('Hello World!')
        '\n        Deploys or updates the infrastructure object.\n        '
        pass

    @abstractmethod
    def teardown(self):
        if False:
            return 10
        '\n        Tears down the infrastructure object.\n        '
        pass

@dataclass
class Infra:
    """
    Represents the set of infrastructure managed by Feast.

    Args:
        infra_objects: A list of InfraObjects, each representing one infrastructure object.
    """
    infra_objects: List[InfraObject] = field(default_factory=list)

    def to_proto(self) -> InfraProto:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts Infra to its protobuf representation.\n\n        Returns:\n            An InfraProto protobuf.\n        '
        infra_proto = InfraProto()
        for infra_object in self.infra_objects:
            infra_object_proto = infra_object.to_infra_object_proto()
            infra_proto.infra_objects.append(infra_object_proto)
        return infra_proto

    @classmethod
    def from_proto(cls, infra_proto: InfraProto):
        if False:
            i = 10
            return i + 15
        '\n        Returns an Infra object created from a protobuf representation.\n        '
        infra = cls()
        infra.infra_objects += [InfraObject.from_infra_object_proto(infra_object_proto) for infra_object_proto in infra_proto.infra_objects]
        return infra

def _get_infra_object_class_from_type(infra_object_class_type: str):
    if False:
        print('Hello World!')
    (module_name, infra_object_class_name) = infra_object_class_type.rsplit('.', 1)
    return import_class(module_name, infra_object_class_name)