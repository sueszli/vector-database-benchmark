from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from feast import Entity
from feast.feature_view import FeatureView
from feast.infra.infra_object import InfraObject
from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import RepoConfig

class OnlineStore(ABC):
    """
    The interface that Feast uses to interact with the storage system that handles online features.
    """

    @abstractmethod
    def online_write_batch(self, config: RepoConfig, table: FeatureView, data: List[Tuple[EntityKeyProto, Dict[str, ValueProto], datetime, Optional[datetime]]], progress: Optional[Callable[[int], Any]]) -> None:
        if False:
            while True:
                i = 10
        '\n        Writes a batch of feature rows to the online store.\n\n        If a tz-naive timestamp is passed to this method, it is assumed to be UTC.\n\n        Args:\n            config: The config for the current feature store.\n            table: Feature view to which these feature rows correspond.\n            data: A list of quadruplets containing feature data. Each quadruplet contains an entity\n                key, a dict containing feature values, an event timestamp for the row, and the created\n                timestamp for the row if it exists.\n            progress: Function to be called once a batch of rows is written to the online store, used\n                to show progress.\n        '
        pass

    @abstractmethod
    def online_read(self, config: RepoConfig, table: FeatureView, entity_keys: List[EntityKeyProto], requested_features: Optional[List[str]]=None) -> List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]]:
        if False:
            while True:
                i = 10
        '\n        Reads features values for the given entity keys.\n\n        Args:\n            config: The config for the current feature store.\n            table: The feature view whose feature values should be read.\n            entity_keys: The list of entity keys for which feature values should be read.\n            requested_features: The list of features that should be read.\n\n        Returns:\n            A list of the same length as entity_keys. Each item in the list is a tuple where the first\n            item is the event timestamp for the row, and the second item is a dict mapping feature names\n            to values, which are returned in proto format.\n        '
        pass

    @abstractmethod
    def update(self, config: RepoConfig, tables_to_delete: Sequence[FeatureView], tables_to_keep: Sequence[FeatureView], entities_to_delete: Sequence[Entity], entities_to_keep: Sequence[Entity], partial: bool):
        if False:
            while True:
                i = 10
        '\n        Reconciles cloud resources with the specified set of Feast objects.\n\n        Args:\n            config: The config for the current feature store.\n            tables_to_delete: Feature views whose corresponding infrastructure should be deleted.\n            tables_to_keep: Feature views whose corresponding infrastructure should not be deleted, and\n                may need to be updated.\n            entities_to_delete: Entities whose corresponding infrastructure should be deleted.\n            entities_to_keep: Entities whose corresponding infrastructure should not be deleted, and\n                may need to be updated.\n            partial: If true, tables_to_delete and tables_to_keep are not exhaustive lists, so\n                infrastructure corresponding to other feature views should be not be touched.\n        '
        pass

    def plan(self, config: RepoConfig, desired_registry_proto: RegistryProto) -> List[InfraObject]:
        if False:
            while True:
                i = 10
        '\n        Returns the set of InfraObjects required to support the desired registry.\n\n        Args:\n            config: The config for the current feature store.\n            desired_registry_proto: The desired registry, in proto form.\n        '
        return []

    @abstractmethod
    def teardown(self, config: RepoConfig, tables: Sequence[FeatureView], entities: Sequence[Entity]):
        if False:
            i = 10
            return i + 15
        '\n        Tears down all cloud resources for the specified set of Feast objects.\n\n        Args:\n            config: The config for the current feature store.\n            tables: Feature views whose corresponding infrastructure should be deleted.\n            entities: Entities whose corresponding infrastructure should be deleted.\n        '
        pass