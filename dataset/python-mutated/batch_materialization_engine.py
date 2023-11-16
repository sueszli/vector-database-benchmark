import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Union
from tqdm import tqdm
from feast.batch_feature_view import BatchFeatureView
from feast.entity import Entity
from feast.feature_view import FeatureView
from feast.infra.offline_stores.offline_store import OfflineStore
from feast.infra.online_stores.online_store import OnlineStore
from feast.infra.registry.base_registry import BaseRegistry
from feast.repo_config import RepoConfig
from feast.stream_feature_view import StreamFeatureView

@dataclass
class MaterializationTask:
    """
    A MaterializationTask represents a unit of data that needs to be materialized from an
    offline store to an online store.
    """
    project: str
    feature_view: Union[BatchFeatureView, StreamFeatureView, FeatureView]
    start_time: datetime
    end_time: datetime
    tqdm_builder: Callable[[int], tqdm]

class MaterializationJobStatus(enum.Enum):
    WAITING = 1
    RUNNING = 2
    AVAILABLE = 3
    ERROR = 4
    CANCELLING = 5
    CANCELLED = 6
    SUCCEEDED = 7

class MaterializationJob(ABC):
    """
    A MaterializationJob represents an ongoing or executed process that materializes data as per the
    definition of a materialization task.
    """
    task: MaterializationTask

    @abstractmethod
    def status(self) -> MaterializationJobStatus:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def error(self) -> Optional[BaseException]:
        if False:
            i = 10
            return i + 15
        ...

    @abstractmethod
    def should_be_retried(self) -> bool:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def job_id(self) -> str:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def url(self) -> Optional[str]:
        if False:
            return 10
        ...

class BatchMaterializationEngine(ABC):
    """
    The interface that Feast uses to control the compute system that handles batch materialization.
    """

    def __init__(self, *, repo_config: RepoConfig, offline_store: OfflineStore, online_store: OnlineStore, **kwargs):
        if False:
            while True:
                i = 10
        self.repo_config = repo_config
        self.offline_store = offline_store
        self.online_store = online_store

    @abstractmethod
    def update(self, project: str, views_to_delete: Sequence[Union[BatchFeatureView, StreamFeatureView, FeatureView]], views_to_keep: Sequence[Union[BatchFeatureView, StreamFeatureView, FeatureView]], entities_to_delete: Sequence[Entity], entities_to_keep: Sequence[Entity]):
        if False:
            i = 10
            return i + 15
        '\n        Prepares cloud resources required for batch materialization for the specified set of Feast objects.\n\n        Args:\n            project: Feast project to which the objects belong.\n            views_to_delete: Feature views whose corresponding infrastructure should be deleted.\n            views_to_keep: Feature views whose corresponding infrastructure should not be deleted, and\n                may need to be updated.\n            entities_to_delete: Entities whose corresponding infrastructure should be deleted.\n            entities_to_keep: Entities whose corresponding infrastructure should not be deleted, and\n                may need to be updated.\n        '
        pass

    @abstractmethod
    def materialize(self, registry: BaseRegistry, tasks: List[MaterializationTask]) -> List[MaterializationJob]:
        if False:
            return 10
        '\n        Materialize data from the offline store to the online store for this feature repo.\n\n        Args:\n            registry: The registry for the current feature store.\n            tasks: A list of individual materialization tasks.\n\n        Returns:\n            A list of materialization jobs representing each task.\n        '
        pass

    @abstractmethod
    def teardown_infra(self, project: str, fvs: Sequence[Union[BatchFeatureView, StreamFeatureView, FeatureView]], entities: Sequence[Entity]):
        if False:
            return 10
        '\n        Tears down all cloud resources used by the materialization engine for the specified set of Feast objects.\n\n        Args:\n            project: Feast project to which the objects belong.\n            fvs: Feature views whose corresponding infrastructure should be deleted.\n            entities: Entities whose corresponding infrastructure should be deleted.\n        '
        pass