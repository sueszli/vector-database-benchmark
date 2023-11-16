from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import pandas
import pyarrow
from tqdm import tqdm
from feast import Entity, FeatureService, FeatureView, RepoConfig
from feast.infra.offline_stores.offline_store import RetrievalJob
from feast.infra.provider import Provider
from feast.infra.registry.base_registry import BaseRegistry
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.saved_dataset import SavedDataset

class FooProvider(Provider):

    def __init__(self, config: RepoConfig):
        if False:
            print('Hello World!')
        pass

    def update_infra(self, project: str, tables_to_delete: Sequence[FeatureView], tables_to_keep: Sequence[FeatureView], entities_to_delete: Sequence[Entity], entities_to_keep: Sequence[Entity], partial: bool):
        if False:
            print('Hello World!')
        pass

    def teardown_infra(self, project: str, tables: Sequence[FeatureView], entities: Sequence[Entity]):
        if False:
            return 10
        pass

    def online_write_batch(self, config: RepoConfig, table: FeatureView, data: List[Tuple[EntityKeyProto, Dict[str, ValueProto], datetime, Optional[datetime]]], progress: Optional[Callable[[int], Any]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def materialize_single_feature_view(self, config: RepoConfig, feature_view: FeatureView, start_date: datetime, end_date: datetime, registry: BaseRegistry, project: str, tqdm_builder: Callable[[int], tqdm]) -> None:
        if False:
            while True:
                i = 10
        pass

    def get_historical_features(self, config: RepoConfig, feature_views: List[FeatureView], feature_refs: List[str], entity_df: Union[pandas.DataFrame, str], registry: BaseRegistry, project: str, full_feature_names: bool=False) -> RetrievalJob:
        if False:
            while True:
                i = 10
        pass

    def online_read(self, config: RepoConfig, table: FeatureView, entity_keys: List[EntityKeyProto], requested_features: List[str]=None) -> List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]]:
        if False:
            i = 10
            return i + 15
        pass

    def retrieve_saved_dataset(self, config: RepoConfig, dataset: SavedDataset):
        if False:
            print('Hello World!')
        pass

    def write_feature_service_logs(self, feature_service: FeatureService, logs: Union[pyarrow.Table, Path], config: RepoConfig, registry: BaseRegistry):
        if False:
            while True:
                i = 10
        pass

    def retrieve_feature_service_logs(self, feature_service: FeatureService, start_date: datetime, end_date: datetime, config: RepoConfig, registry: BaseRegistry) -> RetrievalJob:
        if False:
            i = 10
            return i + 15
        pass