import copy
import itertools
import os
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import pandas as pd
import pyarrow as pa
from colorama import Fore, Style
from google.protobuf.timestamp_pb2 import Timestamp
from tqdm import tqdm
from feast import feature_server, flags_helper, ui_server, utils
from feast.base_feature_view import BaseFeatureView
from feast.batch_feature_view import BatchFeatureView
from feast.data_source import DataSource, KafkaSource, KinesisSource, PushMode, PushSource
from feast.diff.infra_diff import InfraDiff, diff_infra_protos
from feast.diff.registry_diff import RegistryDiff, apply_diff_to_registry, diff_between
from feast.dqm.errors import ValidationFailed
from feast.entity import Entity
from feast.errors import DataSourceRepeatNamesException, EntityNotFoundException, FeatureNameCollisionError, FeatureViewNotFoundException, PushSourceNotFoundException, RequestDataNotFoundInEntityDfException, RequestDataNotFoundInEntityRowsException
from feast.feast_object import FeastObject
from feast.feature_service import FeatureService
from feast.feature_view import DUMMY_ENTITY, DUMMY_ENTITY_ID, DUMMY_ENTITY_NAME, DUMMY_ENTITY_VAL, FeatureView
from feast.inference import update_data_sources_with_inferred_event_timestamp_col, update_feature_views_with_inferred_features_and_entities
from feast.infra.infra_object import Infra
from feast.infra.provider import Provider, RetrievalJob, get_provider
from feast.infra.registry.base_registry import BaseRegistry
from feast.infra.registry.registry import Registry
from feast.infra.registry.sql import SqlRegistry
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.online_response import OnlineResponse
from feast.protos.feast.serving.ServingService_pb2 import FieldStatus, GetOnlineFeaturesResponse
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import RepeatedValue, Value
from feast.repo_config import RepoConfig, load_repo_config
from feast.repo_contents import RepoContents
from feast.request_feature_view import RequestFeatureView
from feast.saved_dataset import SavedDataset, SavedDatasetStorage, ValidationReference
from feast.stream_feature_view import StreamFeatureView
from feast.type_map import python_values_to_proto_values
from feast.usage import log_exceptions, log_exceptions_and_usage, set_usage_attribute
from feast.value_type import ValueType
from feast.version import get_version
warnings.simplefilter('once', DeprecationWarning)

class FeatureStore:
    """
    A FeatureStore object is used to define, create, and retrieve features.

    Attributes:
        config: The config for the feature store.
        repo_path: The path to the feature repo.
        _registry: The registry for the feature store.
        _provider: The provider for the feature store.
    """
    config: RepoConfig
    repo_path: Path
    _registry: BaseRegistry
    _provider: Provider

    @log_exceptions
    def __init__(self, repo_path: Optional[str]=None, config: Optional[RepoConfig]=None, fs_yaml_file: Optional[Path]=None):
        if False:
            i = 10
            return i + 15
        "\n        Creates a FeatureStore object.\n\n        Args:\n            repo_path (optional): Path to the feature repo. Defaults to the current working directory.\n            config (optional): Configuration object used to configure the feature store.\n            fs_yaml_file (optional): Path to the `feature_store.yaml` file used to configure the feature store.\n                At most one of 'fs_yaml_file' and 'config' can be set.\n\n        Raises:\n            ValueError: If both or neither of repo_path and config are specified.\n        "
        if fs_yaml_file is not None and config is not None:
            raise ValueError('You cannot specify both fs_yaml_file and config.')
        if repo_path:
            self.repo_path = Path(repo_path)
        else:
            self.repo_path = Path(os.getcwd())
        if config is not None:
            self.config = config
        elif fs_yaml_file is not None:
            self.config = load_repo_config(self.repo_path, fs_yaml_file)
        else:
            self.config = load_repo_config(self.repo_path, utils.get_default_yaml_file_path(self.repo_path))
        registry_config = self.config.registry
        if registry_config.registry_type == 'sql':
            self._registry = SqlRegistry(registry_config, self.config.project, None)
        elif registry_config.registry_type == 'snowflake.registry':
            from feast.infra.registry.snowflake import SnowflakeRegistry
            self._registry = SnowflakeRegistry(registry_config, self.config.project, None)
        else:
            r = Registry(self.config.project, registry_config, repo_path=self.repo_path)
            r._initialize_registry(self.config.project)
            self._registry = r
        self._provider = get_provider(self.config)

    @log_exceptions
    def version(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the version of the current Feast SDK/CLI.'
        return get_version()

    @property
    def registry(self) -> BaseRegistry:
        if False:
            while True:
                i = 10
        'Gets the registry of this feature store.'
        return self._registry

    @property
    def project(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Gets the project of this feature store.'
        return self.config.project

    def _get_provider(self) -> Provider:
        if False:
            for i in range(10):
                print('nop')
        return self._provider

    @log_exceptions_and_usage
    def refresh_registry(self):
        if False:
            for i in range(10):
                print('nop')
        'Fetches and caches a copy of the feature registry in memory.\n\n        Explicitly calling this method allows for direct control of the state of the registry cache. Every time this\n        method is called the complete registry state will be retrieved from the remote registry store backend\n        (e.g., GCS, S3), and the cache timer will be reset. If refresh_registry() is run before get_online_features()\n        is called, then get_online_features() will use the cached registry instead of retrieving (and caching) the\n        registry itself.\n\n        Additionally, the TTL for the registry cache can be set to infinity (by setting it to 0), which means that\n        refresh_registry() will become the only way to update the cached registry. If the TTL is set to a value\n        greater than 0, then once the cache becomes stale (more time than the TTL has passed), a new cache will be\n        downloaded synchronously, which may increase latencies if the triggering method is get_online_features().\n        '
        registry_config = self.config.registry
        registry = Registry(self.config.project, registry_config, repo_path=self.repo_path)
        registry.refresh(self.config.project)
        self._registry = registry

    @log_exceptions_and_usage
    def list_entities(self, allow_cache: bool=False) -> List[Entity]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the list of entities from the registry.\n\n        Args:\n            allow_cache: Whether to allow returning entities from a cached registry.\n\n        Returns:\n            A list of entities.\n        '
        return self._list_entities(allow_cache)

    def _list_entities(self, allow_cache: bool=False, hide_dummy_entity: bool=True) -> List[Entity]:
        if False:
            print('Hello World!')
        all_entities = self._registry.list_entities(self.project, allow_cache=allow_cache)
        return [entity for entity in all_entities if entity.name != DUMMY_ENTITY_NAME or not hide_dummy_entity]

    @log_exceptions_and_usage
    def list_feature_services(self) -> List[FeatureService]:
        if False:
            while True:
                i = 10
        '\n        Retrieves the list of feature services from the registry.\n\n        Returns:\n            A list of feature services.\n        '
        return self._registry.list_feature_services(self.project)

    @log_exceptions_and_usage
    def list_feature_views(self, allow_cache: bool=False) -> List[FeatureView]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves the list of feature views from the registry.\n\n        Args:\n            allow_cache: Whether to allow returning entities from a cached registry.\n\n        Returns:\n            A list of feature views.\n        '
        return self._list_feature_views(allow_cache)

    @log_exceptions_and_usage
    def list_request_feature_views(self, allow_cache: bool=False) -> List[RequestFeatureView]:
        if False:
            while True:
                i = 10
        '\n        Retrieves the list of feature views from the registry.\n\n        Args:\n            allow_cache: Whether to allow returning entities from a cached registry.\n\n        Returns:\n            A list of feature views.\n        '
        return self._registry.list_request_feature_views(self.project, allow_cache=allow_cache)

    def _list_feature_views(self, allow_cache: bool=False, hide_dummy_entity: bool=True) -> List[FeatureView]:
        if False:
            i = 10
            return i + 15
        feature_views = []
        for fv in self._registry.list_feature_views(self.project, allow_cache=allow_cache):
            if hide_dummy_entity and fv.entities and (fv.entities[0] == DUMMY_ENTITY_NAME):
                fv.entities = []
                fv.entity_columns = []
            feature_views.append(fv)
        return feature_views

    def _list_stream_feature_views(self, allow_cache: bool=False, hide_dummy_entity: bool=True) -> List[StreamFeatureView]:
        if False:
            i = 10
            return i + 15
        stream_feature_views = []
        for sfv in self._registry.list_stream_feature_views(self.project, allow_cache=allow_cache):
            if hide_dummy_entity and sfv.entities[0] == DUMMY_ENTITY_NAME:
                sfv.entities = []
                sfv.entity_columns = []
            stream_feature_views.append(sfv)
        return stream_feature_views

    @log_exceptions_and_usage
    def list_on_demand_feature_views(self, allow_cache: bool=False) -> List[OnDemandFeatureView]:
        if False:
            while True:
                i = 10
        '\n        Retrieves the list of on demand feature views from the registry.\n\n        Returns:\n            A list of on demand feature views.\n        '
        return self._registry.list_on_demand_feature_views(self.project, allow_cache=allow_cache)

    @log_exceptions_and_usage
    def list_stream_feature_views(self, allow_cache: bool=False) -> List[StreamFeatureView]:
        if False:
            return 10
        '\n        Retrieves the list of stream feature views from the registry.\n\n        Returns:\n            A list of stream feature views.\n        '
        return self._list_stream_feature_views(allow_cache)

    @log_exceptions_and_usage
    def list_data_sources(self, allow_cache: bool=False) -> List[DataSource]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the list of data sources from the registry.\n\n        Args:\n            allow_cache: Whether to allow returning data sources from a cached registry.\n\n        Returns:\n            A list of data sources.\n        '
        return self._registry.list_data_sources(self.project, allow_cache=allow_cache)

    @log_exceptions_and_usage
    def get_entity(self, name: str, allow_registry_cache: bool=False) -> Entity:
        if False:
            i = 10
            return i + 15
        '\n        Retrieves an entity.\n\n        Args:\n            name: Name of entity.\n            allow_registry_cache: (Optional) Whether to allow returning this entity from a cached registry\n\n        Returns:\n            The specified entity.\n\n        Raises:\n            EntityNotFoundException: The entity could not be found.\n        '
        return self._registry.get_entity(name, self.project, allow_cache=allow_registry_cache)

    @log_exceptions_and_usage
    def get_feature_service(self, name: str, allow_cache: bool=False) -> FeatureService:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves a feature service.\n\n        Args:\n            name: Name of feature service.\n            allow_cache: Whether to allow returning feature services from a cached registry.\n\n        Returns:\n            The specified feature service.\n\n        Raises:\n            FeatureServiceNotFoundException: The feature service could not be found.\n        '
        return self._registry.get_feature_service(name, self.project, allow_cache)

    @log_exceptions_and_usage
    def get_feature_view(self, name: str, allow_registry_cache: bool=False) -> FeatureView:
        if False:
            while True:
                i = 10
        '\n        Retrieves a feature view.\n\n        Args:\n            name: Name of feature view.\n            allow_registry_cache: (Optional) Whether to allow returning this entity from a cached registry\n\n        Returns:\n            The specified feature view.\n\n        Raises:\n            FeatureViewNotFoundException: The feature view could not be found.\n        '
        return self._get_feature_view(name, allow_registry_cache=allow_registry_cache)

    def _get_feature_view(self, name: str, hide_dummy_entity: bool=True, allow_registry_cache: bool=False) -> FeatureView:
        if False:
            while True:
                i = 10
        feature_view = self._registry.get_feature_view(name, self.project, allow_cache=allow_registry_cache)
        if hide_dummy_entity and feature_view.entities[0] == DUMMY_ENTITY_NAME:
            feature_view.entities = []
        return feature_view

    @log_exceptions_and_usage
    def get_stream_feature_view(self, name: str, allow_registry_cache: bool=False) -> StreamFeatureView:
        if False:
            i = 10
            return i + 15
        '\n        Retrieves a stream feature view.\n\n        Args:\n            name: Name of stream feature view.\n            allow_registry_cache: (Optional) Whether to allow returning this entity from a cached registry\n\n        Returns:\n            The specified stream feature view.\n\n        Raises:\n            FeatureViewNotFoundException: The feature view could not be found.\n        '
        return self._get_stream_feature_view(name, allow_registry_cache=allow_registry_cache)

    def _get_stream_feature_view(self, name: str, hide_dummy_entity: bool=True, allow_registry_cache: bool=False) -> StreamFeatureView:
        if False:
            i = 10
            return i + 15
        stream_feature_view = self._registry.get_stream_feature_view(name, self.project, allow_cache=allow_registry_cache)
        if hide_dummy_entity and stream_feature_view.entities[0] == DUMMY_ENTITY_NAME:
            stream_feature_view.entities = []
        return stream_feature_view

    @log_exceptions_and_usage
    def get_on_demand_feature_view(self, name: str) -> OnDemandFeatureView:
        if False:
            return 10
        '\n        Retrieves a feature view.\n\n        Args:\n            name: Name of feature view.\n\n        Returns:\n            The specified feature view.\n\n        Raises:\n            FeatureViewNotFoundException: The feature view could not be found.\n        '
        return self._registry.get_on_demand_feature_view(name, self.project)

    @log_exceptions_and_usage
    def get_data_source(self, name: str) -> DataSource:
        if False:
            print('Hello World!')
        '\n        Retrieves the list of data sources from the registry.\n\n        Args:\n            name: Name of the data source.\n\n        Returns:\n            The specified data source.\n\n        Raises:\n            DataSourceObjectNotFoundException: The data source could not be found.\n        '
        return self._registry.get_data_source(name, self.project)

    @log_exceptions_and_usage
    def delete_feature_view(self, name: str):
        if False:
            return 10
        '\n        Deletes a feature view.\n\n        Args:\n            name: Name of feature view.\n\n        Raises:\n            FeatureViewNotFoundException: The feature view could not be found.\n        '
        return self._registry.delete_feature_view(name, self.project)

    @log_exceptions_and_usage
    def delete_feature_service(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes a feature service.\n\n        Args:\n            name: Name of feature service.\n\n        Raises:\n            FeatureServiceNotFoundException: The feature view could not be found.\n        '
        return self._registry.delete_feature_service(name, self.project)

    def _get_features(self, features: Union[List[str], FeatureService], allow_cache: bool=False) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        _features = features
        if not _features:
            raise ValueError('No features specified for retrieval')
        _feature_refs = []
        if isinstance(_features, FeatureService):
            feature_service_from_registry = self.get_feature_service(_features.name, allow_cache)
            if feature_service_from_registry != _features:
                warnings.warn('The FeatureService object that has been passed in as an argument is inconsistent with the version from the registry. Potentially a newer version of the FeatureService has been applied to the registry.')
            for projection in feature_service_from_registry.feature_view_projections:
                _feature_refs.extend([f'{projection.name_to_use()}:{f.name}' for f in projection.features])
        else:
            assert isinstance(_features, list)
            _feature_refs = _features
        return _feature_refs

    def _should_use_plan(self):
        if False:
            i = 10
            return i + 15
        'Returns True if plan and _apply_diffs should be used, False otherwise.'
        return self.config.provider == 'local' and (self.config.online_store and self.config.online_store.type == 'sqlite')

    def _validate_all_feature_views(self, views_to_update: List[FeatureView], odfvs_to_update: List[OnDemandFeatureView], request_views_to_update: List[RequestFeatureView], sfvs_to_update: List[StreamFeatureView]):
        if False:
            return 10
        'Validates all feature views.'
        if len(odfvs_to_update) > 0 and (not flags_helper.is_test()):
            warnings.warn('On demand feature view is an experimental feature. This API is stable, but the functionality does not scale well for offline retrieval', RuntimeWarning)
        set_usage_attribute('odfv', bool(odfvs_to_update))
        _validate_feature_views([*views_to_update, *odfvs_to_update, *request_views_to_update, *sfvs_to_update])

    def _make_inferences(self, data_sources_to_update: List[DataSource], entities_to_update: List[Entity], views_to_update: List[FeatureView], odfvs_to_update: List[OnDemandFeatureView], sfvs_to_update: List[StreamFeatureView], feature_services_to_update: List[FeatureService]):
        if False:
            for i in range(10):
                print('nop')
        'Makes inferences for entities, feature views, odfvs, and feature services.'
        update_data_sources_with_inferred_event_timestamp_col(data_sources_to_update, self.config)
        update_data_sources_with_inferred_event_timestamp_col([view.batch_source for view in views_to_update], self.config)
        update_data_sources_with_inferred_event_timestamp_col([view.batch_source for view in sfvs_to_update], self.config)
        entities = self._list_entities()
        update_feature_views_with_inferred_features_and_entities(views_to_update, entities + entities_to_update, self.config)
        update_feature_views_with_inferred_features_and_entities(sfvs_to_update, entities + entities_to_update, self.config)
        for sfv in sfvs_to_update:
            if not sfv.schema:
                raise ValueError(f'schema inference not yet supported for stream feature views. please define schema for stream feature view: {sfv.name}')
        for odfv in odfvs_to_update:
            odfv.infer_features()
        fvs_to_update_map = {view.name: view for view in [*views_to_update, *sfvs_to_update]}
        for feature_service in feature_services_to_update:
            feature_service.infer_features(fvs_to_update=fvs_to_update_map)

    def _get_feature_views_to_materialize(self, feature_views: Optional[List[str]]) -> List[FeatureView]:
        if False:
            while True:
                i = 10
        '\n        Returns the list of feature views that should be materialized.\n\n        If no feature views are specified, all feature views will be returned.\n\n        Args:\n            feature_views: List of names of feature views to materialize.\n\n        Raises:\n            FeatureViewNotFoundException: One of the specified feature views could not be found.\n            ValueError: One of the specified feature views is not configured for materialization.\n        '
        feature_views_to_materialize: List[FeatureView] = []
        if feature_views is None:
            feature_views_to_materialize = self._list_feature_views(hide_dummy_entity=False)
            feature_views_to_materialize = [fv for fv in feature_views_to_materialize if fv.online]
            stream_feature_views_to_materialize = self._list_stream_feature_views(hide_dummy_entity=False)
            feature_views_to_materialize += [sfv for sfv in stream_feature_views_to_materialize if sfv.online]
        else:
            for name in feature_views:
                try:
                    feature_view = self._get_feature_view(name, hide_dummy_entity=False)
                except FeatureViewNotFoundException:
                    feature_view = self._get_stream_feature_view(name, hide_dummy_entity=False)
                if not feature_view.online:
                    raise ValueError(f'FeatureView {feature_view.name} is not configured to be served online.')
                feature_views_to_materialize.append(feature_view)
        return feature_views_to_materialize

    @log_exceptions_and_usage
    def plan(self, desired_repo_contents: RepoContents) -> Tuple[RegistryDiff, InfraDiff, Infra]:
        if False:
            while True:
                i = 10
        'Dry-run registering objects to metadata store.\n\n        The plan method dry-runs registering one or more definitions (e.g., Entity, FeatureView), and produces\n        a list of all the changes the that would be introduced in the feature repo. The changes computed by the plan\n        command are for informational purposes, and are not actually applied to the registry.\n\n        Args:\n            desired_repo_contents: The desired repo state.\n\n        Raises:\n            ValueError: The \'objects\' parameter could not be parsed properly.\n\n        Examples:\n            Generate a plan adding an Entity and a FeatureView.\n\n            >>> from feast import FeatureStore, Entity, FeatureView, Feature, FileSource, RepoConfig\n            >>> from feast.feature_store import RepoContents\n            >>> from datetime import timedelta\n            >>> fs = FeatureStore(repo_path="project/feature_repo")\n            >>> driver = Entity(name="driver_id", description="driver id")\n            >>> driver_hourly_stats = FileSource(\n            ...     path="project/feature_repo/data/driver_stats.parquet",\n            ...     timestamp_field="event_timestamp",\n            ...     created_timestamp_column="created",\n            ... )\n            >>> driver_hourly_stats_view = FeatureView(\n            ...     name="driver_hourly_stats",\n            ...     entities=[driver],\n            ...     ttl=timedelta(seconds=86400 * 1),\n            ...     source=driver_hourly_stats,\n            ... )\n            >>> registry_diff, infra_diff, new_infra = fs.plan(RepoContents(\n            ...     data_sources=[driver_hourly_stats],\n            ...     feature_views=[driver_hourly_stats_view],\n            ...     on_demand_feature_views=list(),\n            ...     stream_feature_views=list(),\n            ...     request_feature_views=list(),\n            ...     entities=[driver],\n            ...     feature_services=list())) # register entity and feature view\n        '
        self._validate_all_feature_views(desired_repo_contents.feature_views, desired_repo_contents.on_demand_feature_views, desired_repo_contents.request_feature_views, desired_repo_contents.stream_feature_views)
        _validate_data_sources(desired_repo_contents.data_sources)
        self._make_inferences(desired_repo_contents.data_sources, desired_repo_contents.entities, desired_repo_contents.feature_views, desired_repo_contents.on_demand_feature_views, desired_repo_contents.stream_feature_views, desired_repo_contents.feature_services)
        registry_diff = diff_between(self._registry, self.project, desired_repo_contents)
        self._registry.refresh(project=self.project)
        current_infra_proto = self._registry.proto().infra.__deepcopy__()
        desired_registry_proto = desired_repo_contents.to_registry_proto()
        new_infra = self._provider.plan_infra(self.config, desired_registry_proto)
        new_infra_proto = new_infra.to_proto()
        infra_diff = diff_infra_protos(current_infra_proto, new_infra_proto)
        return (registry_diff, infra_diff, new_infra)

    @log_exceptions_and_usage
    def _apply_diffs(self, registry_diff: RegistryDiff, infra_diff: InfraDiff, new_infra: Infra):
        if False:
            for i in range(10):
                print('nop')
        'Applies the given diffs to the metadata store and infrastructure.\n\n        Args:\n            registry_diff: The diff between the current registry and the desired registry.\n            infra_diff: The diff between the current infra and the desired infra.\n            new_infra: The desired infra.\n        '
        infra_diff.update()
        apply_diff_to_registry(self._registry, registry_diff, self.project, commit=False)
        self._registry.update_infra(new_infra, self.project, commit=True)

    @log_exceptions_and_usage
    def apply(self, objects: Union[DataSource, Entity, FeatureView, OnDemandFeatureView, RequestFeatureView, BatchFeatureView, StreamFeatureView, FeatureService, ValidationReference, List[FeastObject]], objects_to_delete: Optional[List[FeastObject]]=None, partial: bool=True):
        if False:
            return 10
        'Register objects to metadata store and update related infrastructure.\n\n        The apply method registers one or more definitions (e.g., Entity, FeatureView) and registers or updates these\n        objects in the Feast registry. Once the apply method has updated the infrastructure (e.g., create tables in\n        an online store), it will commit the updated registry. All operations are idempotent, meaning they can safely\n        be rerun.\n\n        Args:\n            objects: A single object, or a list of objects that should be registered with the Feature Store.\n            objects_to_delete: A list of objects to be deleted from the registry and removed from the\n                provider\'s infrastructure. This deletion will only be performed if partial is set to False.\n            partial: If True, apply will only handle the specified objects; if False, apply will also delete\n                all the objects in objects_to_delete, and tear down any associated cloud resources.\n\n        Raises:\n            ValueError: The \'objects\' parameter could not be parsed properly.\n\n        Examples:\n            Register an Entity and a FeatureView.\n\n            >>> from feast import FeatureStore, Entity, FeatureView, Feature, FileSource, RepoConfig\n            >>> from datetime import timedelta\n            >>> fs = FeatureStore(repo_path="project/feature_repo")\n            >>> driver = Entity(name="driver_id", description="driver id")\n            >>> driver_hourly_stats = FileSource(\n            ...     path="project/feature_repo/data/driver_stats.parquet",\n            ...     timestamp_field="event_timestamp",\n            ...     created_timestamp_column="created",\n            ... )\n            >>> driver_hourly_stats_view = FeatureView(\n            ...     name="driver_hourly_stats",\n            ...     entities=[driver],\n            ...     ttl=timedelta(seconds=86400 * 1),\n            ...     source=driver_hourly_stats,\n            ... )\n            >>> fs.apply([driver_hourly_stats_view, driver]) # register entity and feature view\n        '
        if not isinstance(objects, Iterable):
            objects = [objects]
        assert isinstance(objects, list)
        if not objects_to_delete:
            objects_to_delete = []
        entities_to_update = [ob for ob in objects if isinstance(ob, Entity)]
        views_to_update = [ob for ob in objects if (isinstance(ob, FeatureView) or isinstance(ob, BatchFeatureView)) and (not isinstance(ob, StreamFeatureView))]
        sfvs_to_update = [ob for ob in objects if isinstance(ob, StreamFeatureView)]
        request_views_to_update = [ob for ob in objects if isinstance(ob, RequestFeatureView)]
        odfvs_to_update = [ob for ob in objects if isinstance(ob, OnDemandFeatureView)]
        services_to_update = [ob for ob in objects if isinstance(ob, FeatureService)]
        data_sources_set_to_update = {ob for ob in objects if isinstance(ob, DataSource)}
        validation_references_to_update = [ob for ob in objects if isinstance(ob, ValidationReference)]
        batch_sources_to_add: List[DataSource] = []
        for data_source in data_sources_set_to_update:
            if isinstance(data_source, PushSource) or isinstance(data_source, KafkaSource) or isinstance(data_source, KinesisSource):
                assert data_source.batch_source
                batch_sources_to_add.append(data_source.batch_source)
        for batch_source in batch_sources_to_add:
            data_sources_set_to_update.add(batch_source)
        for fv in itertools.chain(views_to_update, sfvs_to_update):
            data_sources_set_to_update.add(fv.batch_source)
            if fv.stream_source:
                data_sources_set_to_update.add(fv.stream_source)
        if request_views_to_update:
            warnings.warn('Request feature view is deprecated. Please use request data source instead', DeprecationWarning)
        for rfv in request_views_to_update:
            data_sources_set_to_update.add(rfv.request_source)
        for odfv in odfvs_to_update:
            for v in odfv.source_request_sources.values():
                data_sources_set_to_update.add(v)
        data_sources_to_update = list(data_sources_set_to_update)
        entities_to_update.append(DUMMY_ENTITY)
        self._validate_all_feature_views(views_to_update, odfvs_to_update, request_views_to_update, sfvs_to_update)
        self._make_inferences(data_sources_to_update, entities_to_update, views_to_update, odfvs_to_update, sfvs_to_update, services_to_update)
        for ds in data_sources_to_update:
            self._registry.apply_data_source(ds, project=self.project, commit=False)
        for view in itertools.chain(views_to_update, odfvs_to_update, request_views_to_update, sfvs_to_update):
            self._registry.apply_feature_view(view, project=self.project, commit=False)
        for ent in entities_to_update:
            self._registry.apply_entity(ent, project=self.project, commit=False)
        for feature_service in services_to_update:
            self._registry.apply_feature_service(feature_service, project=self.project, commit=False)
        for validation_references in validation_references_to_update:
            self._registry.apply_validation_reference(validation_references, project=self.project, commit=False)
        entities_to_delete = []
        views_to_delete = []
        sfvs_to_delete = []
        if not partial:
            entities_to_delete = [ob for ob in objects_to_delete if isinstance(ob, Entity)]
            views_to_delete = [ob for ob in objects_to_delete if (isinstance(ob, FeatureView) or isinstance(ob, BatchFeatureView)) and (not isinstance(ob, StreamFeatureView))]
            request_views_to_delete = [ob for ob in objects_to_delete if isinstance(ob, RequestFeatureView)]
            odfvs_to_delete = [ob for ob in objects_to_delete if isinstance(ob, OnDemandFeatureView)]
            sfvs_to_delete = [ob for ob in objects_to_delete if isinstance(ob, StreamFeatureView)]
            services_to_delete = [ob for ob in objects_to_delete if isinstance(ob, FeatureService)]
            data_sources_to_delete = [ob for ob in objects_to_delete if isinstance(ob, DataSource)]
            validation_references_to_delete = [ob for ob in objects_to_delete if isinstance(ob, ValidationReference)]
            for data_source in data_sources_to_delete:
                self._registry.delete_data_source(data_source.name, project=self.project, commit=False)
            for entity in entities_to_delete:
                self._registry.delete_entity(entity.name, project=self.project, commit=False)
            for view in views_to_delete:
                self._registry.delete_feature_view(view.name, project=self.project, commit=False)
            for request_view in request_views_to_delete:
                self._registry.delete_feature_view(request_view.name, project=self.project, commit=False)
            for odfv in odfvs_to_delete:
                self._registry.delete_feature_view(odfv.name, project=self.project, commit=False)
            for sfv in sfvs_to_delete:
                self._registry.delete_feature_view(sfv.name, project=self.project, commit=False)
            for service in services_to_delete:
                self._registry.delete_feature_service(service.name, project=self.project, commit=False)
            for validation_references in validation_references_to_delete:
                self._registry.delete_validation_reference(validation_references.name, project=self.project, commit=False)
        tables_to_delete: List[FeatureView] = views_to_delete + sfvs_to_delete if not partial else []
        tables_to_keep: List[FeatureView] = views_to_update + sfvs_to_update
        self._get_provider().update_infra(project=self.project, tables_to_delete=tables_to_delete, tables_to_keep=tables_to_keep, entities_to_delete=entities_to_delete if not partial else [], entities_to_keep=entities_to_update, partial=partial)
        self._registry.commit()

    @log_exceptions_and_usage
    def teardown(self):
        if False:
            print('Hello World!')
        'Tears down all local and cloud resources for the feature store.'
        tables: List[FeatureView] = []
        feature_views = self.list_feature_views()
        tables.extend(feature_views)
        entities = self.list_entities()
        self._get_provider().teardown_infra(self.project, tables, entities)
        self._registry.teardown()

    @log_exceptions_and_usage
    def get_historical_features(self, entity_df: Union[pd.DataFrame, str], features: Union[List[str], FeatureService], full_feature_names: bool=False) -> RetrievalJob:
        if False:
            while True:
                i = 10
        'Enrich an entity dataframe with historical feature values for either training or batch scoring.\n\n        This method joins historical feature data from one or more feature views to an entity dataframe by using a time\n        travel join.\n\n        Each feature view is joined to the entity dataframe using all entities configured for the respective feature\n        view. All configured entities must be available in the entity dataframe. Therefore, the entity dataframe must\n        contain all entities found in all feature views, but the individual feature views can have different entities.\n\n        Time travel is based on the configured TTL for each feature view. A shorter TTL will limit the\n        amount of scanning that will be done in order to find feature data for a specific entity key. Setting a short\n        TTL may result in null values being returned.\n\n        Args:\n            entity_df (Union[pd.DataFrame, str]): An entity dataframe is a collection of rows containing all entity\n                columns (e.g., customer_id, driver_id) on which features need to be joined, as well as a event_timestamp\n                column used to ensure point-in-time correctness. Either a Pandas DataFrame can be provided or a string\n                SQL query. The query must be of a format supported by the configured offline store (e.g., BigQuery)\n            features: The list of features that should be retrieved from the offline store. These features can be\n                specified either as a list of string feature references or as a feature service. String feature\n                references must have format "feature_view:feature", e.g. "customer_fv:daily_transactions".\n            full_feature_names: If True, feature names will be prefixed with the corresponding feature view name,\n                changing them from the format "feature" to "feature_view__feature" (e.g. "daily_transactions"\n                changes to "customer_fv__daily_transactions").\n\n        Returns:\n            RetrievalJob which can be used to materialize the results.\n\n        Raises:\n            ValueError: Both or neither of features and feature_refs are specified.\n\n        Examples:\n            Retrieve historical features from a local offline store.\n\n            >>> from feast import FeatureStore, RepoConfig\n            >>> import pandas as pd\n            >>> fs = FeatureStore(repo_path="project/feature_repo")\n            >>> entity_df = pd.DataFrame.from_dict(\n            ...     {\n            ...         "driver_id": [1001, 1002],\n            ...         "event_timestamp": [\n            ...             datetime(2021, 4, 12, 10, 59, 42),\n            ...             datetime(2021, 4, 12, 8, 12, 10),\n            ...         ],\n            ...     }\n            ... )\n            >>> retrieval_job = fs.get_historical_features(\n            ...     entity_df=entity_df,\n            ...     features=[\n            ...         "driver_hourly_stats:conv_rate",\n            ...         "driver_hourly_stats:acc_rate",\n            ...         "driver_hourly_stats:avg_daily_trips",\n            ...     ],\n            ... )\n            >>> feature_data = retrieval_job.to_df()\n        '
        _feature_refs = self._get_features(features)
        (all_feature_views, all_request_feature_views, all_on_demand_feature_views) = self._get_feature_views_to_use(features)
        if all_request_feature_views:
            warnings.warn('Request feature view is deprecated. Please use request data source instead', DeprecationWarning)
        (fvs, odfvs, request_fvs, request_fv_refs) = _group_feature_refs(_feature_refs, all_feature_views, all_request_feature_views, all_on_demand_feature_views)
        feature_views = list((view for (view, _) in fvs))
        on_demand_feature_views = list((view for (view, _) in odfvs))
        request_feature_views = list((view for (view, _) in request_fvs))
        set_usage_attribute('odfv', bool(on_demand_feature_views))
        set_usage_attribute('request_fv', bool(request_feature_views))
        if type(entity_df) == pd.DataFrame:
            if self.config.coerce_tz_aware:
                entity_df = utils.make_df_tzaware(cast(pd.DataFrame, entity_df))
            for fv in request_feature_views:
                for feature in fv.features:
                    if feature.name not in entity_df.columns:
                        raise RequestDataNotFoundInEntityDfException(feature_name=feature.name, feature_view_name=fv.name)
            for odfv in on_demand_feature_views:
                odfv_request_data_schema = odfv.get_request_data_schema()
                for feature_name in odfv_request_data_schema.keys():
                    if feature_name not in entity_df.columns:
                        raise RequestDataNotFoundInEntityDfException(feature_name=feature_name, feature_view_name=odfv.name)
        _validate_feature_refs(_feature_refs, full_feature_names)
        _feature_refs = [ref for ref in _feature_refs if ref not in request_fv_refs]
        provider = self._get_provider()
        job = provider.get_historical_features(self.config, feature_views, _feature_refs, entity_df, self._registry, self.project, full_feature_names)
        return job

    @log_exceptions_and_usage
    def create_saved_dataset(self, from_: RetrievalJob, name: str, storage: SavedDatasetStorage, tags: Optional[Dict[str, str]]=None, feature_service: Optional[FeatureService]=None, allow_overwrite: bool=False) -> SavedDataset:
        if False:
            for i in range(10):
                print('nop')
        "\n        Execute provided retrieval job and persist its outcome in given storage.\n        Storage type (eg, BigQuery or Redshift) must be the same as globally configured offline store.\n        After data successfully persisted saved dataset object with dataset metadata is committed to the registry.\n        Name for the saved dataset should be unique within project, since it's possible to overwrite previously stored dataset\n        with the same name.\n\n        Args:\n            from_: The retrieval job whose result should be persisted.\n            name: The name of the saved dataset.\n            storage: The saved dataset storage object indicating where the result should be persisted.\n            tags (optional): A dictionary of key-value pairs to store arbitrary metadata.\n            feature_service (optional): The feature service that should be associated with this saved dataset.\n            allow_overwrite (optional): If True, the persisted result can overwrite an existing table or file.\n\n        Returns:\n            SavedDataset object with attached RetrievalJob\n\n        Raises:\n            ValueError if given retrieval job doesn't have metadata\n        "
        if not flags_helper.is_test():
            warnings.warn('Saving dataset is an experimental feature. This API is unstable and it could and most probably will be changed in the future. We do not guarantee that future changes will maintain backward compatibility.', RuntimeWarning)
        if not from_.metadata:
            raise ValueError(f'The RetrievalJob {type(from_)} must implement the metadata property.')
        dataset = SavedDataset(name=name, features=from_.metadata.features, join_keys=from_.metadata.keys, full_feature_names=from_.full_feature_names, storage=storage, tags=tags, feature_service_name=feature_service.name if feature_service else None)
        dataset.min_event_timestamp = from_.metadata.min_event_timestamp
        dataset.max_event_timestamp = from_.metadata.max_event_timestamp
        from_.persist(storage=storage, allow_overwrite=allow_overwrite)
        dataset = dataset.with_retrieval_job(self._get_provider().retrieve_saved_dataset(config=self.config, dataset=dataset))
        self._registry.apply_saved_dataset(dataset, self.project, commit=True)
        return dataset

    @log_exceptions_and_usage
    def get_saved_dataset(self, name: str) -> SavedDataset:
        if False:
            return 10
        "\n        Find a saved dataset in the registry by provided name and\n        create a retrieval job to pull whole dataset from storage (offline store).\n\n        If dataset couldn't be found by provided name SavedDatasetNotFound exception will be raised.\n\n        Data will be retrieved from globally configured offline store.\n\n        Returns:\n            SavedDataset with RetrievalJob attached\n\n        Raises:\n            SavedDatasetNotFound\n        "
        if not flags_helper.is_test():
            warnings.warn('Retrieving datasets is an experimental feature. This API is unstable and it could and most probably will be changed in the future. We do not guarantee that future changes will maintain backward compatibility.', RuntimeWarning)
        dataset = self._registry.get_saved_dataset(name, self.project)
        provider = self._get_provider()
        retrieval_job = provider.retrieve_saved_dataset(config=self.config, dataset=dataset)
        return dataset.with_retrieval_job(retrieval_job)

    @log_exceptions_and_usage
    def materialize_incremental(self, end_date: datetime, feature_views: Optional[List[str]]=None) -> None:
        if False:
            return 10
        '\n        Materialize incremental new data from the offline store into the online store.\n\n        This method loads incremental new feature data up to the specified end time from either\n        the specified feature views, or all feature views if none are specified,\n        into the online store where it is available for online serving. The start time of\n        the interval materialized is either the most recent end time of a prior materialization or\n        (now - ttl) if no such prior materialization exists.\n\n        Args:\n            end_date (datetime): End date for time range of data to materialize into the online store\n            feature_views (List[str]): Optional list of feature view names. If selected, will only run\n                materialization for the specified feature views.\n\n        Raises:\n            Exception: A feature view being materialized does not have a TTL set.\n\n        Examples:\n            Materialize all features into the online store up to 5 minutes ago.\n\n            >>> from feast import FeatureStore, RepoConfig\n            >>> from datetime import datetime, timedelta\n            >>> fs = FeatureStore(repo_path="project/feature_repo")\n            >>> fs.materialize_incremental(end_date=datetime.utcnow() - timedelta(minutes=5))\n            Materializing...\n            <BLANKLINE>\n            ...\n        '
        feature_views_to_materialize = self._get_feature_views_to_materialize(feature_views)
        _print_materialization_log(None, end_date, len(feature_views_to_materialize), self.config.online_store.type)
        for feature_view in feature_views_to_materialize:
            start_date = feature_view.most_recent_end_time
            if start_date is None:
                if feature_view.ttl is None:
                    raise Exception(f'No start time found for feature view {feature_view.name}. materialize_incremental() requires either a ttl to be set or for materialize() to have been run at least once.')
                elif feature_view.ttl.total_seconds() > 0:
                    start_date = datetime.utcnow() - feature_view.ttl
                else:
                    print(f'Since the ttl is 0 for feature view {Style.BRIGHT + Fore.GREEN}{feature_view.name}{Style.RESET_ALL}, the start date will be set to 1 year before the current time.')
                    start_date = datetime.utcnow() - timedelta(weeks=52)
            provider = self._get_provider()
            print(f'{Style.BRIGHT + Fore.GREEN}{feature_view.name}{Style.RESET_ALL} from {Style.BRIGHT + Fore.GREEN}{start_date.replace(microsecond=0).astimezone()}{Style.RESET_ALL} to {Style.BRIGHT + Fore.GREEN}{end_date.replace(microsecond=0).astimezone()}{Style.RESET_ALL}:')

            def tqdm_builder(length):
                if False:
                    print('Hello World!')
                return tqdm(total=length, ncols=100)
            start_date = utils.make_tzaware(start_date)
            end_date = utils.make_tzaware(end_date)
            provider.materialize_single_feature_view(config=self.config, feature_view=feature_view, start_date=start_date, end_date=end_date, registry=self._registry, project=self.project, tqdm_builder=tqdm_builder)
            self._registry.apply_materialization(feature_view, self.project, start_date, end_date)

    @log_exceptions_and_usage
    def materialize(self, start_date: datetime, end_date: datetime, feature_views: Optional[List[str]]=None) -> None:
        if False:
            return 10
        '\n        Materialize data from the offline store into the online store.\n\n        This method loads feature data in the specified interval from either\n        the specified feature views, or all feature views if none are specified,\n        into the online store where it is available for online serving.\n\n        Args:\n            start_date (datetime): Start date for time range of data to materialize into the online store\n            end_date (datetime): End date for time range of data to materialize into the online store\n            feature_views (List[str]): Optional list of feature view names. If selected, will only run\n                materialization for the specified feature views.\n\n        Examples:\n            Materialize all features into the online store over the interval\n            from 3 hours ago to 10 minutes ago.\n            >>> from feast import FeatureStore, RepoConfig\n            >>> from datetime import datetime, timedelta\n            >>> fs = FeatureStore(repo_path="project/feature_repo")\n            >>> fs.materialize(\n            ...     start_date=datetime.utcnow() - timedelta(hours=3), end_date=datetime.utcnow() - timedelta(minutes=10)\n            ... )\n            Materializing...\n            <BLANKLINE>\n            ...\n        '
        if utils.make_tzaware(start_date) > utils.make_tzaware(end_date):
            raise ValueError(f'The given start_date {start_date} is greater than the given end_date {end_date}.')
        feature_views_to_materialize = self._get_feature_views_to_materialize(feature_views)
        _print_materialization_log(start_date, end_date, len(feature_views_to_materialize), self.config.online_store.type)
        for feature_view in feature_views_to_materialize:
            provider = self._get_provider()
            print(f'{Style.BRIGHT + Fore.GREEN}{feature_view.name}{Style.RESET_ALL}:')

            def tqdm_builder(length):
                if False:
                    while True:
                        i = 10
                return tqdm(total=length, ncols=100)
            start_date = utils.make_tzaware(start_date)
            end_date = utils.make_tzaware(end_date)
            provider.materialize_single_feature_view(config=self.config, feature_view=feature_view, start_date=start_date, end_date=end_date, registry=self._registry, project=self.project, tqdm_builder=tqdm_builder)
            self._registry.apply_materialization(feature_view, self.project, start_date, end_date)

    @log_exceptions_and_usage
    def push(self, push_source_name: str, df: pd.DataFrame, allow_registry_cache: bool=True, to: PushMode=PushMode.ONLINE):
        if False:
            return 10
        '\n        Push features to a push source. This updates all the feature views that have the push source as stream source.\n\n        Args:\n            push_source_name: The name of the push source we want to push data to.\n            df: The data being pushed.\n            allow_registry_cache: Whether to allow cached versions of the registry.\n            to: Whether to push to online or offline store. Defaults to online store only.\n        '
        from feast.data_source import PushSource
        all_fvs = self.list_feature_views(allow_cache=allow_registry_cache)
        all_fvs += self.list_stream_feature_views(allow_cache=allow_registry_cache)
        fvs_with_push_sources = {fv for fv in all_fvs if fv.stream_source is not None and isinstance(fv.stream_source, PushSource) and (fv.stream_source.name == push_source_name)}
        if not fvs_with_push_sources:
            raise PushSourceNotFoundException(push_source_name)
        for fv in fvs_with_push_sources:
            if to == PushMode.ONLINE or to == PushMode.ONLINE_AND_OFFLINE:
                self.write_to_online_store(fv.name, df, allow_registry_cache=allow_registry_cache)
            if to == PushMode.OFFLINE or to == PushMode.ONLINE_AND_OFFLINE:
                self.write_to_offline_store(fv.name, df, allow_registry_cache=allow_registry_cache)

    @log_exceptions_and_usage
    def write_to_online_store(self, feature_view_name: str, df: pd.DataFrame, allow_registry_cache: bool=True):
        if False:
            return 10
        '\n        Persists a dataframe to the online store.\n\n        Args:\n            feature_view_name: The feature view to which the dataframe corresponds.\n            df: The dataframe to be persisted.\n            allow_registry_cache (optional): Whether to allow retrieving feature views from a cached registry.\n        '
        try:
            feature_view = self.get_stream_feature_view(feature_view_name, allow_registry_cache=allow_registry_cache)
        except FeatureViewNotFoundException:
            feature_view = self.get_feature_view(feature_view_name, allow_registry_cache=allow_registry_cache)
        provider = self._get_provider()
        provider.ingest_df(feature_view, df)

    @log_exceptions_and_usage
    def write_to_offline_store(self, feature_view_name: str, df: pd.DataFrame, allow_registry_cache: bool=True, reorder_columns: bool=True):
        if False:
            print('Hello World!')
        '\n        Persists the dataframe directly into the batch data source for the given feature view.\n\n        Fails if the dataframe columns do not match the columns of the batch data source. Optionally\n        reorders the columns of the dataframe to match.\n        '
        try:
            feature_view = self.get_stream_feature_view(feature_view_name, allow_registry_cache=allow_registry_cache)
        except FeatureViewNotFoundException:
            feature_view = self.get_feature_view(feature_view_name, allow_registry_cache=allow_registry_cache)
        column_names_and_types = feature_view.batch_source.get_table_column_names_and_types(self.config)
        source_columns = [column for (column, _) in column_names_and_types]
        input_columns = df.columns.values.tolist()
        if set(input_columns) != set(source_columns):
            raise ValueError(f'The input dataframe has columns {set(input_columns)} but the batch source has columns {set(source_columns)}.')
        if reorder_columns:
            df = df.reindex(columns=source_columns)
        table = pa.Table.from_pandas(df)
        provider = self._get_provider()
        provider.ingest_df_to_offline_store(feature_view, table)

    @log_exceptions_and_usage
    def get_online_features(self, features: Union[List[str], FeatureService], entity_rows: List[Dict[str, Any]], full_feature_names: bool=False) -> OnlineResponse:
        if False:
            return 10
        '\n        Retrieves the latest online feature data.\n\n        Note: This method will download the full feature registry the first time it is run. If you are using a\n        remote registry like GCS or S3 then that may take a few seconds. The registry remains cached up to a TTL\n        duration (which can be set to infinity). If the cached registry is stale (more time than the TTL has\n        passed), then a new registry will be downloaded synchronously by this method. This download may\n        introduce latency to online feature retrieval. In order to avoid synchronous downloads, please call\n        refresh_registry() prior to the TTL being reached. Remember it is possible to set the cache TTL to\n        infinity (cache forever).\n\n        Args:\n            features: The list of features that should be retrieved from the online store. These features can be\n                specified either as a list of string feature references or as a feature service. String feature\n                references must have format "feature_view:feature", e.g. "customer_fv:daily_transactions".\n            entity_rows: A list of dictionaries where each key-value is an entity-name, entity-value pair.\n            full_feature_names: If True, feature names will be prefixed with the corresponding feature view name,\n                changing them from the format "feature" to "feature_view__feature" (e.g. "daily_transactions"\n                changes to "customer_fv__daily_transactions").\n\n        Returns:\n            OnlineResponse containing the feature data in records.\n\n        Raises:\n            Exception: No entity with the specified name exists.\n\n        Examples:\n            Retrieve online features from an online store.\n\n            >>> from feast import FeatureStore, RepoConfig\n            >>> fs = FeatureStore(repo_path="project/feature_repo")\n            >>> online_response = fs.get_online_features(\n            ...     features=[\n            ...         "driver_hourly_stats:conv_rate",\n            ...         "driver_hourly_stats:acc_rate",\n            ...         "driver_hourly_stats:avg_daily_trips",\n            ...     ],\n            ...     entity_rows=[{"driver_id": 1001}, {"driver_id": 1002}, {"driver_id": 1003}, {"driver_id": 1004}],\n            ... )\n            >>> online_response_dict = online_response.to_dict()\n        '
        columnar: Dict[str, List[Any]] = {k: [] for k in entity_rows[0].keys()}
        for entity_row in entity_rows:
            for (key, value) in entity_row.items():
                try:
                    columnar[key].append(value)
                except KeyError as e:
                    raise ValueError('All entity_rows must have the same keys.') from e
        return self._get_online_features(features=features, entity_values=columnar, full_feature_names=full_feature_names, native_entity_values=True)

    def _get_online_features(self, features: Union[List[str], FeatureService], entity_values: Mapping[str, Union[Sequence[Any], Sequence[Value], RepeatedValue]], full_feature_names: bool=False, native_entity_values: bool=True):
        if False:
            return 10
        entity_value_lists: Dict[str, Union[List[Any], List[Value]]] = {k: list(v) if isinstance(v, Sequence) else list(v.val) for (k, v) in entity_values.items()}
        _feature_refs = self._get_features(features, allow_cache=True)
        (requested_feature_views, requested_request_feature_views, requested_on_demand_feature_views) = self._get_feature_views_to_use(features=features, allow_cache=True, hide_dummy_entity=False)
        if requested_request_feature_views:
            warnings.warn('Request feature view is deprecated. Please use request data source instead', DeprecationWarning)
        (entity_name_to_join_key_map, entity_type_map, join_keys_set) = self._get_entity_maps(requested_feature_views)
        entity_proto_values: Dict[str, List[Value]]
        if native_entity_values:
            entity_proto_values = {k: python_values_to_proto_values(v, entity_type_map.get(k, ValueType.UNKNOWN)) for (k, v) in entity_value_lists.items()}
        else:
            entity_proto_values = entity_value_lists
        num_rows = _validate_entity_values(entity_proto_values)
        _validate_feature_refs(_feature_refs, full_feature_names)
        (grouped_refs, grouped_odfv_refs, grouped_request_fv_refs, _) = _group_feature_refs(_feature_refs, requested_feature_views, requested_request_feature_views, requested_on_demand_feature_views)
        set_usage_attribute('odfv', bool(grouped_odfv_refs))
        set_usage_attribute('request_fv', bool(grouped_request_fv_refs))
        requested_result_row_names = {feat_ref.replace(':', '__') for feat_ref in _feature_refs}
        if not full_feature_names:
            requested_result_row_names = {name.rpartition('__')[-1] for name in requested_result_row_names}
        feature_views = list((view for (view, _) in grouped_refs))
        (needed_request_data, needed_request_fv_features) = self.get_needed_request_data(grouped_odfv_refs, grouped_request_fv_refs)
        join_key_values: Dict[str, List[Value]] = {}
        request_data_features: Dict[str, List[Value]] = {}
        for (join_key_or_entity_name, values) in entity_proto_values.items():
            if join_key_or_entity_name in needed_request_data or join_key_or_entity_name in needed_request_fv_features:
                if join_key_or_entity_name in needed_request_fv_features:
                    requested_result_row_names.add(join_key_or_entity_name)
                request_data_features[join_key_or_entity_name] = values
            else:
                if join_key_or_entity_name in join_keys_set:
                    join_key = join_key_or_entity_name
                else:
                    try:
                        join_key = entity_name_to_join_key_map[join_key_or_entity_name]
                    except KeyError:
                        raise EntityNotFoundException(join_key_or_entity_name, self.project)
                    else:
                        warnings.warn('Using entity name is deprecated. Use join_key instead.')
                requested_result_row_names.add(join_key)
                join_key_values[join_key] = values
        self.ensure_request_data_values_exist(needed_request_data, needed_request_fv_features, request_data_features)
        online_features_response = GetOnlineFeaturesResponse(results=[])
        self._populate_result_rows_from_columnar(online_features_response=online_features_response, data=dict(**join_key_values, **request_data_features))
        entityless_case = DUMMY_ENTITY_NAME in [entity_name for feature_view in feature_views for entity_name in feature_view.entities]
        if entityless_case:
            join_key_values[DUMMY_ENTITY_ID] = python_values_to_proto_values([DUMMY_ENTITY_VAL] * num_rows, DUMMY_ENTITY.value_type)
        provider = self._get_provider()
        for (table, requested_features) in grouped_refs:
            (table_entity_values, idxs) = self._get_unique_entities(table, join_key_values, entity_name_to_join_key_map)
            feature_data = self._read_from_online_store(table_entity_values, provider, requested_features, table)
            self._populate_response_from_feature_data(feature_data, idxs, online_features_response, full_feature_names, requested_features, table)
        if grouped_odfv_refs:
            self._augment_response_with_on_demand_transforms(online_features_response, _feature_refs, requested_on_demand_feature_views, full_feature_names)
        self._drop_unneeded_columns(online_features_response, requested_result_row_names)
        return OnlineResponse(online_features_response)

    @staticmethod
    def _get_columnar_entity_values(rowise: Optional[List[Dict[str, Any]]], columnar: Optional[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        if False:
            return 10
        if rowise is None and columnar is None or (rowise is not None and columnar is not None):
            raise ValueError('Exactly one of `columnar_entity_values` and `rowise_entity_values` must be set.')
        if rowise is not None:
            res = defaultdict(list)
            for entity_row in rowise:
                for (key, value) in entity_row.items():
                    res[key].append(value)
            return res
        return cast(Dict[str, List[Any]], columnar)

    def _get_entity_maps(self, feature_views) -> Tuple[Dict[str, str], Dict[str, ValueType], Set[str]]:
        if False:
            i = 10
            return i + 15
        entities = self._list_entities(allow_cache=True, hide_dummy_entity=False)
        entity_name_to_join_key_map: Dict[str, str] = {}
        entity_type_map: Dict[str, ValueType] = {}
        for entity in entities:
            entity_name_to_join_key_map[entity.name] = entity.join_key
        for feature_view in feature_views:
            for entity_name in feature_view.entities:
                entity = self._registry.get_entity(entity_name, self.project, allow_cache=True)
                entity_name = feature_view.projection.join_key_map.get(entity.join_key, entity.name)
                join_key = feature_view.projection.join_key_map.get(entity.join_key, entity.join_key)
                entity_name_to_join_key_map[entity_name] = join_key
            for entity_column in feature_view.entity_columns:
                entity_type_map[entity_column.name] = entity_column.dtype.to_value_type()
        return (entity_name_to_join_key_map, entity_type_map, set(entity_name_to_join_key_map.values()))

    @staticmethod
    def _get_table_entity_values(table: FeatureView, entity_name_to_join_key_map: Dict[str, str], join_key_proto_values: Dict[str, List[Value]]) -> Dict[str, List[Value]]:
        if False:
            i = 10
            return i + 15
        table_join_keys = [entity_name_to_join_key_map[entity_name] for entity_name in table.entities]
        alias_to_join_key_map = {v: k for (k, v) in table.projection.join_key_map.items()}
        entity_values = {alias_to_join_key_map.get(k, k): v for (k, v) in join_key_proto_values.items() if alias_to_join_key_map.get(k, k) in table_join_keys}
        return entity_values

    @staticmethod
    def _populate_result_rows_from_columnar(online_features_response: GetOnlineFeaturesResponse, data: Dict[str, List[Value]]):
        if False:
            print('Hello World!')
        timestamp = Timestamp()
        for (feature_name, feature_values) in data.items():
            online_features_response.metadata.feature_names.val.append(feature_name)
            online_features_response.results.append(GetOnlineFeaturesResponse.FeatureVector(values=feature_values, statuses=[FieldStatus.PRESENT] * len(feature_values), event_timestamps=[timestamp] * len(feature_values)))

    @staticmethod
    def get_needed_request_data(grouped_odfv_refs: List[Tuple[OnDemandFeatureView, List[str]]], grouped_request_fv_refs: List[Tuple[RequestFeatureView, List[str]]]) -> Tuple[Set[str], Set[str]]:
        if False:
            i = 10
            return i + 15
        needed_request_data: Set[str] = set()
        needed_request_fv_features: Set[str] = set()
        for (odfv, _) in grouped_odfv_refs:
            odfv_request_data_schema = odfv.get_request_data_schema()
            needed_request_data.update(odfv_request_data_schema.keys())
        for (request_fv, _) in grouped_request_fv_refs:
            for feature in request_fv.features:
                needed_request_fv_features.add(feature.name)
        return (needed_request_data, needed_request_fv_features)

    @staticmethod
    def ensure_request_data_values_exist(needed_request_data: Set[str], needed_request_fv_features: Set[str], request_data_features: Dict[str, List[Any]]):
        if False:
            return 10
        if len(needed_request_data) + len(needed_request_fv_features) != len(request_data_features.keys()):
            missing_features = [x for x in itertools.chain(needed_request_data, needed_request_fv_features) if x not in request_data_features]
            raise RequestDataNotFoundInEntityRowsException(feature_names=missing_features)

    def _get_unique_entities(self, table: FeatureView, join_key_values: Dict[str, List[Value]], entity_name_to_join_key_map: Dict[str, str]) -> Tuple[Tuple[Dict[str, Value], ...], Tuple[List[int], ...]]:
        if False:
            return 10
        'Return the set of unique composite Entities for a Feature View and the indexes at which they appear.\n\n        This method allows us to query the OnlineStore for data we need only once\n        rather than requesting and processing data for the same combination of\n        Entities multiple times.\n        '
        table_entity_values = self._get_table_entity_values(table, entity_name_to_join_key_map, join_key_values)
        keys = table_entity_values.keys()
        rowise = list(enumerate(zip(*table_entity_values.values())))
        rowise.sort(key=lambda row: tuple((getattr(x, x.WhichOneof('val')) for x in row[1])))
        unique_entities: Tuple[Dict[str, Value], ...]
        indexes: Tuple[List[int], ...]
        (unique_entities, indexes) = tuple(zip(*[(dict(zip(keys, k)), [_[0] for _ in g]) for (k, g) in itertools.groupby(rowise, key=lambda x: x[1])]))
        return (unique_entities, indexes)

    def _read_from_online_store(self, entity_rows: Iterable[Mapping[str, Value]], provider: Provider, requested_features: List[str], table: FeatureView) -> List[Tuple[List[Timestamp], List['FieldStatus.ValueType'], List[Value]]]:
        if False:
            for i in range(10):
                print('nop')
        'Read and process data from the OnlineStore for a given FeatureView.\n\n        This method guarantees that the order of the data in each element of the\n        List returned is the same as the order of `requested_features`.\n\n        This method assumes that `provider.online_read` returns data for each\n        combination of Entities in `entity_rows` in the same order as they\n        are provided.\n        '
        entity_key_protos = [EntityKeyProto(join_keys=row.keys(), entity_values=row.values()) for row in entity_rows]
        read_rows = provider.online_read(config=self.config, table=table, entity_keys=entity_key_protos, requested_features=requested_features)
        null_value = Value()
        read_row_protos = []
        for read_row in read_rows:
            row_ts_proto = Timestamp()
            (row_ts, feature_data) = read_row
            if row_ts is not None:
                row_ts_proto.FromDatetime(row_ts)
            event_timestamps = [row_ts_proto] * len(requested_features)
            if feature_data is None:
                statuses = [FieldStatus.NOT_FOUND] * len(requested_features)
                values = [null_value] * len(requested_features)
            else:
                statuses = []
                values = []
                for feature_name in requested_features:
                    if feature_name not in feature_data:
                        statuses.append(FieldStatus.NOT_FOUND)
                        values.append(null_value)
                    else:
                        statuses.append(FieldStatus.PRESENT)
                        values.append(feature_data[feature_name])
            read_row_protos.append((event_timestamps, statuses, values))
        return read_row_protos

    @staticmethod
    def _populate_response_from_feature_data(feature_data: Iterable[Tuple[Iterable[Timestamp], Iterable['FieldStatus.ValueType'], Iterable[Value]]], indexes: Iterable[List[int]], online_features_response: GetOnlineFeaturesResponse, full_feature_names: bool, requested_features: Iterable[str], table: FeatureView):
        if False:
            i = 10
            return i + 15
        'Populate the GetOnlineFeaturesResponse with feature data.\n\n        This method assumes that `_read_from_online_store` returns data for each\n        combination of Entities in `entity_rows` in the same order as they\n        are provided.\n\n        Args:\n            feature_data: A list of data in Protobuf form which was retrieved from the OnlineStore.\n            indexes: A list of indexes which should be the same length as `feature_data`. Each list\n                of indexes corresponds to a set of result rows in `online_features_response`.\n            online_features_response: The object to populate.\n            full_feature_names: A boolean that provides the option to add the feature view prefixes to the feature names,\n                changing them from the format "feature" to "feature_view__feature" (e.g., "daily_transactions" changes to\n                "customer_fv__daily_transactions").\n            requested_features: The names of the features in `feature_data`. This should be ordered in the same way as the\n                data in `feature_data`.\n            table: The FeatureView that `feature_data` was retrieved from.\n        '
        requested_feature_refs = [f'{table.projection.name_to_use()}__{feature_name}' if full_feature_names else feature_name for feature_name in requested_features]
        online_features_response.metadata.feature_names.val.extend(requested_feature_refs)
        (timestamps, statuses, values) = zip(*feature_data)
        for (feature_idx, (timestamp_vector, statuses_vector, values_vector)) in enumerate(zip(zip(*timestamps), zip(*statuses), zip(*values))):
            online_features_response.results.append(GetOnlineFeaturesResponse.FeatureVector(values=apply_list_mapping(values_vector, indexes), statuses=apply_list_mapping(statuses_vector, indexes), event_timestamps=apply_list_mapping(timestamp_vector, indexes)))

    @staticmethod
    def _augment_response_with_on_demand_transforms(online_features_response: GetOnlineFeaturesResponse, feature_refs: List[str], requested_on_demand_feature_views: List[OnDemandFeatureView], full_feature_names: bool):
        if False:
            return 10
        'Computes on demand feature values and adds them to the result rows.\n\n        Assumes that \'online_features_response\' already contains the necessary request data and input feature\n        views for the on demand feature views. Unneeded feature values such as request data and\n        unrequested input feature views will be removed from \'online_features_response\'.\n\n        Args:\n            online_features_response: Protobuf object to populate\n            feature_refs: List of all feature references to be returned.\n            requested_on_demand_feature_views: List of all odfvs that have been requested.\n            full_feature_names: A boolean that provides the option to add the feature view prefixes to the feature names,\n                changing them from the format "feature" to "feature_view__feature" (e.g., "daily_transactions" changes to\n                "customer_fv__daily_transactions").\n        '
        requested_odfv_map = {odfv.name: odfv for odfv in requested_on_demand_feature_views}
        requested_odfv_feature_names = requested_odfv_map.keys()
        odfv_feature_refs = defaultdict(list)
        for feature_ref in feature_refs:
            (view_name, feature_name) = feature_ref.split(':')
            if view_name in requested_odfv_feature_names:
                odfv_feature_refs[view_name].append(f'{requested_odfv_map[view_name].projection.name_to_use()}__{feature_name}' if full_feature_names else feature_name)
        initial_response = OnlineResponse(online_features_response)
        initial_response_df = initial_response.to_df()
        odfv_result_names = set()
        for (odfv_name, _feature_refs) in odfv_feature_refs.items():
            odfv = requested_odfv_map[odfv_name]
            transformed_features_df = odfv.get_transformed_features_df(initial_response_df, full_feature_names)
            selected_subset = [f for f in transformed_features_df.columns if f in _feature_refs]
            proto_values = [python_values_to_proto_values(transformed_features_df[feature].values, ValueType.UNKNOWN) for feature in selected_subset]
            odfv_result_names |= set(selected_subset)
            online_features_response.metadata.feature_names.val.extend(selected_subset)
            for feature_idx in range(len(selected_subset)):
                online_features_response.results.append(GetOnlineFeaturesResponse.FeatureVector(values=proto_values[feature_idx], statuses=[FieldStatus.PRESENT] * len(proto_values[feature_idx]), event_timestamps=[Timestamp()] * len(proto_values[feature_idx])))

    @staticmethod
    def _drop_unneeded_columns(online_features_response: GetOnlineFeaturesResponse, requested_result_row_names: Set[str]):
        if False:
            while True:
                i = 10
        "\n        Unneeded feature values such as request data and unrequested input feature views will\n        be removed from 'online_features_response'.\n\n        Args:\n            online_features_response: Protobuf object to populate\n            requested_result_row_names: Fields from 'result_rows' that have been requested, and\n                    therefore should not be dropped.\n        "
        unneeded_feature_indices = [idx for (idx, val) in enumerate(online_features_response.metadata.feature_names.val) if val not in requested_result_row_names]
        for idx in reversed(unneeded_feature_indices):
            del online_features_response.metadata.feature_names.val[idx]
            del online_features_response.results[idx]

    def _get_feature_views_to_use(self, features: Optional[Union[List[str], FeatureService]], allow_cache=False, hide_dummy_entity: bool=True) -> Tuple[List[FeatureView], List[RequestFeatureView], List[OnDemandFeatureView]]:
        if False:
            while True:
                i = 10
        fvs = {fv.name: fv for fv in [*self._list_feature_views(allow_cache, hide_dummy_entity), *self._registry.list_stream_feature_views(project=self.project, allow_cache=allow_cache)]}
        request_fvs = {fv.name: fv for fv in self._registry.list_request_feature_views(project=self.project, allow_cache=allow_cache)}
        od_fvs = {fv.name: fv for fv in self._registry.list_on_demand_feature_views(project=self.project, allow_cache=allow_cache)}
        if isinstance(features, FeatureService):
            (fvs_to_use, request_fvs_to_use, od_fvs_to_use) = ([], [], [])
            for (fv_name, projection) in [(projection.name, projection) for projection in features.feature_view_projections]:
                if fv_name in fvs:
                    fvs_to_use.append(fvs[fv_name].with_projection(copy.copy(projection)))
                elif fv_name in request_fvs:
                    request_fvs_to_use.append(request_fvs[fv_name].with_projection(copy.copy(projection)))
                elif fv_name in od_fvs:
                    odfv = od_fvs[fv_name].with_projection(copy.copy(projection))
                    od_fvs_to_use.append(odfv)
                    for projection in odfv.source_feature_view_projections.values():
                        fv = fvs[projection.name].with_projection(copy.copy(projection))
                        if fv not in fvs_to_use:
                            fvs_to_use.append(fv)
                else:
                    raise ValueError(f"""The provided feature service {features.name} contains a reference to a feature view{fv_name} which doesn't exist. Please make sure that you have created the feature view{fv_name} and that you have registered it by running "apply".""")
            views_to_use = (fvs_to_use, request_fvs_to_use, od_fvs_to_use)
        else:
            views_to_use = ([*fvs.values()], [*request_fvs.values()], [*od_fvs.values()])
        return views_to_use

    @log_exceptions_and_usage
    def serve(self, host: str, port: int, type_: str, no_access_log: bool, no_feature_log: bool, workers: int, keep_alive_timeout: int, registry_ttl_sec: int) -> None:
        if False:
            return 10
        'Start the feature consumption server locally on a given port.'
        type_ = type_.lower()
        if type_ != 'http':
            raise ValueError(f"Python server only supports 'http'. Got '{type_}' instead.")
        feature_server.start_server(self, host=host, port=port, no_access_log=no_access_log, workers=workers, keep_alive_timeout=keep_alive_timeout, registry_ttl_sec=registry_ttl_sec)

    @log_exceptions_and_usage
    def get_feature_server_endpoint(self) -> Optional[str]:
        if False:
            print('Hello World!')
        'Returns endpoint for the feature server, if it exists.'
        return self._provider.get_feature_server_endpoint()

    @log_exceptions_and_usage
    def serve_ui(self, host: str, port: int, get_registry_dump: Callable, registry_ttl_sec: int, root_path: str='') -> None:
        if False:
            while True:
                i = 10
        'Start the UI server locally'
        if flags_helper.is_test():
            warnings.warn('The Feast UI is an experimental feature. We do not guarantee that future changes will maintain backward compatibility.', RuntimeWarning)
        ui_server.start_server(self, host=host, port=port, get_registry_dump=get_registry_dump, project_id=self.config.project, registry_ttl_sec=registry_ttl_sec, root_path=root_path)

    @log_exceptions_and_usage
    def serve_transformations(self, port: int) -> None:
        if False:
            print('Hello World!')
        'Start the feature transformation server locally on a given port.'
        warnings.warn('On demand feature view is an experimental feature. This API is stable, but the functionality does not scale well for offline retrieval', RuntimeWarning)
        from feast import transformation_server
        transformation_server.start_server(self, port)

    @log_exceptions_and_usage
    def write_logged_features(self, logs: Union[pa.Table, Path], source: FeatureService):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write logs produced by a source (currently only feature service is supported as a source)\n        to an offline store.\n\n        Args:\n            logs: Arrow Table or path to parquet dataset directory on disk\n            source: Object that produces logs\n        '
        if not isinstance(source, FeatureService):
            raise ValueError('Only feature service is currently supported as a source')
        assert source.logging_config is not None, 'Feature service must be configured with logging config in order to use this functionality'
        assert isinstance(logs, (pa.Table, Path))
        self._get_provider().write_feature_service_logs(feature_service=source, logs=logs, config=self.config, registry=self._registry)

    @log_exceptions_and_usage
    def validate_logged_features(self, source: FeatureService, start: datetime, end: datetime, reference: ValidationReference, throw_exception: bool=True, cache_profile: bool=True) -> Optional[ValidationFailed]:
        if False:
            while True:
                i = 10
        '\n        Load logged features from an offline store and validate them against provided validation reference.\n\n        Args:\n            source: Logs source object (currently only feature services are supported)\n            start: lower bound for loading logged features\n            end:  upper bound for loading logged features\n            reference: validation reference\n            throw_exception: throw exception or return it as a result\n            cache_profile: store cached profile in Feast registry\n\n        Returns:\n            Throw or return (depends on parameter) ValidationFailed exception if validation was not successful\n            or None if successful.\n\n        '
        if not flags_helper.is_test():
            warnings.warn('Logged features validation is an experimental feature. This API is unstable and it could and most probably will be changed in the future. We do not guarantee that future changes will maintain backward compatibility.', RuntimeWarning)
        if not isinstance(source, FeatureService):
            raise ValueError('Only feature service is currently supported as a source')
        j = self._get_provider().retrieve_feature_service_logs(feature_service=source, start_date=start, end_date=end, config=self.config, registry=self.registry)
        try:
            t = j.to_arrow(validation_reference=reference)
        except ValidationFailed as exc:
            if throw_exception:
                raise
            return exc
        else:
            print(f'{t.shape[0]} rows were validated.')
        if cache_profile:
            self.apply(reference)
        return None

    @log_exceptions_and_usage
    def get_validation_reference(self, name: str, allow_cache: bool=False) -> ValidationReference:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves a validation reference.\n\n        Raises:\n            ValidationReferenceNotFoundException: The validation reference could not be found.\n        '
        ref = self._registry.get_validation_reference(name, project=self.project, allow_cache=allow_cache)
        ref._dataset = self.get_saved_dataset(ref.dataset_name)
        return ref

def _validate_entity_values(join_key_values: Dict[str, List[Value]]):
    if False:
        print('Hello World!')
    set_of_row_lengths = {len(v) for v in join_key_values.values()}
    if len(set_of_row_lengths) > 1:
        raise ValueError('All entity rows must have the same columns.')
    return set_of_row_lengths.pop()

def _validate_feature_refs(feature_refs: List[str], full_feature_names: bool=False):
    if False:
        return 10
    '\n    Validates that there are no collisions among the feature references.\n\n    Args:\n        feature_refs: List of feature references to validate. Feature references must have format\n            "feature_view:feature", e.g. "customer_fv:daily_transactions".\n        full_feature_names: If True, the full feature references are compared for collisions; if False,\n            only the feature names are compared.\n\n    Raises:\n        FeatureNameCollisionError: There is a collision among the feature references.\n    '
    collided_feature_refs = []
    if full_feature_names:
        collided_feature_refs = [ref for (ref, occurrences) in Counter(feature_refs).items() if occurrences > 1]
    else:
        feature_names = [ref.split(':')[1] for ref in feature_refs]
        collided_feature_names = [ref for (ref, occurrences) in Counter(feature_names).items() if occurrences > 1]
        for feature_name in collided_feature_names:
            collided_feature_refs.extend([ref for ref in feature_refs if ref.endswith(':' + feature_name)])
    if len(collided_feature_refs) > 0:
        raise FeatureNameCollisionError(collided_feature_refs, full_feature_names)

def _group_feature_refs(features: List[str], all_feature_views: List[FeatureView], all_request_feature_views: List[RequestFeatureView], all_on_demand_feature_views: List[OnDemandFeatureView]) -> Tuple[List[Tuple[FeatureView, List[str]]], List[Tuple[OnDemandFeatureView, List[str]]], List[Tuple[RequestFeatureView, List[str]]], Set[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Get list of feature views and corresponding feature names based on feature references'
    view_index = {view.projection.name_to_use(): view for view in all_feature_views}
    request_view_index = {view.projection.name_to_use(): view for view in all_request_feature_views}
    on_demand_view_index = {view.projection.name_to_use(): view for view in all_on_demand_feature_views}
    views_features = defaultdict(set)
    request_views_features = defaultdict(set)
    request_view_refs = set()
    on_demand_view_features = defaultdict(set)
    for ref in features:
        (view_name, feat_name) = ref.split(':')
        if view_name in view_index:
            view_index[view_name].projection.get_feature(feat_name)
            views_features[view_name].add(feat_name)
        elif view_name in on_demand_view_index:
            on_demand_view_index[view_name].projection.get_feature(feat_name)
            on_demand_view_features[view_name].add(feat_name)
            for input_fv_projection in on_demand_view_index[view_name].source_feature_view_projections.values():
                for input_feat in input_fv_projection.features:
                    views_features[input_fv_projection.name].add(input_feat.name)
        elif view_name in request_view_index:
            request_view_index[view_name].projection.get_feature(feat_name)
            request_views_features[view_name].add(feat_name)
            request_view_refs.add(ref)
        else:
            raise FeatureViewNotFoundException(view_name)
    fvs_result: List[Tuple[FeatureView, List[str]]] = []
    odfvs_result: List[Tuple[OnDemandFeatureView, List[str]]] = []
    request_fvs_result: List[Tuple[RequestFeatureView, List[str]]] = []
    for (view_name, feature_names) in views_features.items():
        fvs_result.append((view_index[view_name], list(feature_names)))
    for (view_name, feature_names) in request_views_features.items():
        request_fvs_result.append((request_view_index[view_name], list(feature_names)))
    for (view_name, feature_names) in on_demand_view_features.items():
        odfvs_result.append((on_demand_view_index[view_name], list(feature_names)))
    return (fvs_result, odfvs_result, request_fvs_result, request_view_refs)

def _print_materialization_log(start_date, end_date, num_feature_views: int, online_store: str):
    if False:
        return 10
    if start_date:
        print(f'Materializing {Style.BRIGHT + Fore.GREEN}{num_feature_views}{Style.RESET_ALL} feature views from {Style.BRIGHT + Fore.GREEN}{start_date.replace(microsecond=0).astimezone()}{Style.RESET_ALL} to {Style.BRIGHT + Fore.GREEN}{end_date.replace(microsecond=0).astimezone()}{Style.RESET_ALL} into the {Style.BRIGHT + Fore.GREEN}{online_store}{Style.RESET_ALL} online store.\n')
    else:
        print(f'Materializing {Style.BRIGHT + Fore.GREEN}{num_feature_views}{Style.RESET_ALL} feature views to {Style.BRIGHT + Fore.GREEN}{end_date.replace(microsecond=0).astimezone()}{Style.RESET_ALL} into the {Style.BRIGHT + Fore.GREEN}{online_store}{Style.RESET_ALL} online store.\n')

def _validate_feature_views(feature_views: List[BaseFeatureView]):
    if False:
        return 10
    'Verify feature views have case-insensitively unique names'
    fv_names = set()
    for fv in feature_views:
        case_insensitive_fv_name = fv.name.lower()
        if case_insensitive_fv_name in fv_names:
            raise ValueError(f'More than one feature view with name {case_insensitive_fv_name} found. Please ensure that all feature view names are case-insensitively unique. It may be necessary to ignore certain files in your feature repository by using a .feastignore file.')
        else:
            fv_names.add(case_insensitive_fv_name)

def _validate_data_sources(data_sources: List[DataSource]):
    if False:
        print('Hello World!')
    'Verify data sources have case-insensitively unique names.'
    ds_names = set()
    for ds in data_sources:
        case_insensitive_ds_name = ds.name.lower()
        if case_insensitive_ds_name in ds_names:
            raise DataSourceRepeatNamesException(case_insensitive_ds_name)
        else:
            ds_names.add(case_insensitive_ds_name)

def apply_list_mapping(lst: Iterable[Any], mapping_indexes: Iterable[List[int]]) -> Iterable[Any]:
    if False:
        return 10
    output_len = sum((len(item) for item in mapping_indexes))
    output = [None] * output_len
    for (elem, destinations) in zip(lst, mapping_indexes):
        for idx in destinations:
            output[idx] = elem
    return output