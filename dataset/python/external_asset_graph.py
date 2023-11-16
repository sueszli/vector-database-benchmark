from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import dagster._check as check
from dagster._core.definitions.assets_job import ASSET_BASE_JOB_PREFIX
from dagster._core.definitions.auto_materialize_policy import AutoMaterializePolicy
from dagster._core.host_representation.external import ExternalRepository
from dagster._core.host_representation.handle import RepositoryHandle
from dagster._core.selector.subset_selector import DependencyGraph
from dagster._core.workspace.workspace import IWorkspace

from .asset_graph import AssetGraph, AssetKeyOrCheckKey
from .backfill_policy import BackfillPolicy
from .events import AssetKey
from .freshness_policy import FreshnessPolicy
from .partition import PartitionsDefinition
from .partition_mapping import PartitionMapping

if TYPE_CHECKING:
    from dagster._core.host_representation.external_data import (
        ExternalAssetCheck,
        ExternalAssetNode,
    )


class ExternalAssetGraph(AssetGraph):
    def __init__(
        self,
        asset_dep_graph: DependencyGraph[AssetKey],
        source_asset_keys: AbstractSet[AssetKey],
        partitions_defs_by_key: Mapping[AssetKey, Optional[PartitionsDefinition]],
        partition_mappings_by_key: Mapping[AssetKey, Optional[Mapping[AssetKey, PartitionMapping]]],
        group_names_by_key: Mapping[AssetKey, Optional[str]],
        freshness_policies_by_key: Mapping[AssetKey, Optional[FreshnessPolicy]],
        auto_materialize_policies_by_key: Mapping[AssetKey, Optional[AutoMaterializePolicy]],
        backfill_policies_by_key: Mapping[AssetKey, Optional[BackfillPolicy]],
        repo_handles_by_key: Mapping[AssetKey, RepositoryHandle],
        job_names_by_key: Mapping[AssetKey, Sequence[str]],
        code_versions_by_key: Mapping[AssetKey, Optional[str]],
        is_observable_by_key: Mapping[AssetKey, bool],
        auto_observe_interval_minutes_by_key: Mapping[AssetKey, Optional[float]],
        required_assets_and_checks_by_key: Mapping[
            AssetKeyOrCheckKey, AbstractSet[AssetKeyOrCheckKey]
        ],
    ):
        super().__init__(
            asset_dep_graph=asset_dep_graph,
            source_asset_keys=source_asset_keys,
            partitions_defs_by_key=partitions_defs_by_key,
            partition_mappings_by_key=partition_mappings_by_key,
            group_names_by_key=group_names_by_key,
            freshness_policies_by_key=freshness_policies_by_key,
            auto_materialize_policies_by_key=auto_materialize_policies_by_key,
            backfill_policies_by_key=backfill_policies_by_key,
            code_versions_by_key=code_versions_by_key,
            is_observable_by_key=is_observable_by_key,
            auto_observe_interval_minutes_by_key=auto_observe_interval_minutes_by_key,
            required_assets_and_checks_by_key=required_assets_and_checks_by_key,
        )
        self._repo_handles_by_key = repo_handles_by_key
        self._materialization_job_names_by_key = job_names_by_key

        self._asset_keys_by_job_name: Mapping[str, List[AssetKey]] = defaultdict(list)
        for asset_key, job_names in self._materialization_job_names_by_key.items():
            for job_name in job_names:
                self._asset_keys_by_job_name[job_name].append(asset_key)

    @classmethod
    def from_workspace(cls, context: IWorkspace) -> "ExternalAssetGraph":
        code_locations = (
            location_entry.code_location
            for location_entry in context.get_workspace_snapshot().values()
            if location_entry.code_location
        )
        repos = (
            repo
            for code_location in code_locations
            for repo in code_location.get_repositories().values()
        )
        repo_handle_external_asset_nodes: Sequence[
            Tuple[RepositoryHandle, "ExternalAssetNode"]
        ] = []
        asset_checks: Sequence["ExternalAssetCheck"] = []

        for repo in repos:
            for external_asset_node in repo.get_external_asset_nodes():
                repo_handle_external_asset_nodes.append((repo.handle, external_asset_node))

            asset_checks.extend(repo.get_external_asset_checks())

        return cls.from_repository_handles_and_external_asset_nodes(
            repo_handle_external_asset_nodes=repo_handle_external_asset_nodes,
            external_asset_checks=asset_checks,
        )

    @classmethod
    def from_external_repository(
        cls, external_repository: ExternalRepository
    ) -> "ExternalAssetGraph":
        return cls.from_repository_handles_and_external_asset_nodes(
            repo_handle_external_asset_nodes=[
                (external_repository.handle, asset_node)
                for asset_node in external_repository.get_external_asset_nodes()
            ],
            external_asset_checks=external_repository.get_external_asset_checks(),
        )

    @classmethod
    def from_repository_handles_and_external_asset_nodes(
        cls,
        repo_handle_external_asset_nodes: Sequence[Tuple[RepositoryHandle, "ExternalAssetNode"]],
        external_asset_checks: Sequence["ExternalAssetCheck"],
    ) -> "ExternalAssetGraph":
        upstream: Dict[AssetKey, AbstractSet[AssetKey]] = {}
        source_asset_keys: Set[AssetKey] = set()
        partitions_defs_by_key: Dict[AssetKey, Optional[PartitionsDefinition]] = {}
        partition_mappings_by_key: Dict[AssetKey, Dict[AssetKey, PartitionMapping]] = defaultdict(
            defaultdict
        )
        group_names_by_key = {}
        freshness_policies_by_key = {}
        auto_materialize_policies_by_key = {}
        backfill_policies_by_key = {}
        keys_by_atomic_execution_unit_id: Dict[str, Set[AssetKeyOrCheckKey]] = defaultdict(set)
        repo_handles_by_key = {
            node.asset_key: repo_handle
            for repo_handle, node in repo_handle_external_asset_nodes
            if not node.is_source or node.is_observable
        }
        job_names_by_key = {
            node.asset_key: node.job_names
            for _, node in repo_handle_external_asset_nodes
            if not node.is_source or node.is_observable
        }
        code_versions_by_key = {
            node.asset_key: node.code_version
            for _, node in repo_handle_external_asset_nodes
            if not node.is_source
        }

        all_non_source_keys = {
            node.asset_key for _, node in repo_handle_external_asset_nodes if not node.is_source
        }

        is_observable_by_key = {key: False for key in all_non_source_keys}
        auto_observe_interval_minutes_by_key = {}

        for repo_handle, node in repo_handle_external_asset_nodes:
            if node.is_source:
                # We need to set this even if the node is a regular asset in another code location.
                # `is_observable` will only ever be consulted in the source asset context.
                is_observable_by_key[node.asset_key] = node.is_observable
                auto_observe_interval_minutes_by_key[
                    node.asset_key
                ] = node.auto_observe_interval_minutes

                if node.asset_key in all_non_source_keys:
                    # one location's source is another location's non-source
                    continue

                source_asset_keys.add(node.asset_key)

            upstream[node.asset_key] = {dep.upstream_asset_key for dep in node.dependencies}
            for dep in node.dependencies:
                if dep.partition_mapping is not None:
                    partition_mappings_by_key[node.asset_key][
                        dep.upstream_asset_key
                    ] = dep.partition_mapping
            partitions_defs_by_key[node.asset_key] = (
                node.partitions_def_data.get_partitions_definition()
                if node.partitions_def_data
                else None
            )
            group_names_by_key[node.asset_key] = node.group_name
            freshness_policies_by_key[node.asset_key] = node.freshness_policy
            auto_materialize_policies_by_key[node.asset_key] = node.auto_materialize_policy
            backfill_policies_by_key[node.asset_key] = node.backfill_policy

            if node.atomic_execution_unit_id is not None:
                keys_by_atomic_execution_unit_id[node.atomic_execution_unit_id].add(node.asset_key)

        for asset_check in external_asset_checks:
            if asset_check.atomic_execution_unit_id is not None:
                keys_by_atomic_execution_unit_id[asset_check.atomic_execution_unit_id].add(
                    asset_check.key
                )

        downstream: Dict[AssetKey, Set[AssetKey]] = defaultdict(set)
        for asset_key, upstream_keys in upstream.items():
            for upstream_key in upstream_keys:
                downstream[upstream_key].add(asset_key)

        required_assets_and_checks_by_key: Dict[
            AssetKeyOrCheckKey, AbstractSet[AssetKeyOrCheckKey]
        ] = {}
        for keys in keys_by_atomic_execution_unit_id.values():
            if len(keys) > 1:
                for key in keys:
                    required_assets_and_checks_by_key[key] = keys

        return cls(
            asset_dep_graph={"upstream": upstream, "downstream": downstream},
            source_asset_keys=source_asset_keys,
            partitions_defs_by_key=partitions_defs_by_key,
            partition_mappings_by_key=partition_mappings_by_key,
            group_names_by_key=group_names_by_key,
            freshness_policies_by_key=freshness_policies_by_key,
            auto_materialize_policies_by_key=auto_materialize_policies_by_key,
            backfill_policies_by_key=backfill_policies_by_key,
            repo_handles_by_key=repo_handles_by_key,
            job_names_by_key=job_names_by_key,
            code_versions_by_key=code_versions_by_key,
            is_observable_by_key=is_observable_by_key,
            auto_observe_interval_minutes_by_key=auto_observe_interval_minutes_by_key,
            required_assets_and_checks_by_key=required_assets_and_checks_by_key,
        )

    @property
    def repository_handles_by_key(self) -> Mapping[AssetKey, RepositoryHandle]:
        return self._repo_handles_by_key

    def get_repository_handle(self, asset_key: AssetKey) -> RepositoryHandle:
        return self._repo_handles_by_key[asset_key]

    def get_materialization_job_names(self, asset_key: AssetKey) -> Iterable[str]:
        """Returns the names of jobs that materialize this asset."""
        return self._materialization_job_names_by_key[asset_key]

    def get_materialization_asset_keys_for_job(self, job_name: str) -> Sequence[AssetKey]:
        """Returns asset keys that are targeted for materialization in the given job."""
        return [
            k
            for k in self.materializable_asset_keys
            if job_name in self.get_materialization_job_names(k)
        ]

    def get_asset_keys_for_job(self, job_name: str) -> Sequence[AssetKey]:
        return self._asset_keys_by_job_name[job_name]

    def get_implicit_job_name_for_assets(
        self,
        asset_keys: Iterable[AssetKey],
        external_repo: Optional[ExternalRepository],
    ) -> Optional[str]:
        """Returns the name of the asset base job that contains all the given assets, or None if there is no such
        job.

        Note: all asset_keys should be in the same repository.
        """
        if all(self.is_observable(asset_key) for asset_key in asset_keys):
            if external_repo is None:
                check.failed(
                    "external_repo must be passed in when getting job names for observable assets"
                )
            # for observable source assets, we need to select the job based on the partitions def
            target_partitions_defs = {
                self.get_partitions_def(asset_key) for asset_key in asset_keys
            }
            check.invariant(len(target_partitions_defs) == 1, "Expected exactly one partitions def")
            target_partitions_def = next(iter(target_partitions_defs))

            # create a mapping from job name to the partitions def of that job
            partitions_def_by_job_name = {}
            for (
                external_partition_set_data
            ) in external_repo.external_repository_data.external_partition_set_datas:
                if external_partition_set_data.external_partitions_data is None:
                    partitions_def = None
                else:
                    partitions_def = external_partition_set_data.external_partitions_data.get_partitions_definition()
                partitions_def_by_job_name[external_partition_set_data.job_name] = partitions_def
            # add any jobs that don't have a partitions def
            for external_job in external_repo.get_all_external_jobs():
                job_name = external_job.external_job_data.name
                if job_name not in partitions_def_by_job_name:
                    partitions_def_by_job_name[job_name] = None
            # find the job that matches the expected partitions definition
            for job_name, external_partitions_def in partitions_def_by_job_name.items():
                if not job_name.startswith(ASSET_BASE_JOB_PREFIX):
                    continue
                if (
                    # unpartitioned observable source assets may be materialized in any job
                    target_partitions_def is None
                    or external_partitions_def == target_partitions_def
                ) and all(
                    asset_key in self._asset_keys_by_job_name[job_name] for asset_key in asset_keys
                ):
                    return job_name
        else:
            for job_name in sorted(self._asset_keys_by_job_name.keys()):
                if not job_name.startswith(ASSET_BASE_JOB_PREFIX):
                    continue
                if all(
                    asset_key in self._asset_keys_by_job_name[job_name] for asset_key in asset_keys
                ):
                    return job_name
        return None

    def split_asset_keys_by_repository(
        self, asset_keys: AbstractSet[AssetKey]
    ) -> Sequence[AbstractSet[AssetKey]]:
        asset_keys_by_repo = defaultdict(set)
        for asset_key in asset_keys:
            repo_handle = self.get_repository_handle(asset_key)
            asset_keys_by_repo[(repo_handle.location_name, repo_handle.repository_name)].add(
                asset_key
            )
        return list(asset_keys_by_repo.values())
