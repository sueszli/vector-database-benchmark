from abc import ABC, abstractmethod
from typing import AbstractSet, Optional, Sequence, Union
import dagster._check as check
from dagster._config import ConfigSchemaSnapshot
from dagster._core.snap.dagster_types import DagsterTypeSnap
from dagster._core.snap.dep_snapshot import DependencyStructureIndex
from dagster._core.snap.job_snapshot import JobSnapshot
from dagster._core.snap.mode import ModeDefSnap
from dagster._core.snap.node import GraphDefSnap, OpDefSnap
from .job_index import JobIndex

class RepresentedJob(ABC):
    """RepresentedJob is a base class for ExternalPipeline or HistoricalPipeline.

    The name is "represented" because this is an in-memory representation of a job.
    The representation of a job could be referring to a job resident in
    another process *or* could be referring to a historical view of the job.
    """

    @property
    @abstractmethod
    def _job_index(self) -> JobIndex:
        if False:
            while True:
                i = 10
        ...

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        return self._job_index.name

    @property
    def description(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._job_index.description

    @property
    @abstractmethod
    def computed_job_snapshot_id(self) -> str:
        if False:
            while True:
                i = 10
        pass

    @property
    @abstractmethod
    def identifying_job_snapshot_id(self) -> str:
        if False:
            i = 10
            return i + 15
        pass

    @property
    def job_snapshot(self) -> JobSnapshot:
        if False:
            print('Hello World!')
        return self._job_index.job_snapshot

    @property
    def parent_job_snapshot(self) -> Optional[JobSnapshot]:
        if False:
            print('Hello World!')
        return self._job_index.parent_job_snapshot

    @property
    def op_selection(self) -> Optional[Sequence[str]]:
        if False:
            while True:
                i = 10
        return self._job_index.job_snapshot.lineage_snapshot.op_selection if self._job_index.job_snapshot.lineage_snapshot else None

    @property
    def resolved_op_selection(self) -> Optional[AbstractSet[str]]:
        if False:
            i = 10
            return i + 15
        return self._job_index.job_snapshot.lineage_snapshot.resolved_op_selection if self._job_index.job_snapshot.lineage_snapshot else None

    @property
    def config_schema_snapshot(self) -> ConfigSchemaSnapshot:
        if False:
            print('Hello World!')
        return self._job_index.config_schema_snapshot

    @property
    def dagster_type_snaps(self) -> Sequence[DagsterTypeSnap]:
        if False:
            return 10
        return self._job_index.get_dagster_type_snaps()

    def has_dagster_type_named(self, type_name: str) -> bool:
        if False:
            print('Hello World!')
        return self._job_index.has_dagster_type_name(type_name)

    def get_dagster_type_by_name(self, type_name: str) -> DagsterTypeSnap:
        if False:
            i = 10
            return i + 15
        return self._job_index.get_dagster_type_from_name(type_name)

    @property
    def mode_def_snaps(self) -> Sequence[ModeDefSnap]:
        if False:
            print('Hello World!')
        return self._job_index.job_snapshot.mode_def_snaps

    def get_mode_def_snap(self, mode_name: str) -> ModeDefSnap:
        if False:
            while True:
                i = 10
        return self._job_index.get_mode_def_snap(mode_name)

    @property
    def dep_structure_index(self) -> DependencyStructureIndex:
        if False:
            i = 10
            return i + 15
        return self._job_index.dep_structure_index

    def get_node_def_snap(self, node_def_name: str) -> Union[OpDefSnap, GraphDefSnap]:
        if False:
            return 10
        check.str_param(node_def_name, 'node_def_name')
        return self._job_index.get_node_def_snap(node_def_name)

    def get_dep_structure_index(self, node_def_name: str) -> DependencyStructureIndex:
        if False:
            while True:
                i = 10
        check.str_param(node_def_name, 'node_def_name')
        return self._job_index.get_dep_structure_index(node_def_name)

    def get_graph_name(self) -> str:
        if False:
            while True:
                i = 10
        return self._job_index.job_snapshot.graph_def_name