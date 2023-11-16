from typing import TYPE_CHECKING, Union
import dagster._check as check
from dagster._core.instance import DagsterInstance
from dagster._core.storage.dagster_run import DagsterRun
from dagster_graphql.schema.util import ResolveInfo
from .external import get_external_job_or_raise, get_full_external_job_or_raise
from .utils import JobSubsetSelector, UserFacingGraphQLError
if TYPE_CHECKING:
    from ..schema.pipelines.pipeline import GraphenePipeline
    from ..schema.pipelines.pipeline_ref import GrapheneUnknownPipeline
    from ..schema.pipelines.snapshot import GraphenePipelineSnapshot

def get_job_snapshot_or_error_from_job_selector(graphene_info: ResolveInfo, job_selector: JobSubsetSelector) -> 'GraphenePipelineSnapshot':
    if False:
        while True:
            i = 10
    from ..schema.pipelines.snapshot import GraphenePipelineSnapshot
    check.inst_param(job_selector, 'pipeline_selector', JobSubsetSelector)
    return GraphenePipelineSnapshot(get_full_external_job_or_raise(graphene_info, job_selector))

def get_job_snapshot_or_error_from_snapshot_id(graphene_info: ResolveInfo, snapshot_id: str) -> 'GraphenePipelineSnapshot':
    if False:
        print('Hello World!')
    check.str_param(snapshot_id, 'snapshot_id')
    return _get_job_snapshot_from_instance(graphene_info.context.instance, snapshot_id)

def _get_job_snapshot_from_instance(instance: DagsterInstance, snapshot_id: str) -> 'GraphenePipelineSnapshot':
    if False:
        return 10
    from ..schema.errors import GraphenePipelineSnapshotNotFoundError
    from ..schema.pipelines.snapshot import GraphenePipelineSnapshot
    if not instance.has_job_snapshot(snapshot_id):
        raise UserFacingGraphQLError(GraphenePipelineSnapshotNotFoundError(snapshot_id))
    historical_pipeline = instance.get_historical_job(snapshot_id)
    if not historical_pipeline:
        raise UserFacingGraphQLError(GraphenePipelineSnapshotNotFoundError(snapshot_id))
    return GraphenePipelineSnapshot(historical_pipeline)

def get_job_or_error(graphene_info: ResolveInfo, selector: JobSubsetSelector) -> 'GraphenePipeline':
    if False:
        for i in range(10):
            print('nop')
    'Returns a PipelineOrError.'
    return get_job_from_selector(graphene_info, selector)

def get_job_or_raise(graphene_info: ResolveInfo, selector: JobSubsetSelector) -> 'GraphenePipeline':
    if False:
        while True:
            i = 10
    'Returns a Pipeline or raises a UserFacingGraphQLError if one cannot be retrieved\n    from the selector, e.g., the pipeline is not present in the loaded repository.\n    '
    return get_job_from_selector(graphene_info, selector)

def get_job_reference_or_raise(graphene_info: ResolveInfo, dagster_run: DagsterRun) -> Union['GraphenePipelineSnapshot', 'GrapheneUnknownPipeline']:
    if False:
        for i in range(10):
            print('nop')
    'Returns a PipelineReference or raises a UserFacingGraphQLError if a pipeline\n    reference cannot be retrieved based on the run, e.g, a UserFacingGraphQLError that wraps an\n    InvalidSubsetError.\n    '
    from ..schema.pipelines.pipeline_ref import GrapheneUnknownPipeline
    check.inst_param(dagster_run, 'pipeline_run', DagsterRun)
    op_selection = list(dagster_run.resolved_op_selection) if dagster_run.resolved_op_selection else None
    if dagster_run.job_snapshot_id is None:
        return GrapheneUnknownPipeline(dagster_run.job_name, op_selection)
    return _get_job_snapshot_from_instance(graphene_info.context.instance, dagster_run.job_snapshot_id)

def get_job_from_selector(graphene_info: ResolveInfo, selector: JobSubsetSelector) -> 'GraphenePipeline':
    if False:
        for i in range(10):
            print('nop')
    from ..schema.pipelines.pipeline import GraphenePipeline
    check.inst_param(selector, 'selector', JobSubsetSelector)
    return GraphenePipeline(get_external_job_or_raise(graphene_info, selector))