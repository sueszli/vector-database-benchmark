from typing import List, Sequence

from dagster import _check as check
from dagster._core.definitions.asset_spec import (
    SYSTEM_METADATA_KEY_ASSET_EXECUTION_TYPE,
    AssetExecutionType,
    AssetSpec,
)
from dagster._core.definitions.assets import AssetsDefinition
from dagster._core.definitions.decorators.asset_decorator import asset, multi_asset
from dagster._core.definitions.source_asset import (
    SourceAsset,
    wrap_source_asset_observe_fn_in_op_compute_fn,
)
from dagster._core.errors import DagsterInvariantViolationError
from dagster._core.execution.context.compute import AssetExecutionContext


def external_asset_from_spec(spec: AssetSpec) -> AssetsDefinition:
    return external_assets_from_specs([spec])[0]


def external_assets_from_specs(specs: Sequence[AssetSpec]) -> List[AssetsDefinition]:
    """Create an external assets definition from a sequence of asset specs.

    An external asset is an asset that is not materialized by Dagster, but is tracked in the
    asset graph and asset catalog.

    A common use case for external assets is modeling data produced by an process not
    under Dagster's control. For example a daily drop of a file from a third party in s3.

    In most systems these are described as sources. This includes Dagster, which includes
    :py:class:`SourceAsset`, which will be supplanted by external assets in the near-term
    future, as external assets are a superset of the functionality
    of Source Assets.

    External assets can act as sources, but that is not their only use.

    In particular, external assets have themselves have lineage-specified through the
    ``deps`` argument of :py:class:`AssetSpec`- and can depend on other external assets.
    External assets are not allowed to depend on non-external assets.

    The user can emit `AssetMaterialization`, `AssetObservation`, and `AssetCheckEvaluations`
    events attached external assets.  And Dagster now has the ability to have "runless"
    events to enable many use cases that were previously not possible.  Runless events
    are events generated outside the context of a particular run (for example, in a
    sensor or by an script), allowing for greater flexibility in event generation.
    This can be done in a few ways:

    Note to reviewers that this in an in-progress doc block and the below will have links and examples.

    1) DagsterInstance exposes `report_runless_event` that can be used to generate events for
        external assets directly on an instance. See docs.
    2) Sensors can build these events and return them using :py:class:`SensorResult`. A use
        case for this is using a sensor to continously monitor the metadata exhaust from
        an external system and inserting events that
        reflect that exhaust. See docs.
    3) Dagster Cloud exposes a REST API for ingesting runless events. Users can copy and
        paste a curl command in the their external computations (such as Airflow operator)
        to register metadata associated with those computations See docs.
    4) Dagster ops can generate these events directly and yield them or by calling
        ``log_event`` on :py:class:`OpExecutionContext`.  Use cases for this include
        querying metadata in an external system that is too expensive to do so in a sensor. Or
        for adapting pure op-based Dagster code to take advantage of asset-oriented lineage,
        observability, and data quality features, without having to port them wholesale
        to `@asset`- and `@multi_asset`-based code.

    This feature set allows users to use Dagster as an observability, lineage, and
    data quality tool for assets that are not materialized by Dagster. In addition to
    traditional use cases like sources, this feature can model entire lineage graphs of
    assets that are scheduled and materialized by other tools and workflow engines. This
    allows users to use Dagster as a cross-cutting observability tool without migrating
    their entire data platform to a single orchestration engine.

    External assets do not have all the features of normal assets: they cannot be
    materialized ad hoc by Dagster (this is diabled in the UI); cannot be backfilled; cannot
    be scheduled using auto-materialize policies; and opt out of other features around
    direct materialization, both now and in the future. External assets also provide fewer
    guarantees around the correctness of information of their information in the asset
    catalog. In other words, in exchange for the flexibility Dagster provides less guardrails
    for external assets than assets that are materialized by Dagster, and there is an increased
    chance that they will insert non-sensical information into the asset catalog, potentially
    eroding trust.

    Args:
            specs (Sequence[AssetSpec]): The specs for the assets.
    """
    assets_defs = []
    for spec in specs:
        check.invariant(
            spec.auto_materialize_policy is None,
            "auto_materialize_policy must be None since it is ignored",
        )
        check.invariant(spec.code_version is None, "code_version must be None since it is ignored")
        check.invariant(
            spec.freshness_policy is None, "freshness_policy must be None since it is ignored"
        )
        check.invariant(
            spec.skippable is False,
            "skippable must be False since it is ignored and False is the default",
        )

        @multi_asset(
            name=spec.key.to_python_identifier(),
            specs=[
                AssetSpec(
                    key=spec.key,
                    description=spec.description,
                    group_name=spec.group_name,
                    metadata={
                        **(spec.metadata or {}),
                        **{
                            SYSTEM_METADATA_KEY_ASSET_EXECUTION_TYPE: (
                                AssetExecutionType.UNEXECUTABLE.value
                            )
                        },
                    },
                    deps=spec.deps,
                )
            ],
        )
        def _external_assets_def(context: AssetExecutionContext) -> None:
            raise DagsterInvariantViolationError(
                "You have attempted to execute an unexecutable asset"
                f" {context.asset_key.to_user_string}."
            )

        assets_defs.append(_external_assets_def)

    return assets_defs


def create_external_asset_from_source_asset(source_asset: SourceAsset) -> AssetsDefinition:
    check.invariant(
        source_asset.auto_observe_interval_minutes is None,
        "Automatically observed external assets not supported yet: auto_observe_interval_minutes"
        " should be None",
    )

    injected_metadata = (
        {SYSTEM_METADATA_KEY_ASSET_EXECUTION_TYPE: AssetExecutionType.UNEXECUTABLE.value}
        if source_asset.observe_fn is None
        else {}
    )

    injected_metadata = (
        {SYSTEM_METADATA_KEY_ASSET_EXECUTION_TYPE: AssetExecutionType.UNEXECUTABLE.value}
        if source_asset.observe_fn is None
        else {SYSTEM_METADATA_KEY_ASSET_EXECUTION_TYPE: AssetExecutionType.OBSERVATION.value}
    )

    kwargs = {
        "key": source_asset.key,
        "metadata": {
            **source_asset.metadata,
            **injected_metadata,
        },
        "group_name": source_asset.group_name,
        "description": source_asset.description,
        "partitions_def": source_asset.partitions_def,
    }

    if source_asset.io_manager_def:
        kwargs["io_manager_def"] = source_asset.io_manager_def
    elif source_asset.io_manager_key:
        kwargs["io_manager_key"] = source_asset.io_manager_key

    @asset(**kwargs)
    def _shim_assets_def(context: AssetExecutionContext):
        if not source_asset.observe_fn:
            raise NotImplementedError(f"Asset {source_asset.key} is not executable")

        op_function = wrap_source_asset_observe_fn_in_op_compute_fn(source_asset)
        return_value = op_function.decorated_fn(context)
        check.invariant(
            return_value is None,
            "The wrapped decorated_fn should return a value. If this changes, this code path must"
            " changed to process the events appopriately.",
        )

    check.invariant(isinstance(_shim_assets_def, AssetsDefinition))
    assert isinstance(_shim_assets_def, AssetsDefinition)  # appease pyright

    return _shim_assets_def
