import re
from typing import NamedTuple

import pytest
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetKey,
    DagsterEventType,
    DagsterInstance,
    Definitions,
    EventRecordsFilter,
    ExecuteInProcessResult,
    IOManager,
    MetadataValue,
    ResourceParam,
    SourceAsset,
    asset,
    asset_check,
    define_asset_job,
)
from dagster._core.definitions.asset_check_spec import AssetCheckKey
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvariantViolationError


def execute_assets_and_checks(
    assets=None, asset_checks=None, raise_on_error: bool = True, resources=None, instance=None
) -> ExecuteInProcessResult:
    defs = Definitions(assets=assets, asset_checks=asset_checks, resources=resources)
    job_def = defs.get_implicit_global_asset_job_def()
    return job_def.execute_in_process(raise_on_error=raise_on_error, instance=instance)


def test_asset_check_decorator():
    @asset_check(asset="asset1", description="desc")
    def check1():
        ...

    assert check1.name == "check1"
    assert check1.description == "desc"
    assert check1.asset_key == AssetKey("asset1")


def test_asset_check_decorator_name():
    @asset_check(asset="asset1", description="desc", name="check1")
    def _check():
        ...

    assert _check.name == "check1"


def test_asset_check_with_prefix():
    @asset(key_prefix="prefix")
    def asset1():
        ...

    @asset_check(asset=asset1)
    def my_check():
        ...

    assert my_check.asset_key == AssetKey(["prefix", "asset1"])


def test_asset_check_input_with_prefix():
    @asset(key_prefix="prefix")
    def asset1():
        ...

    @asset_check(asset=asset1)
    def my_check(asset1):
        ...

    assert my_check.asset_key == AssetKey(["prefix", "asset1"])


def test_execute_asset_and_check():
    @asset
    def asset1():
        ...

    @asset_check(asset=asset1, description="desc")
    def check1(context: AssetExecutionContext):
        assert context.asset_key_for_input("asset1") == asset1.key
        asset_check_spec = context.asset_check_spec
        return AssetCheckResult(
            asset_key=asset_check_spec.asset_key,
            check_name=asset_check_spec.name,
            passed=True,
            metadata={"foo": "bar"},
        )

    instance = DagsterInstance.ephemeral()
    result = execute_assets_and_checks(assets=[asset1], asset_checks=[check1], instance=instance)
    assert result.success

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == asset1.key
    assert check_eval.check_name == "check1"
    assert check_eval.metadata == {"foo": MetadataValue.text("bar")}

    assert check_eval.target_materialization_data is not None
    assert check_eval.target_materialization_data.run_id == result.run_id
    materialization_record = instance.get_event_records(
        EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION)
    )[0]
    assert check_eval.target_materialization_data.storage_id == materialization_record.storage_id
    assert check_eval.target_materialization_data.timestamp == materialization_record.timestamp

    assert (
        len(
            instance.get_event_records(
                EventRecordsFilter(event_type=DagsterEventType.ASSET_CHECK_EVALUATION_PLANNED)
            )
        )
        == 1
    )
    assert (
        len(
            instance.get_event_records(
                EventRecordsFilter(event_type=DagsterEventType.ASSET_CHECK_EVALUATION)
            )
        )
        == 1
    )
    assert (
        len(
            instance.event_log_storage.get_asset_check_execution_history(
                AssetCheckKey(asset_key=AssetKey("asset1"), name="check1"), limit=10
            )
        )
        == 1
    )


def test_execute_check_without_asset():
    @asset_check(asset="asset1", description="desc")
    def check1():
        return AssetCheckResult(passed=True, metadata={"foo": "bar"})

    result = execute_assets_and_checks(asset_checks=[check1])
    assert result.success

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey("asset1")
    assert check_eval.check_name == "check1"
    assert check_eval.metadata == {"foo": MetadataValue.text("bar")}

    assert check_eval.target_materialization_data is None


def test_execute_check_and_asset_in_separate_run():
    @asset
    def asset1():
        ...

    @asset_check(asset=asset1, description="desc")
    def check1(context: AssetExecutionContext):
        assert context.asset_key_for_input("asset1") == asset1.key
        asset_check_spec = context.asset_check_spec
        return AssetCheckResult(
            asset_key=asset_check_spec.asset_key,
            check_name=asset_check_spec.name,
            passed=True,
            metadata={"foo": "bar"},
        )

    instance = DagsterInstance.ephemeral()
    materialize_result = execute_assets_and_checks(assets=[asset1], instance=instance)

    result = execute_assets_and_checks(asset_checks=[check1], instance=instance)
    assert result.success

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]

    assert check_eval.target_materialization_data is not None
    assert check_eval.target_materialization_data.run_id == materialize_result.run_id
    materialization_record = instance.get_event_records(
        EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION)
    )[0]
    assert check_eval.target_materialization_data.storage_id == materialization_record.storage_id
    assert check_eval.target_materialization_data.timestamp == materialization_record.timestamp


def test_execute_check_and_unrelated_asset():
    @asset
    def asset2():
        ...

    @asset_check(asset="asset1", description="desc")
    def check1():
        return AssetCheckResult(passed=True)

    result = execute_assets_and_checks(assets=[asset2], asset_checks=[check1])
    assert result.success

    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 1

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey("asset1")
    assert check_eval.check_name == "check1"


def test_check_doesnt_execute_if_asset_fails():
    check_executed = [False]

    @asset
    def asset1():
        raise ValueError()

    @asset_check(asset=asset1)
    def asset1_check(context: AssetExecutionContext):
        check_executed[0] = True

    result = execute_assets_and_checks(
        assets=[asset1], asset_checks=[asset1_check], raise_on_error=False
    )
    assert not result.success

    assert not check_executed[0]


def test_check_decorator_unexpected_asset_key():
    @asset_check(asset="asset1", description="desc")
    def asset1_check():
        return AssetCheckResult(asset_key=AssetKey("asset2"), passed=True)

    with pytest.raises(
        DagsterInvariantViolationError,
        match=re.escape(
            "Received unexpected AssetCheckResult. It targets asset 'asset2' which is not targeted"
            " by any of the checks currently being evaluated. Targeted assets: ['asset1']."
        ),
    ):
        execute_assets_and_checks(asset_checks=[asset1_check])


def test_asset_check_separate_op_downstream_still_executes():
    @asset
    def asset1():
        ...

    @asset_check(asset=asset1)
    def asset1_check(context: AssetExecutionContext):
        return AssetCheckResult(passed=False)

    @asset(deps=[asset1])
    def asset2():
        ...

    result = execute_assets_and_checks(assets=[asset1, asset2], asset_checks=[asset1_check])
    assert result.success

    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 2

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey("asset1")
    assert check_eval.check_name == "asset1_check"
    assert not check_eval.passed


def test_error_severity_skip_downstream():
    pytest.skip("Currently users should raise exceptions instead of using checks for control flow.")

    @asset
    def asset1():
        ...

    @asset_check(asset=asset1, severity=AssetCheckSeverity.ERROR)
    def check1(context: AssetExecutionContext):
        return AssetCheckResult(passed=False)

    @asset(deps=[asset1])
    def asset2():
        ...

    result = execute_assets_and_checks(
        assets=[asset1, asset2], asset_checks=[check1], raise_on_error=False
    )
    assert not result.success

    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 1

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey("asset1")
    assert check_eval.check_name == "check1"
    assert not check_eval.success

    error = result.failure_data_for_node("asset1_check1").error
    assert error.message.startswith(
        "dagster._core.errors.DagsterAssetCheckFailedError: Check 'check1' for asset 'asset1'"
        " failed with ERROR severity."
    )


def test_error_severity_with_source_asset_fail():
    pytest.skip("Currently users should raise exceptions instead of using checks for control flow.")

    asset1 = SourceAsset("asset1")

    @asset_check(asset=asset1, severity=AssetCheckSeverity.ERROR)
    def check1(context: AssetExecutionContext):
        return AssetCheckResult(passed=False)

    @asset(deps=[asset1])
    def asset2():
        ...

    result = execute_assets_and_checks(
        assets=[asset1, asset2], asset_checks=[check1], raise_on_error=False
    )
    assert not result.success

    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 0

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey("asset1")
    assert check_eval.check_name == "check1"
    assert not check_eval.success

    error = result.failure_data_for_node("asset1_check1").error
    assert error.message.startswith(
        "dagster._core.errors.DagsterAssetCheckFailedError: Check 'check1' for asset 'asset1'"
        " failed with ERROR severity."
    )


def test_error_severity_with_source_asset_success():
    asset1 = SourceAsset("asset1", io_manager_key="asset1_io_manager")

    @asset_check(asset=asset1)
    def check1(context: AssetExecutionContext):
        return AssetCheckResult(passed=True, severity=AssetCheckSeverity.ERROR)

    @asset
    def asset2(asset1):
        assert asset1 == 5

    class MyIOManager(IOManager):
        def load_input(self, context):
            return 5

        def handle_output(self, context, obj):
            raise NotImplementedError()

    result = execute_assets_and_checks(
        assets=[asset1, asset2],
        asset_checks=[check1],
        raise_on_error=False,
        resources={"asset1_io_manager": MyIOManager()},
    )
    assert result.success

    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 1

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey("asset1")
    assert check_eval.check_name == "check1"
    assert check_eval.passed


def test_definitions_conflicting_checks():
    def make_check():
        @asset_check(asset="asset1")
        def check1(context: AssetExecutionContext):
            ...

        return check1

    with pytest.raises(
        DagsterInvalidDefinitionError,
        match='Detected conflicting node definitions with the same name "asset1_check1"',
    ):
        Definitions(asset_checks=[make_check(), make_check()])


def test_definitions_same_name_different_asset():
    def make_check_for_asset(asset_key: str):
        @asset_check(asset=asset_key)
        def check1(context: AssetExecutionContext):
            ...

        return check1

    Definitions(asset_checks=[make_check_for_asset("asset1"), make_check_for_asset("asset2")])


def test_definitions_same_asset_different_name():
    def make_check(check_name: str):
        @asset_check(asset="asset1", name=check_name)
        def _check(context: AssetExecutionContext):
            ...

        return _check

    Definitions(asset_checks=[make_check("check1"), make_check("check2")])


def test_resource_params():
    class MyResource(NamedTuple):
        value: int

    @asset_check(asset=AssetKey("asset1"))
    def check1(my_resource: ResourceParam[MyResource]):
        assert my_resource.value == 5
        return AssetCheckResult(passed=True)

    execute_assets_and_checks(asset_checks=[check1], resources={"my_resource": MyResource(5)})


def test_job_only_execute_checks_downstream_of_selected_assets():
    @asset
    def asset1():
        ...

    @asset
    def asset2():
        ...

    @asset_check(asset=asset1)
    def check1():
        return AssetCheckResult(passed=False)

    @asset_check(asset=asset2)
    def check2():
        return AssetCheckResult(passed=False)

    defs = Definitions(
        assets=[asset1, asset2],
        asset_checks=[check1, check2],
        jobs=[define_asset_job("job1", selection=[asset1])],
    )
    job_def = defs.get_job_def("job1")
    result = job_def.execute_in_process()
    assert result.success

    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == asset1.key
    assert check_eval.check_name == "check1"


def test_asset_not_provided():
    with pytest.raises(Exception):

        @asset_check(description="desc")
        def check1():
            ...


def test_managed_input():
    @asset
    def asset1() -> int:
        return 4

    @asset_check(asset=asset1, description="desc")
    def check1(asset1):
        assert asset1 == 4
        return AssetCheckResult(passed=True)

    class MyIOManager(IOManager):
        def load_input(self, context):
            assert context.asset_key == asset1.key
            return 4

        def handle_output(self, context, obj):
            ...

    assert check1.name == "check1"
    assert check1.asset_key == asset1.key

    assert execute_assets_and_checks(
        assets=[asset1], asset_checks=[check1], resources={"io_manager": MyIOManager()}
    ).success


def test_multiple_managed_inputs():
    with pytest.raises(
        DagsterInvalidDefinitionError,
        match=re.escape(
            "When defining check 'check1', multiple target assets provided as parameters:"
            " ['asset1', 'asset2']. Only one is allowed."
        ),
    ):

        @asset_check(asset="asset1", description="desc")
        def check1(asset1, asset2):
            ...


def test_managed_input_with_context():
    @asset
    def asset1() -> int:
        return 4

    @asset_check(asset=asset1, description="desc")
    def check1(context: AssetExecutionContext, asset1):
        assert context
        assert asset1 == 4
        return AssetCheckResult(passed=True)

    assert check1.name == "check1"
    assert check1.asset_key == asset1.key

    execute_assets_and_checks(assets=[asset1], asset_checks=[check1])


def test_doesnt_invoke_io_manager():
    class DummyIOManager(IOManager):
        def handle_output(self, context, obj):
            assert False

        def load_input(self, context):
            assert False

    @asset_check(asset="asset1", description="desc")
    def check1(context: AssetExecutionContext):
        return AssetCheckResult(passed=True)

    execute_assets_and_checks(asset_checks=[check1], resources={"io_manager": DummyIOManager()})
