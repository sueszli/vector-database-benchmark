from typing import Iterator
from dagster import AssetMaterialization, Output, asset, job, materialize, op
from dagster._core.definitions.events import AssetKey, CoercibleToAssetKey
from dagster._core.definitions.metadata import TextMetadataValue
from dagster._core.execution.context.compute import AssetExecutionContext, OpExecutionContext
from dagster._core.instance import DagsterInstance

def assert_latest_mat_metadata_entry(instance: DagsterInstance, asset_key: CoercibleToAssetKey, key: str, value: str):
    if False:
        while True:
            i = 10
    mat_event = instance.get_latest_materialization_event(AssetKey.from_coercible(asset_key))
    assert mat_event
    assert mat_event.asset_materialization
    assert mat_event.asset_materialization.metadata[key] == TextMetadataValue(value)

def test_op_that_yields() -> None:
    if False:
        for i in range(10):
            print('nop')
    ran = {}

    @op
    def op_that_yields(context: OpExecutionContext) -> Iterator:
        if False:
            while True:
                i = 10
        asset_key = AssetKey('some_asset_key')
        yield AssetMaterialization(asset_key=asset_key, metadata={'when': 'during_run'})
        assert_latest_mat_metadata_entry(context.instance, asset_key, 'when', 'during_run')
        ran['yup'] = True
        yield Output(value=None, metadata={'when': 'after_run'})

    @job
    def job_that_yields() -> None:
        if False:
            return 10
        op_that_yields()
    instance = DagsterInstance.ephemeral()
    assert job_that_yields.execute_in_process(instance=instance).success
    assert ran['yup']

def test_op_that_logs_event() -> None:
    if False:
        print('Hello World!')
    ran = {}

    @op
    def op_that_yields(context: OpExecutionContext) -> Iterator:
        if False:
            while True:
                i = 10
        asset_key = AssetKey('some_asset_key')
        context.log_event(AssetMaterialization(asset_key=asset_key, metadata={'when': 'during_run'}))
        assert_latest_mat_metadata_entry(context.instance, asset_key, 'when', 'during_run')
        ran['yup'] = True
        yield Output(value=None, metadata={'when': 'after_run'})

    @job
    def job_that_yields() -> None:
        if False:
            return 10
        op_that_yields()
    instance = DagsterInstance.ephemeral()
    assert job_that_yields.execute_in_process(instance=instance).success
    assert ran['yup']

def test_op_that_logs_event_with_implicit_yield() -> None:
    if False:
        i = 10
        return i + 15
    ran = {}

    @op
    def op_that_yields(context: OpExecutionContext) -> None:
        if False:
            for i in range(10):
                print('nop')
        asset_key = AssetKey('some_asset_key')
        context.log_event(AssetMaterialization(asset_key=asset_key, metadata={'when': 'during_run'}))
        assert_latest_mat_metadata_entry(context.instance, asset_key, 'when', 'during_run')
        ran['yup'] = True

    @job
    def job_that_yields() -> None:
        if False:
            return 10
        op_that_yields()
    instance = DagsterInstance.ephemeral()
    assert job_that_yields.execute_in_process(instance=instance).success
    assert ran['yup']

def test_asset_that_logs_on_other_asset() -> None:
    if False:
        i = 10
        return i + 15
    ran = {}
    logs_other_asset_key = AssetKey('logs_other_asset')

    @asset(key=logs_other_asset_key)
    def asset_that_logs_mat_on_other_asset(context: AssetExecutionContext) -> None:
        if False:
            print('Hello World!')
        context.log_event(AssetMaterialization(asset_key='other_asset', metadata={'when': 'during_run'}))
        assert_latest_mat_metadata_entry(context.instance, 'other_asset', 'when', 'during_run')
        context.add_output_metadata({'when': 'after_run'})
        assert context.instance.get_latest_materialization_event(logs_other_asset_key) is None
        ran['yup'] = True
    instance = DagsterInstance.ephemeral()
    assert materialize([asset_that_logs_mat_on_other_asset], instance=instance).success
    assert ran['yup']
    assert_latest_mat_metadata_entry(instance, logs_other_asset_key, 'when', 'after_run')

def test_asset_that_logs_on_itself() -> None:
    if False:
        while True:
            i = 10
    ran = {}
    logs_itself_key = AssetKey('logs_itself')

    @asset(key=logs_itself_key)
    def asset_that_logs_mat_on_other_asset(context: AssetExecutionContext) -> None:
        if False:
            while True:
                i = 10
        context.log_event(AssetMaterialization(asset_key=logs_itself_key, metadata={'when': 'during_run'}))
        assert_latest_mat_metadata_entry(context.instance, logs_itself_key, 'when', 'during_run')
        context.add_output_metadata({'when': 'after_run'})
        assert_latest_mat_metadata_entry(context.instance, logs_itself_key, 'when', 'during_run')
        ran['yup'] = True
    instance = DagsterInstance.ephemeral()
    assert materialize([asset_that_logs_mat_on_other_asset], instance=instance).success
    assert ran['yup']
    assert_latest_mat_metadata_entry(instance, logs_itself_key, 'when', 'after_run')