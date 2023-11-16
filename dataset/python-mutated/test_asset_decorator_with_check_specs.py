import re
import pytest
from dagster import AssetCheckResult, AssetCheckSeverity, AssetCheckSpec, AssetKey, AssetOut, DagsterEventType, DagsterInstance, EventRecordsFilter, In, MetadataValue, Output, asset, graph_asset, graph_multi_asset, materialize, multi_asset, op
from dagster._core.definitions.asset_check_spec import AssetCheckKey
from dagster._core.definitions.asset_selection import AssetChecksForHandles, AssetSelection
from dagster._core.definitions.asset_spec import AssetSpec
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvariantViolationError, DagsterStepOutputNotFoundError
from dagster._core.execution.context.compute import AssetExecutionContext
from dagster._core.storage.asset_check_execution_record import AssetCheckExecutionRecordStatus
from dagster._core.storage.io_manager import IOManager
from dagster._core.test_utils import instance_for_test
from dagster._core.types.dagster_type import Nothing

def test_asset_check_same_op():
    if False:
        return 10

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1', description='desc')])
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        yield Output(None)
        yield AssetCheckResult(check_name='check1', passed=True, metadata={'foo': 'bar'})
    instance = DagsterInstance.ephemeral()
    result = materialize(assets=[asset1], instance=instance)
    assert result.success
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == asset1.key
    assert check_eval.check_name == 'check1'
    assert check_eval.metadata == {'foo': MetadataValue.text('bar')}
    assert check_eval.target_materialization_data is not None
    assert check_eval.target_materialization_data.run_id == result.run_id
    materialization_record = instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION))[0]
    assert check_eval.target_materialization_data.storage_id == materialization_record.storage_id
    assert check_eval.target_materialization_data.timestamp == materialization_record.timestamp

def test_asset_check_same_op_with_key_prefix():
    if False:
        print('Hello World!')

    @asset(key_prefix='my_prefix', check_specs=[AssetCheckSpec('check1', asset=AssetKey(['my_prefix', 'asset1']), description='desc')])
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        yield Output(None)
        yield AssetCheckResult(check_name='check1', passed=True, metadata={'foo': 'bar'})
    instance = DagsterInstance.ephemeral()
    result = materialize(assets=[asset1], instance=instance)
    assert result.success
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == asset1.key
    assert check_eval.check_name == 'check1'
    assert check_eval.metadata == {'foo': MetadataValue.text('bar')}
    assert check_eval.target_materialization_data is not None
    assert check_eval.target_materialization_data.run_id == result.run_id
    materialization_record = instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION))[0]
    assert check_eval.target_materialization_data.storage_id == materialization_record.storage_id
    assert check_eval.target_materialization_data.timestamp == materialization_record.timestamp

def test_multiple_asset_checks_same_op():
    if False:
        while True:
            i = 10

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1', description='desc'), AssetCheckSpec('check2', asset='asset1', description='desc')])
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        yield Output(None)
        yield AssetCheckResult(check_name='check1', passed=True, metadata={'foo': 'bar'})
        yield AssetCheckResult(check_name='check2', passed=False, metadata={'baz': 'bla'})
    result = materialize(assets=[asset1])
    assert result.success
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 2
    assert check_evals[0].asset_key == asset1.key
    assert check_evals[0].check_name == 'check1'
    assert check_evals[0].metadata == {'foo': MetadataValue.text('bar')}
    assert check_evals[1].asset_key == asset1.key
    assert check_evals[1].check_name == 'check2'
    assert check_evals[1].metadata == {'baz': MetadataValue.text('bla')}

def test_no_result_for_check():
    if False:
        i = 10
        return i + 15

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1')])
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        yield Output(None)
    with pytest.raises(DagsterStepOutputNotFoundError, match='Core compute for op "asset1" did not return an output for non-optional output "asset1_check1"'):
        materialize(assets=[asset1])

def test_check_result_but_no_output():
    if False:
        print('Hello World!')

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1')])
    def asset1():
        if False:
            return 10
        yield AssetCheckResult(passed=True)
    with pytest.raises(DagsterStepOutputNotFoundError, match='Core compute for op "asset1" did not return an output for non-optional output "result"'):
        materialize(assets=[asset1])

def test_unexpected_check_name():
    if False:
        while True:
            i = 10

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1', description='desc')])
    def asset1():
        if False:
            i = 10
            return i + 15
        return AssetCheckResult(check_name='check2', passed=True, metadata={'foo': 'bar'})
    with pytest.raises(DagsterInvariantViolationError, match="No checks currently being evaluated target asset 'asset1' and have name 'check2'. Checks being evaluated for this asset: {'check1'}"):
        materialize(assets=[asset1])

def test_asset_decorator_unexpected_asset_key():
    if False:
        for i in range(10):
            print('nop')

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1', description='desc')])
    def asset1():
        if False:
            while True:
                i = 10
        return AssetCheckResult(asset_key=AssetKey('asset2'), check_name='check1', passed=True)
    with pytest.raises(DagsterInvariantViolationError, match=re.escape("Received unexpected AssetCheckResult. It targets asset 'asset2' which is not targeted by any of the checks currently being evaluated. Targeted assets: ['asset1'].")):
        materialize(assets=[asset1])

def test_result_missing_check_name():
    if False:
        i = 10
        return i + 15

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1'), AssetCheckSpec('check2', asset='asset1')])
    def asset1():
        if False:
            while True:
                i = 10
        yield Output(None)
        yield AssetCheckResult(passed=True)
    with pytest.raises(DagsterInvariantViolationError, match=re.escape("AssetCheckResult result didn't specify a check name, but there are multiple checks to choose from for the this asset key:")):
        materialize(assets=[asset1])

def test_asset_check_fails_downstream_still_executes():
    if False:
        i = 10
        return i + 15

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1')])
    def asset1():
        if False:
            return 10
        yield Output(None)
        yield AssetCheckResult(passed=False)

    @asset(deps=[asset1])
    def asset2():
        if False:
            return 10
        ...
    result = materialize(assets=[asset1, asset2])
    assert result.success
    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 2
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey('asset1')
    assert check_eval.check_name == 'check1'
    assert not check_eval.passed

def test_error_severity_skip_downstream():
    if False:
        return 10
    pytest.skip('Currently users should raise exceptions instead of using checks for control flow.')

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1', severity=AssetCheckSeverity.ERROR)])
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        yield Output(5)
        yield AssetCheckResult(passed=False)

    @asset
    def asset2(asset1: int):
        if False:
            for i in range(10):
                print('nop')
        assert asset1 == 5
    result = materialize(assets=[asset1, asset2], raise_on_error=False)
    assert not result.success
    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 1
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey('asset1')
    assert check_eval.check_name == 'check1'
    assert not check_eval.success
    error = result.failure_data_for_node('asset1').error
    assert error.message.startswith("dagster._core.errors.DagsterAssetCheckFailedError: Check 'check1' for asset 'asset1' failed with ERROR severity.")

def test_duplicate_checks_same_asset():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvalidDefinitionError, match=re.escape("Duplicate check specs: {(AssetKey(['asset1']), 'check1'): 2}")):

        @asset(check_specs=[AssetCheckSpec('check1', asset='asset1', description='desc1'), AssetCheckSpec('check1', asset='asset1', description='desc2')])
        def asset1():
            if False:
                print('Hello World!')
            ...

def test_check_wrong_asset():
    if False:
        return 10
    with pytest.raises(DagsterInvalidDefinitionError, match=re.escape('Invalid asset key')):

        @asset(check_specs=[AssetCheckSpec('check1', asset='other_asset', description='desc1')])
        def asset1():
            if False:
                i = 10
                return i + 15
            ...

def test_multi_asset_with_check():
    if False:
        i = 10
        return i + 15

    @multi_asset(outs={'one': AssetOut('asset1'), 'two': AssetOut('asset2')}, check_specs=[AssetCheckSpec('check1', asset=AssetKey(['asset1', 'one']), description='desc')])
    def asset_1_and_2():
        if False:
            for i in range(10):
                print('nop')
        yield Output(None, output_name='one')
        yield Output(None, output_name='two')
        yield AssetCheckResult(check_name='check1', passed=True, metadata={'foo': 'bar'})
    result = materialize(assets=[asset_1_and_2])
    assert result.success
    assert len(result.get_asset_materialization_events()) == 2
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey(['asset1', 'one'])
    assert check_eval.check_name == 'check1'
    assert check_eval.metadata == {'foo': MetadataValue.text('bar')}

def test_multi_asset_no_result_for_check():
    if False:
        return 10

    @multi_asset(outs={'one': AssetOut('asset1'), 'two': AssetOut('asset2')}, check_specs=[AssetCheckSpec('check1', asset=AssetKey(['asset1', 'one']), description='desc')])
    def asset_1_and_2():
        if False:
            while True:
                i = 10
        yield Output(None, output_name='one')
        yield Output(None, output_name='two')
    with pytest.raises(DagsterStepOutputNotFoundError, match='Core compute for op "asset_1_and_2" did not return an output for non-optional output "asset1__one_check1"'):
        materialize(assets=[asset_1_and_2])

def test_result_missing_asset_key():
    if False:
        print('Hello World!')

    @multi_asset(outs={'one': AssetOut(key='asset1'), 'two': AssetOut(key='asset2')}, check_specs=[AssetCheckSpec('check1', asset='asset1'), AssetCheckSpec('check2', asset='asset2')])
    def asset_1_and_2():
        if False:
            i = 10
            return i + 15
        yield Output(None, output_name='one')
        yield Output(None, output_name='two')
        yield AssetCheckResult(passed=True)
    with pytest.raises(DagsterInvariantViolationError, match=re.escape("AssetCheckResult didn't specify an asset key, but there are multiple assets to choose from: ['asset1', 'asset2']")):
        materialize(assets=[asset_1_and_2])

def test_asset_check_doesnt_store_output():
    if False:
        i = 10
        return i + 15
    handle_output_called = 0

    class DummyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            assert context
            assert obj == 'the-only-allowed-output'
            nonlocal handle_output_called
            handle_output_called += 1

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            assert context
            return {}

    @asset(check_specs=[AssetCheckSpec('check1', asset='asset1', description='desc')])
    def asset1():
        if False:
            return 10
        yield Output('the-only-allowed-output')
        yield AssetCheckResult(check_name='check1', passed=True, metadata={'foo': 'bar'})
    instance = DagsterInstance.ephemeral()
    result = materialize(assets=[asset1], instance=instance, resources={'io_manager': DummyIOManager()})
    assert result.success
    assert handle_output_called == 1
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == asset1.key
    assert check_eval.check_name == 'check1'
    assert check_eval.metadata == {'foo': MetadataValue.text('bar')}
    assert check_eval.target_materialization_data is not None
    assert check_eval.target_materialization_data.run_id == result.run_id
    materialization_record = instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION))[0]
    assert check_eval.target_materialization_data.storage_id == materialization_record.storage_id
    assert check_eval.target_materialization_data.timestamp == materialization_record.timestamp

def test_multi_asset_with_check_subset():
    if False:
        i = 10
        return i + 15

    @multi_asset(outs={'one': AssetOut(key='asset1', is_required=False), 'two': AssetOut(key='asset2', is_required=False)}, check_specs=[AssetCheckSpec('check1', asset=AssetKey(['asset1']))], can_subset=True)
    def asset_1_and_2(context: AssetExecutionContext):
        if False:
            for i in range(10):
                print('nop')
        if AssetKey('asset1') in context.selected_asset_keys:
            yield Output(None, output_name='one')
            yield AssetCheckResult(check_name='check1', passed=True)
        if AssetKey('asset2') in context.selected_asset_keys:
            yield Output(None, output_name='two')
    with instance_for_test() as instance:
        result = materialize(assets=[asset_1_and_2], instance=instance)
        assert result.success
        assert len(result.get_asset_materialization_events()) == 2
        check_evals = result.get_asset_check_evaluations()
        assert len(check_evals) == 1
        check_eval = check_evals[0]
        assert check_eval.asset_key == AssetKey(['asset1'])
        assert check_eval.check_name == 'check1'
        assert instance.event_log_storage.get_asset_check_execution_history(AssetCheckKey(AssetKey(['asset1']), name='check1'), limit=1)[0].status == AssetCheckExecutionRecordStatus.SUCCEEDED
    with instance_for_test() as instance:
        result = materialize(assets=[asset_1_and_2], selection=['asset1'], instance=instance)
        assert result.success
        assert len(result.get_asset_materialization_events()) == 1
        check_evals = result.get_asset_check_evaluations()
        assert len(check_evals) == 1
        check_eval = check_evals[0]
        assert check_eval.asset_key == AssetKey(['asset1'])
        assert check_eval.check_name == 'check1'
        assert instance.event_log_storage.get_asset_check_execution_history(check_key=AssetCheckKey(AssetKey(['asset1']), name='check1'), limit=1)[0].status == AssetCheckExecutionRecordStatus.SUCCEEDED
    with instance_for_test() as instance:
        result = materialize(assets=[asset_1_and_2], selection=['asset2'], instance=instance)
        assert result.success
        assert len(result.get_asset_materialization_events()) == 1
        assert not result.get_asset_check_evaluations()
        assert not instance.event_log_storage.get_asset_check_execution_history(check_key=AssetCheckKey(AssetKey(['asset1']), name='check1'), limit=1)

def test_graph_asset():
    if False:
        while True:
            i = 10

    @op
    def create_asset():
        if False:
            print('Hello World!')
        return None

    @op
    def validate_asset(word):
        if False:
            for i in range(10):
                print('nop')
        return AssetCheckResult(check_name='check1', passed=True, metadata={'foo': 'bar'})

    @op
    def non_blocking_validation(word):
        if False:
            i = 10
            return i + 15
        return AssetCheckResult(check_name='check2', passed=True, metadata={'biz': 'buz'})

    @op(ins={'staging_asset': In(Nothing), 'check_result': In(Nothing)})
    def promote_asset():
        if False:
            i = 10
            return i + 15
        return None

    @graph_asset(name='foo', check_specs=[AssetCheckSpec('check1', asset='foo', description='desc'), AssetCheckSpec('check2', asset='foo', description='desc')])
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        staging_asset = create_asset()
        check_result = validate_asset(staging_asset)
        promoted_asset = promote_asset(staging_asset=staging_asset, check_result=check_result)
        return {'result': promoted_asset, 'foo_check1': check_result, 'foo_check2': non_blocking_validation(staging_asset)}
    result = materialize(assets=[asset1])
    assert result.success
    assert len(result.get_asset_materialization_events()) == 1
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 2
    evals_by_name = {check_eval.check_name: check_eval for check_eval in check_evals}
    assert evals_by_name.keys() == {'check1', 'check2'}
    assert evals_by_name['check1'].asset_key == AssetKey('foo')
    assert evals_by_name['check1'].metadata == {'foo': MetadataValue.text('bar')}
    assert evals_by_name['check2'].asset_key == AssetKey('foo')
    assert evals_by_name['check2'].metadata == {'biz': MetadataValue.text('buz')}

def test_graph_multi_asset():
    if False:
        print('Hello World!')

    @op
    def create_asset():
        if False:
            print('Hello World!')
        return None

    @op
    def validate_asset(word):
        if False:
            print('Hello World!')
        return AssetCheckResult(check_name='check1', passed=True, metadata={'foo': 'bar'})

    @op(ins={'staging_asset': In(Nothing), 'check_result': In(Nothing)})
    def promote_asset():
        if False:
            print('Hello World!')
        return None

    @op
    def create_asset_2():
        if False:
            return 10
        return None

    @graph_multi_asset(outs={'asset_one': AssetOut(), 'asset_two': AssetOut()}, check_specs=[AssetCheckSpec('check1', asset='asset_one', description='desc')])
    def asset1():
        if False:
            while True:
                i = 10
        staging_asset = create_asset()
        check_result = validate_asset(staging_asset)
        promoted_asset = promote_asset(staging_asset=staging_asset, check_result=check_result)
        return {'asset_one': promoted_asset, 'asset_one_check1': check_result, 'asset_two': create_asset_2()}
    result = materialize(assets=[asset1])
    assert result.success
    assert len(result.get_asset_materialization_events()) == 2
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_eval = check_evals[0]
    assert check_eval.asset_key == AssetKey('asset_one')
    assert check_eval.check_name == 'check1'
    assert check_eval.metadata == {'foo': MetadataValue.text('bar')}

def test_can_subset_no_selection() -> None:
    if False:
        while True:
            i = 10

    @multi_asset(can_subset=True, specs=[AssetSpec('asset1'), AssetSpec('asset2')], check_specs=[AssetCheckSpec('check1', asset='asset1'), AssetCheckSpec('check2', asset='asset2')])
    def foo(context: AssetExecutionContext):
        if False:
            i = 10
            return i + 15
        assert context.selected_asset_keys == {AssetKey('asset1'), AssetKey('asset2')}
        assert context.selected_asset_check_keys == {AssetCheckKey(AssetKey('asset1'), 'check1'), AssetCheckKey(AssetKey('asset2'), 'check2')}
        yield Output(value=None, output_name='asset1')
        yield AssetCheckResult(asset_key='asset1', check_name='check1', passed=True)
    result = materialize([foo])
    assert len(result.get_asset_materialization_events()) == 1
    [check_eval] = result.get_asset_check_evaluations()
    assert check_eval.asset_key == AssetKey('asset1')
    assert check_eval.check_name == 'check1'

def test_can_subset() -> None:
    if False:
        return 10

    @multi_asset(can_subset=True, specs=[AssetSpec('asset1'), AssetSpec('asset2')], check_specs=[AssetCheckSpec('check1', asset='asset1'), AssetCheckSpec('check2', asset='asset2')])
    def foo(context: AssetExecutionContext):
        if False:
            for i in range(10):
                print('nop')
        assert context.selected_asset_keys == {AssetKey('asset1')}
        assert context.selected_asset_check_keys == {AssetCheckKey(AssetKey('asset1'), 'check1')}
        yield Output(value=None, output_name='asset1')
        yield AssetCheckResult(asset_key='asset1', check_name='check1', passed=True)
    result = materialize([foo], selection=['asset1'])
    assert len(result.get_asset_materialization_events()) == 1
    [check_eval] = result.get_asset_check_evaluations()
    assert check_eval.asset_key == AssetKey('asset1')
    assert check_eval.check_name == 'check1'

def test_can_subset_result_for_unselected_check() -> None:
    if False:
        return 10

    @multi_asset(can_subset=True, specs=[AssetSpec('asset1'), AssetSpec('asset2')], check_specs=[AssetCheckSpec('check1', asset='asset1'), AssetCheckSpec('check2', asset='asset2')])
    def foo(context: AssetExecutionContext):
        if False:
            for i in range(10):
                print('nop')
        assert context.selected_asset_keys == {AssetKey('asset1')}
        assert context.selected_asset_check_keys == {AssetCheckKey(AssetKey('asset1'), 'check1')}
        yield Output(value=None, output_name='asset1')
        yield AssetCheckResult(asset_key='asset1', check_name='check1', passed=True)
        yield AssetCheckResult(asset_key='asset2', check_name='check2', passed=True)
    with pytest.raises(DagsterInvariantViolationError):
        materialize([foo], selection=['asset1'])

def test_can_subset_select_only_asset() -> None:
    if False:
        print('Hello World!')

    @multi_asset(can_subset=True, specs=[AssetSpec('asset1'), AssetSpec('asset2')], check_specs=[AssetCheckSpec('check1', asset='asset1'), AssetCheckSpec('check2', asset='asset2')])
    def foo(context: AssetExecutionContext):
        if False:
            i = 10
            return i + 15
        assert context.selected_asset_keys == {AssetKey('asset1')}
        assert context.selected_asset_check_keys == set()
        yield Output(value=None, output_name='asset1')
    result = materialize([foo], selection=AssetSelection.keys(AssetKey('asset1')) - AssetSelection.checks_for_assets(foo))
    assert len(result.get_asset_materialization_events()) == 1
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 0

def test_can_subset_select_only_check() -> None:
    if False:
        print('Hello World!')

    @multi_asset(can_subset=True, specs=[AssetSpec('asset1'), AssetSpec('asset2')], check_specs=[AssetCheckSpec('check1', asset='asset1'), AssetCheckSpec('check2', asset='asset2')])
    def foo(context: AssetExecutionContext):
        if False:
            while True:
                i = 10
        assert context.selected_asset_keys == set()
        assert context.selected_asset_check_keys == {AssetCheckKey(AssetKey('asset1'), 'check1')}
        yield AssetCheckResult(asset_key='asset1', check_name='check1', passed=True)
    result = materialize([foo], selection=AssetChecksForHandles([AssetCheckKey(asset_key=AssetKey('asset1'), name='check1')]))
    assert len(result.get_asset_materialization_events()) == 0
    [check_eval] = result.get_asset_check_evaluations()
    assert check_eval.asset_key == AssetKey('asset1')
    assert check_eval.check_name == 'check1'