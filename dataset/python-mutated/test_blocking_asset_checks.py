from dagster import AssetCheckResult, AssetExecutionContext, AssetKey, Definitions, ExecuteInProcessResult, asset, asset_check
from dagster._core.definitions.asset_checks import build_asset_with_blocking_check
from dagster._core.definitions.asset_in import AssetIn

def execute_assets_and_checks(assets=None, asset_checks=None, raise_on_error: bool=True, resources=None, instance=None, tags=None) -> ExecuteInProcessResult:
    if False:
        print('Hello World!')
    defs = Definitions(assets=assets, asset_checks=asset_checks, resources=resources)
    job_def = defs.get_implicit_global_asset_job_def()
    return job_def.execute_in_process(raise_on_error=raise_on_error, instance=instance, tags=tags)

@asset
def upstream_asset():
    if False:
        return 10
    return 'foo'

@asset(deps=[upstream_asset])
def my_asset():
    if False:
        i = 10
        return i + 15
    pass

@asset_check(asset='my_asset')
def pass_check():
    if False:
        while True:
            i = 10
    return AssetCheckResult(passed=True, check_name='pass_check')

@asset_check(asset='my_asset')
def fail_check_if_tagged(context: AssetExecutionContext):
    if False:
        i = 10
        return i + 15
    return AssetCheckResult(passed=not context.has_tag('fail_check'), check_name='fail_check_if_tagged')
blocking_asset = build_asset_with_blocking_check(asset_def=my_asset, checks=[pass_check, fail_check_if_tagged])

@asset(deps=[blocking_asset])
def downstream_asset():
    if False:
        print('Hello World!')
    pass

def test_check_pass():
    if False:
        print('Hello World!')
    result = execute_assets_and_checks(assets=[upstream_asset, blocking_asset, downstream_asset], raise_on_error=False)
    assert result.success
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 2
    check_evals_by_name = {check_eval.check_name: check_eval for check_eval in check_evals}
    assert check_evals_by_name['pass_check'].passed
    assert check_evals_by_name['pass_check'].asset_key == AssetKey(['my_asset'])
    assert check_evals_by_name['fail_check_if_tagged'].passed
    assert check_evals_by_name['fail_check_if_tagged'].asset_key == AssetKey(['my_asset'])
    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 3
    assert materialization_events[0].asset_key == AssetKey(['upstream_asset'])
    assert materialization_events[1].asset_key == AssetKey(['my_asset'])
    assert materialization_events[2].asset_key == AssetKey(['downstream_asset'])

def test_check_fail_and_block():
    if False:
        i = 10
        return i + 15
    result = execute_assets_and_checks(assets=[upstream_asset, blocking_asset, downstream_asset], raise_on_error=False, tags={'fail_check': 'true'})
    assert not result.success
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 2
    check_evals_by_name = {check_eval.check_name: check_eval for check_eval in check_evals}
    assert check_evals_by_name['pass_check'].passed
    assert check_evals_by_name['pass_check'].asset_key == AssetKey(['my_asset'])
    assert not check_evals_by_name['fail_check_if_tagged'].passed
    assert check_evals_by_name['fail_check_if_tagged'].asset_key == AssetKey(['my_asset'])
    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 2
    assert materialization_events[0].asset_key == AssetKey(['upstream_asset'])
    assert materialization_events[1].asset_key == AssetKey(['my_asset'])

@asset
def my_asset_with_managed_input(upstream_asset):
    if False:
        i = 10
        return i + 15
    assert upstream_asset == 'foo'
    return 'bar'

@asset_check(asset='my_asset_with_managed_input')
def fail_check_if_tagged_2(context: AssetExecutionContext, my_asset_with_managed_input):
    if False:
        while True:
            i = 10
    assert my_asset_with_managed_input == 'bar'
    return AssetCheckResult(passed=not context.has_tag('fail_check'), check_name='fail_check_if_tagged_2')
blocking_asset_with_managed_input = build_asset_with_blocking_check(asset_def=my_asset_with_managed_input, checks=[fail_check_if_tagged_2])

@asset(ins={'input_asset': AssetIn(blocking_asset_with_managed_input.key)})
def downstream_asset_2(input_asset):
    if False:
        while True:
            i = 10
    assert input_asset == 'bar'

def test_check_pass_with_inputs():
    if False:
        for i in range(10):
            print('nop')
    result = execute_assets_and_checks(assets=[upstream_asset, blocking_asset_with_managed_input, downstream_asset_2], raise_on_error=False)
    assert result.success
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_evals_by_name = {check_eval.check_name: check_eval for check_eval in check_evals}
    assert check_evals_by_name['fail_check_if_tagged_2'].passed
    assert check_evals_by_name['fail_check_if_tagged_2'].asset_key == AssetKey(['my_asset_with_managed_input'])
    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 3
    assert materialization_events[0].asset_key == AssetKey(['upstream_asset'])
    assert materialization_events[1].asset_key == AssetKey(['my_asset_with_managed_input'])
    assert materialization_events[2].asset_key == AssetKey(['downstream_asset_2'])

def test_check_fail_and_block_with_inputs():
    if False:
        for i in range(10):
            print('nop')
    result = execute_assets_and_checks(assets=[upstream_asset, blocking_asset_with_managed_input, downstream_asset_2], raise_on_error=False, tags={'fail_check': 'true'})
    assert not result.success
    check_evals = result.get_asset_check_evaluations()
    assert len(check_evals) == 1
    check_evals_by_name = {check_eval.check_name: check_eval for check_eval in check_evals}
    assert not check_evals_by_name['fail_check_if_tagged_2'].passed
    assert check_evals_by_name['fail_check_if_tagged_2'].asset_key == AssetKey(['my_asset_with_managed_input'])
    materialization_events = result.get_asset_materialization_events()
    assert len(materialization_events) == 2
    assert materialization_events[0].asset_key == AssetKey(['upstream_asset'])
    assert materialization_events[1].asset_key == AssetKey(['my_asset_with_managed_input'])

@asset
def asset1():
    if False:
        for i in range(10):
            print('nop')
    return 'asset1'

@asset
def asset2():
    if False:
        i = 10
        return i + 15
    return 'asset2'

@asset_check(asset='asset1')
def check1():
    if False:
        print('Hello World!')
    return AssetCheckResult(passed=True)

@asset_check(asset='asset2')
def check2():
    if False:
        while True:
            i = 10
    return AssetCheckResult(passed=True)
blocking_asset_1 = build_asset_with_blocking_check(asset_def=asset1, checks=[check1])
blocking_asset_2 = build_asset_with_blocking_check(asset_def=asset2, checks=[check2])

def test_multiple_blocking_assets():
    if False:
        print('Hello World!')
    result = execute_assets_and_checks(assets=[blocking_asset_1, blocking_asset_2], raise_on_error=False)
    assert result.success