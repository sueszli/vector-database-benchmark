import pytest
from dagster import AssetKey, asset_check, load_assets_from_modules
from dagster._core.definitions.load_asset_checks_from_modules import (
    load_asset_checks_from_current_module,
    load_asset_checks_from_modules,
    load_asset_checks_from_package_module,
    load_asset_checks_from_package_name,
)

from dagster_tests.definitions_tests.decorators_tests.test_asset_check_decorator import (
    execute_assets_and_checks,
)


def test_load_asset_checks_from_modules():
    from . import checks_module
    from .checks_module import asset_check_1

    checks = load_asset_checks_from_modules([checks_module])
    assert len(checks) == 1

    assert checks[0].spec.asset_key == asset_check_1.asset_key
    assert checks[0].spec.name == asset_check_1.name

    result = execute_assets_and_checks(
        asset_checks=checks, assets=load_assets_from_modules([checks_module])
    )
    assert result.success

    assert len(result.get_asset_check_evaluations()) == 1
    assert result.get_asset_check_evaluations()[0].passed
    assert result.get_asset_check_evaluations()[0].asset_key == asset_check_1.asset_key
    assert result.get_asset_check_evaluations()[0].check_name == "asset_check_1"


def test_load_asset_checks_from_modules_prefix():
    from . import checks_module
    from .checks_module import asset_check_1

    checks = load_asset_checks_from_modules([checks_module], asset_key_prefix="foo")
    assert len(checks) == 1

    assert checks[0].spec.asset_key == AssetKey(["foo", "asset_1"])
    assert checks[0].spec.name == asset_check_1.name

    result = execute_assets_and_checks(
        asset_checks=checks, assets=load_assets_from_modules([checks_module], key_prefix="foo")
    )
    assert result.success

    assert len(result.get_asset_check_evaluations()) == 1
    assert result.get_asset_check_evaluations()[0].passed
    assert result.get_asset_check_evaluations()[0].asset_key == AssetKey(["foo", "asset_1"])
    assert result.get_asset_check_evaluations()[0].check_name == "asset_check_1"


@asset_check(asset=AssetKey("asset_1"))
def check_in_current_module():
    pass


def test_load_asset_checks_from_current_module():
    checks = load_asset_checks_from_current_module(asset_key_prefix="foo")
    assert len(checks) == 1
    assert checks[0].name == "check_in_current_module"
    assert checks[0].asset_key == AssetKey(["foo", "asset_1"])


@pytest.mark.parametrize(
    "load_fn",
    [
        load_asset_checks_from_package_module,
        lambda package, **kwargs: load_asset_checks_from_package_name(package.__name__, **kwargs),
    ],
)
def test_load_asset_checks_from_package(load_fn):
    from . import checks_module

    checks = load_fn(checks_module, asset_key_prefix="foo")
    assert len(checks) == 2
    assert checks[0].name == "asset_check_1"
    assert checks[0].asset_key == AssetKey(["foo", "asset_1"])
    assert checks[1].name == "submodule_check"
    assert checks[1].asset_key == AssetKey(["foo", "asset_1"])
