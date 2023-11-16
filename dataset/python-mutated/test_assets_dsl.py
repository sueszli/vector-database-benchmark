from typing import List
import yaml
from assets_yaml_dsl.pure_assets_dsl.assets_dsl import from_asset_entries
from dagster import AssetsDefinition
from dagster._core.definitions.events import AssetKey
from dagster._core.execution.context.invocation import build_asset_context
from dagster._core.pipes.subprocess import PipesSubprocessClient

def assets_defs_from_yaml(yaml_string) -> List[AssetsDefinition]:
    if False:
        for i in range(10):
            print('nop')
    return from_asset_entries(yaml.safe_load(yaml_string))

def test_basic() -> None:
    if False:
        print('Hello World!')
    assets_defs = assets_defs_from_yaml('\nassets:\n    - asset_key: asset_one\n      sql: "SELECT * from asset_one"\n')
    assert assets_defs
    assert len(assets_defs) == 1
    assets_def = assets_defs[0]
    assert assets_def.key == AssetKey('asset_one')
    assert len(assets_def.keys) == 1

def test_single_dep() -> None:
    if False:
        i = 10
        return i + 15
    assets_defs = assets_defs_from_yaml('\nassets:\n    - asset_key: key_ns/asset_one\n      sql: "SELECT * from asset_one"\n    - asset_key: key_ns/asset_two\n      deps:\n        - key_ns/asset_one\n      sql: "SELECT * from asset_two"\n')
    assert assets_defs
    assert len(assets_defs) == 2
    asset_one = assets_defs[0]
    asset_two = assets_defs[1]
    assert asset_one.key == AssetKey.from_user_string('key_ns/asset_one')
    assert asset_two.key == AssetKey.from_user_string('key_ns/asset_two')
    assert asset_two.asset_deps[asset_two.key] == {asset_one.key}

def test_description() -> None:
    if False:
        while True:
            i = 10
    assets_defs = assets_defs_from_yaml('\nassets:\n    - asset_key: asset_one\n      description: "asset one description"\n      sql: "SELECT * from asset_one"\n')
    assert assets_defs
    assert len(assets_defs) == 1
    assets_def = assets_defs[0]
    assert assets_def.key == AssetKey('asset_one')
    assert assets_def.descriptions_by_key[assets_def.key] == 'asset one description'

def test_execution() -> None:
    if False:
        for i in range(10):
            print('nop')
    assets_defs = assets_defs_from_yaml('\nassets:\n    - asset_key: asset_one\n      sql: "SELECT * from asset_one"\n')
    assert assets_defs
    assert len(assets_defs) == 1
    assets_def = assets_defs[0]
    assets_def(context=build_asset_context(), pipes_subprocess_client=PipesSubprocessClient())

def test_basic_group() -> None:
    if False:
        while True:
            i = 10
    assets_defs = assets_defs_from_yaml('\ngroup_name: my_group\nassets:\n    - asset_key: asset_one\n      sql: "SELECT * from asset_one"\n')
    assert assets_defs
    assert len(assets_defs) == 1
    assets_def = assets_defs[0]
    assert assets_def.group_names_by_key[assets_def.key] == 'my_group'