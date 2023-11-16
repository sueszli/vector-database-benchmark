from dagster import AssetIn, AssetKey, AssetOut, asset, multi_asset
from dagster._core.definitions.resolved_asset_deps import resolve_assets_def_deps

def test_same_name_twice_and_downstream():
    if False:
        return 10

    @asset(name='apple', key_prefix='a')
    def asset1():
        if False:
            i = 10
            return i + 15
        ...

    @asset(name='apple', key_prefix='b')
    def asset2():
        if False:
            i = 10
            return i + 15
        ...

    @asset(ins={'apple': AssetIn(key_prefix='a')})
    def asset3(apple):
        if False:
            while True:
                i = 10
        del apple
    assert len(resolve_assets_def_deps([asset1, asset2, asset3], [])) == 0

def test_multi_asset_group_name():
    if False:
        for i in range(10):
            print('nop')

    @asset(group_name='somegroup', key_prefix=['some', 'path'])
    def upstream():
        if False:
            return 10
        pass

    @multi_asset(group_name='somegroup', outs={'a': AssetOut(), 'b': AssetOut()})
    def multi_downstream(upstream):
        if False:
            return 10
        pass
    resolved = resolve_assets_def_deps([upstream, multi_downstream], [])
    assert len(resolved) == 1
    resolution = next(iter(resolved.values()))
    assert resolution == {AssetKey(['upstream']): AssetKey(['some', 'path', 'upstream'])}

def test_input_has_asset_key():
    if False:
        print('Hello World!')

    @asset(key_prefix='a')
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        ...

    @asset(deps=[AssetKey(['b', 'asset1'])])
    def asset2():
        if False:
            return 10
        ...
    assert len(resolve_assets_def_deps([asset1, asset2], [])) == 0

def test_upstream_same_name_as_asset():
    if False:
        print('Hello World!')

    @asset(deps=[AssetKey('asset1')], key_prefix='b')
    def asset1():
        if False:
            while True:
                i = 10
        ...
    assert len(resolve_assets_def_deps([asset1], [])) == 0

    @multi_asset(outs={'asset1': AssetOut(key_prefix='b')}, deps=[AssetKey(['a', 'asset1'])])
    def multi_asset1():
        if False:
            while True:
                i = 10
        ...
    assert len(resolve_assets_def_deps([multi_asset1], [])) == 0