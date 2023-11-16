import inspect
import os
import pkgutil
from importlib import import_module
from types import ModuleType
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union, cast
import dagster._check as check
from dagster._core.definitions.auto_materialize_policy import AutoMaterializePolicy
from dagster._core.definitions.backfill_policy import BackfillPolicy
from dagster._core.definitions.freshness_policy import FreshnessPolicy
from dagster._core.errors import DagsterInvalidDefinitionError
from .assets import AssetsDefinition
from .cacheable_assets import CacheableAssetsDefinition
from .events import AssetKey, CoercibleToAssetKeyPrefix, check_opt_coercible_to_asset_key_prefix_param
from .source_asset import SourceAsset

def find_objects_in_module_of_types(module: ModuleType, types) -> Iterator:
    if False:
        for i in range(10):
            print('nop')
    'Yields objects of the given type(s).'
    for attr in dir(module):
        value = getattr(module, attr)
        if isinstance(value, types):
            yield value
        elif isinstance(value, list) and all((isinstance(el, types) for el in value)):
            yield from value

def assets_from_modules(modules: Iterable[ModuleType], extra_source_assets: Optional[Sequence[SourceAsset]]=None) -> Tuple[Sequence[AssetsDefinition], Sequence[SourceAsset], Sequence[CacheableAssetsDefinition]]:
    if False:
        return 10
    'Constructs three lists, a list of assets, a list of source assets, and a list of cacheable\n    assets from the given modules.\n\n    Args:\n        modules (Iterable[ModuleType]): The Python modules to look for assets inside.\n        extra_source_assets (Optional[Sequence[SourceAsset]]): Source assets to include in the\n            group in addition to the source assets found in the modules.\n\n    Returns:\n        Tuple[Sequence[AssetsDefinition], Sequence[SourceAsset], Sequence[CacheableAssetsDefinition]]]:\n            A tuple containing a list of assets, a list of source assets, and a list of\n            cacheable assets defined in the given modules.\n    '
    asset_ids: Set[int] = set()
    asset_keys: Dict[AssetKey, ModuleType] = dict()
    source_assets: List[SourceAsset] = list(check.opt_sequence_param(extra_source_assets, 'extra_source_assets', of_type=SourceAsset))
    cacheable_assets: List[CacheableAssetsDefinition] = []
    assets: Dict[AssetKey, AssetsDefinition] = {}
    for module in modules:
        for asset in find_objects_in_module_of_types(module, (AssetsDefinition, SourceAsset, CacheableAssetsDefinition)):
            asset = cast(Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition], asset)
            if id(asset) not in asset_ids:
                asset_ids.add(id(asset))
                if isinstance(asset, CacheableAssetsDefinition):
                    cacheable_assets.append(asset)
                else:
                    keys = asset.keys if isinstance(asset, AssetsDefinition) else [asset.key]
                    for key in keys:
                        if key in asset_keys:
                            modules_str = ', '.join(set([asset_keys[key].__name__, module.__name__]))
                            error_str = f'Asset key {key} is defined multiple times. Definitions found in modules: {modules_str}. '
                            if key in assets and isinstance(asset, AssetsDefinition):
                                if assets[key].node_def == asset.node_def:
                                    error_str += 'One possible cause of this bug is a call to with_resources outside of a repository definition, causing a duplicate asset definition.'
                            raise DagsterInvalidDefinitionError(error_str)
                        else:
                            asset_keys[key] = module
                            if isinstance(asset, AssetsDefinition):
                                assets[key] = asset
                    if isinstance(asset, SourceAsset):
                        source_assets.append(asset)
    return (list(set(assets.values())), source_assets, cacheable_assets)

def load_assets_from_modules(modules: Iterable[ModuleType], group_name: Optional[str]=None, key_prefix: Optional[CoercibleToAssetKeyPrefix]=None, *, freshness_policy: Optional[FreshnessPolicy]=None, auto_materialize_policy: Optional[AutoMaterializePolicy]=None, backfill_policy: Optional[BackfillPolicy]=None, source_key_prefix: Optional[CoercibleToAssetKeyPrefix]=None) -> Sequence[Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition]]:
    if False:
        print('Hello World!')
    'Constructs a list of assets and source assets from the given modules.\n\n    Args:\n        modules (Iterable[ModuleType]): The Python modules to look for assets inside.\n        group_name (Optional[str]):\n            Group name to apply to the loaded assets. The returned assets will be copies of the\n            loaded objects, with the group name added.\n        key_prefix (Optional[Union[str, Sequence[str]]]):\n            Prefix to prepend to the keys of the loaded assets. The returned assets will be copies\n            of the loaded objects, with the prefix prepended.\n        freshness_policy (Optional[FreshnessPolicy]): FreshnessPolicy to apply to all the loaded\n            assets.\n        auto_materialize_policy (Optional[AutoMaterializePolicy]): AutoMaterializePolicy to apply\n            to all the loaded assets.\n        backfill_policy (Optional[AutoMaterializePolicy]): BackfillPolicy to apply to all the loaded assets.\n        source_key_prefix (bool): Prefix to prepend to the keys of loaded SourceAssets. The returned\n            assets will be copies of the loaded objects, with the prefix prepended.\n\n    Returns:\n        Sequence[Union[AssetsDefinition, SourceAsset]]:\n            A list containing assets and source assets defined in the given modules.\n    '
    group_name = check.opt_str_param(group_name, 'group_name')
    key_prefix = check_opt_coercible_to_asset_key_prefix_param(key_prefix, 'key_prefix')
    freshness_policy = check.opt_inst_param(freshness_policy, 'freshness_policy', FreshnessPolicy)
    auto_materialize_policy = check.opt_inst_param(auto_materialize_policy, 'auto_materialize_policy', AutoMaterializePolicy)
    backfill_policy = check.opt_inst_param(backfill_policy, 'backfill_policy', BackfillPolicy)
    (assets, source_assets, cacheable_assets) = assets_from_modules(modules)
    return assets_with_attributes(assets, source_assets, cacheable_assets, key_prefix=key_prefix, group_name=group_name, freshness_policy=freshness_policy, auto_materialize_policy=auto_materialize_policy, backfill_policy=backfill_policy, source_key_prefix=source_key_prefix)

def load_assets_from_current_module(group_name: Optional[str]=None, key_prefix: Optional[CoercibleToAssetKeyPrefix]=None, *, freshness_policy: Optional[FreshnessPolicy]=None, auto_materialize_policy: Optional[AutoMaterializePolicy]=None, backfill_policy: Optional[BackfillPolicy]=None, source_key_prefix: Optional[CoercibleToAssetKeyPrefix]=None) -> Sequence[Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition]]:
    if False:
        i = 10
        return i + 15
    'Constructs a list of assets, source assets, and cacheable assets from the module where\n    this function is called.\n\n    Args:\n        group_name (Optional[str]):\n            Group name to apply to the loaded assets. The returned assets will be copies of the\n            loaded objects, with the group name added.\n        key_prefix (Optional[Union[str, Sequence[str]]]):\n            Prefix to prepend to the keys of the loaded assets. The returned assets will be copies\n            of the loaded objects, with the prefix prepended.\n        freshness_policy (Optional[FreshnessPolicy]): FreshnessPolicy to apply to all the loaded\n            assets.\n        auto_materialize_policy (Optional[AutoMaterializePolicy]): AutoMaterializePolicy to apply\n            to all the loaded assets.\n        backfill_policy (Optional[AutoMaterializePolicy]): BackfillPolicy to apply to all the loaded assets.\n        source_key_prefix (bool): Prefix to prepend to the keys of loaded SourceAssets. The returned\n            assets will be copies of the loaded objects, with the prefix prepended.\n\n    Returns:\n        Sequence[Union[AssetsDefinition, SourceAsset, CachableAssetsDefinition]]:\n            A list containing assets, source assets, and cacheable assets defined in the module.\n    '
    caller = inspect.stack()[1]
    module = inspect.getmodule(caller[0])
    if module is None:
        check.failed('Could not find a module for the caller')
    return load_assets_from_modules([module], group_name=group_name, key_prefix=key_prefix, freshness_policy=freshness_policy, auto_materialize_policy=auto_materialize_policy, backfill_policy=backfill_policy, source_key_prefix=source_key_prefix)

def assets_from_package_module(package_module: ModuleType, extra_source_assets: Optional[Sequence[SourceAsset]]=None) -> Tuple[Sequence[AssetsDefinition], Sequence[SourceAsset], Sequence[CacheableAssetsDefinition]]:
    if False:
        print('Hello World!')
    'Constructs three lists, a list of assets, a list of source assets, and a list of cacheable assets\n    from the given package module.\n\n    Args:\n        package_module (ModuleType): The package module to looks for assets inside.\n        extra_source_assets (Optional[Sequence[SourceAsset]]): Source assets to include in the\n            group in addition to the source assets found in the modules.\n\n    Returns:\n        Tuple[Sequence[AssetsDefinition], Sequence[SourceAsset], Sequence[CacheableAssetsDefinition]]:\n            A tuple containing a list of assets, a list of source assets, and a list of cacheable assets\n            defined in the given modules.\n    '
    return assets_from_modules(find_modules_in_package(package_module), extra_source_assets=extra_source_assets)

def load_assets_from_package_module(package_module: ModuleType, group_name: Optional[str]=None, key_prefix: Optional[CoercibleToAssetKeyPrefix]=None, *, freshness_policy: Optional[FreshnessPolicy]=None, auto_materialize_policy: Optional[AutoMaterializePolicy]=None, backfill_policy: Optional[BackfillPolicy]=None, source_key_prefix: Optional[CoercibleToAssetKeyPrefix]=None) -> Sequence[Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition]]:
    if False:
        while True:
            i = 10
    'Constructs a list of assets and source assets that includes all asset\n    definitions, source assets, and cacheable assets in all sub-modules of the given package module.\n\n    A package module is the result of importing a package.\n\n    Args:\n        package_module (ModuleType): The package module to looks for assets inside.\n        group_name (Optional[str]):\n            Group name to apply to the loaded assets. The returned assets will be copies of the\n            loaded objects, with the group name added.\n        key_prefix (Optional[Union[str, Sequence[str]]]):\n            Prefix to prepend to the keys of the loaded assets. The returned assets will be copies\n            of the loaded objects, with the prefix prepended.\n        freshness_policy (Optional[FreshnessPolicy]): FreshnessPolicy to apply to all the loaded\n            assets.\n        auto_materialize_policy (Optional[AutoMaterializePolicy]): AutoMaterializePolicy to apply\n            to all the loaded assets.\n        backfill_policy (Optional[AutoMaterializePolicy]): BackfillPolicy to apply to all the loaded assets.\n        source_key_prefix (bool): Prefix to prepend to the keys of loaded SourceAssets. The returned\n            assets will be copies of the loaded objects, with the prefix prepended.\n\n    Returns:\n        Sequence[Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition]]:\n            A list containing assets, source assets, and cacheable assets defined in the module.\n    '
    group_name = check.opt_str_param(group_name, 'group_name')
    key_prefix = check_opt_coercible_to_asset_key_prefix_param(key_prefix, 'key_prefix')
    freshness_policy = check.opt_inst_param(freshness_policy, 'freshness_policy', FreshnessPolicy)
    auto_materialize_policy = check.opt_inst_param(auto_materialize_policy, 'auto_materialize_policy', AutoMaterializePolicy)
    backfill_policy = check.opt_inst_param(backfill_policy, 'backfill_policy', BackfillPolicy)
    (assets, source_assets, cacheable_assets) = assets_from_package_module(package_module)
    return assets_with_attributes(assets, source_assets, cacheable_assets, key_prefix=key_prefix, group_name=group_name, freshness_policy=freshness_policy, auto_materialize_policy=auto_materialize_policy, backfill_policy=backfill_policy, source_key_prefix=source_key_prefix)

def load_assets_from_package_name(package_name: str, group_name: Optional[str]=None, key_prefix: Optional[CoercibleToAssetKeyPrefix]=None, *, freshness_policy: Optional[FreshnessPolicy]=None, auto_materialize_policy: Optional[AutoMaterializePolicy]=None, backfill_policy: Optional[BackfillPolicy]=None, source_key_prefix: Optional[CoercibleToAssetKeyPrefix]=None) -> Sequence[Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition]]:
    if False:
        print('Hello World!')
    'Constructs a list of assets, source assets, and cacheable assets that includes all asset\n    definitions and source assets in all sub-modules of the given package.\n\n    Args:\n        package_name (str): The name of a Python package to look for assets inside.\n        group_name (Optional[str]):\n            Group name to apply to the loaded assets. The returned assets will be copies of the\n            loaded objects, with the group name added.\n        key_prefix (Optional[Union[str, Sequence[str]]]):\n            Prefix to prepend to the keys of the loaded assets. The returned assets will be copies\n            of the loaded objects, with the prefix prepended.\n        freshness_policy (Optional[FreshnessPolicy]): FreshnessPolicy to apply to all the loaded\n            assets.\n        auto_materialize_policy (Optional[AutoMaterializePolicy]): AutoMaterializePolicy to apply\n            to all the loaded assets.\n        backfill_policy (Optional[AutoMaterializePolicy]): BackfillPolicy to apply to all the loaded assets.\n        source_key_prefix (bool): Prefix to prepend to the keys of loaded SourceAssets. The returned\n            assets will be copies of the loaded objects, with the prefix prepended.\n\n    Returns:\n        Sequence[Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition]]:\n            A list containing assets, source assets, and cacheable assets defined in the module.\n    '
    package_module = import_module(package_name)
    return load_assets_from_package_module(package_module, group_name=group_name, key_prefix=key_prefix, freshness_policy=freshness_policy, auto_materialize_policy=auto_materialize_policy, backfill_policy=backfill_policy, source_key_prefix=source_key_prefix)

def find_modules_in_package(package_module: ModuleType) -> Iterable[ModuleType]:
    if False:
        return 10
    yield package_module
    package_path = package_module.__file__
    if package_path:
        for (_, modname, is_pkg) in pkgutil.walk_packages([os.path.dirname(package_path)]):
            submodule = import_module(f'{package_module.__name__}.{modname}')
            if is_pkg:
                yield from find_modules_in_package(submodule)
            else:
                yield submodule
    else:
        raise ValueError(f'Tried to find modules in package {package_module}, but its __file__ is None')

def prefix_assets(assets_defs: Sequence[AssetsDefinition], key_prefix: CoercibleToAssetKeyPrefix, source_assets: Sequence[SourceAsset], source_key_prefix: Optional[CoercibleToAssetKeyPrefix]) -> Tuple[Sequence[AssetsDefinition], Sequence[SourceAsset]]:
    if False:
        print('Hello World!')
    'Given a list of assets, prefix the input and output asset keys and check specs with key_prefix.\n    The prefix is not added to source assets.\n\n    Input asset keys that reference other assets within assets_defs are "brought along" -\n    i.e. prefixed as well.\n\n    Example with a single asset:\n\n        .. code-block:: python\n\n            @asset\n            def asset1():\n                ...\n\n            result = prefixed_asset_key_replacements([asset_1], "my_prefix")\n            assert result.assets[0].asset_key == AssetKey(["my_prefix", "asset1"])\n\n    Example with dependencies within the list of assets:\n\n        .. code-block:: python\n\n            @asset\n            def asset1():\n                ...\n\n            @asset\n            def asset2(asset1):\n                ...\n\n            result = prefixed_asset_key_replacements([asset1, asset2], "my_prefix")\n            assert result.assets[0].asset_key == AssetKey(["my_prefix", "asset1"])\n            assert result.assets[1].asset_key == AssetKey(["my_prefix", "asset2"])\n            assert result.assets[1].dependency_keys == {AssetKey(["my_prefix", "asset1"])}\n\n    '
    asset_keys = {asset_key for assets_def in assets_defs for asset_key in assets_def.keys}
    source_asset_keys = {source_asset.key for source_asset in source_assets}
    if isinstance(key_prefix, str):
        key_prefix = [key_prefix]
    key_prefix = check.is_list(key_prefix, of_type=str)
    result_assets: List[AssetsDefinition] = []
    for assets_def in assets_defs:
        output_asset_key_replacements = {asset_key: AssetKey([*key_prefix, *asset_key.path]) for asset_key in assets_def.keys}
        input_asset_key_replacements = {}
        for dep_asset_key in assets_def.dependency_keys:
            if dep_asset_key in asset_keys:
                input_asset_key_replacements[dep_asset_key] = AssetKey([*key_prefix, *dep_asset_key.path])
            elif source_key_prefix and dep_asset_key in source_asset_keys:
                input_asset_key_replacements[dep_asset_key] = AssetKey([*source_key_prefix, *dep_asset_key.path])
        check_specs_by_output_name = {output_name: check_spec.with_asset_key_prefix(key_prefix) for (output_name, check_spec) in assets_def.check_specs_by_output_name.items()}
        selected_asset_check_keys = {key.with_asset_key_prefix(key_prefix) for key in assets_def.check_keys}
        result_assets.append(assets_def.with_attributes(output_asset_key_replacements=output_asset_key_replacements, input_asset_key_replacements=input_asset_key_replacements, check_specs_by_output_name=check_specs_by_output_name, selected_asset_check_keys=selected_asset_check_keys))
    if source_key_prefix:
        result_source_assets = [source_asset.with_attributes(key=AssetKey([*source_key_prefix, *source_asset.key.path])) for source_asset in source_assets]
    else:
        result_source_assets = source_assets
    return (result_assets, result_source_assets)

def assets_with_attributes(assets_defs: Sequence[AssetsDefinition], source_assets: Sequence[SourceAsset], cacheable_assets: Sequence[CacheableAssetsDefinition], key_prefix: Optional[Sequence[str]], group_name: Optional[str], freshness_policy: Optional[FreshnessPolicy], auto_materialize_policy: Optional[AutoMaterializePolicy], backfill_policy: Optional[BackfillPolicy], source_key_prefix: Optional[Sequence[str]]) -> Sequence[Union[AssetsDefinition, SourceAsset, CacheableAssetsDefinition]]:
    if False:
        print('Hello World!')
    if key_prefix:
        (assets_defs, source_assets) = prefix_assets(assets_defs, key_prefix, source_assets, source_key_prefix)
        cacheable_assets = [cached_asset.with_prefix_for_all(key_prefix) for cached_asset in cacheable_assets]
    if group_name or freshness_policy or auto_materialize_policy or backfill_policy:
        assets_defs = [asset.with_attributes(group_names_by_key={asset_key: group_name for asset_key in asset.keys} if group_name is not None else None, freshness_policy=freshness_policy, auto_materialize_policy=auto_materialize_policy, backfill_policy=backfill_policy) for asset in assets_defs]
        if group_name:
            source_assets = [source_asset.with_attributes(group_name=group_name) for source_asset in source_assets]
        cacheable_assets = [cached_asset.with_attributes_for_all(group_name, freshness_policy=freshness_policy, auto_materialize_policy=auto_materialize_policy, backfill_policy=backfill_policy) for cached_asset in cacheable_assets]
    return [*assets_defs, *source_assets, *cacheable_assets]