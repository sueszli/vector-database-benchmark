import collections.abc
import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import AbstractSet, Iterable, Optional, Sequence, Union, cast
from typing_extensions import TypeAlias
import dagster._check as check
from dagster._annotations import deprecated, public
from dagster._core.definitions.asset_checks import AssetChecksDefinition
from dagster._core.errors import DagsterInvalidSubsetError
from dagster._core.selector.subset_selector import fetch_connected, fetch_sinks, fetch_sources, parse_clause
from .asset_check_spec import AssetCheckKey
from .asset_graph import AssetGraph, InternalAssetGraph
from .assets import AssetsDefinition
from .events import AssetKey, CoercibleToAssetKey, CoercibleToAssetKeyPrefix, key_prefix_from_coercible
from .source_asset import SourceAsset
CoercibleToAssetSelection: TypeAlias = Union[str, Sequence[str], Sequence[AssetKey], Sequence[Union['AssetsDefinition', 'SourceAsset']], 'AssetSelection']

class AssetSelection(ABC):
    """An AssetSelection defines a query over a set of assets and asset checks, normally all that are defined in a code location.

    You can use the "|", "&", and "-" operators to create unions, intersections, and differences of selections, respectively.

    AssetSelections are typically used with :py:func:`define_asset_job`.

    By default, selecting assets will also select all of the asset checks that target those assets.

    Examples:
        .. code-block:: python

            # Select all assets in group "marketing":
            AssetSelection.groups("marketing")

            # Select all assets in group "marketing", as well as the asset with key "promotion":
            AssetSelection.groups("marketing") | AssetSelection.keys("promotion")

            # Select all assets in group "marketing" that are downstream of asset "leads":
            AssetSelection.groups("marketing") & AssetSelection.keys("leads").downstream()

            # Select a list of assets:
            AssetSelection.assets(*my_assets_list)

            # Select all assets except for those in group "marketing"
            AssetSelection.all() - AssetSelection.groups("marketing")

            # Select all assets which are materialized by the same op as "projections":
            AssetSelection.keys("projections").required_multi_asset_neighbors()

            # Select all assets in group "marketing" and exclude their asset checks:
            AssetSelection.groups("marketing") - AssetSelection.all_asset_checks()

            # Select all asset checks that target a list of assets:
            AssetSelection.checks_for_assets(*my_assets_list)

            # Select a specific asset check:
            AssetSelection.checks(my_asset_check)

    """

    @public
    @staticmethod
    def all() -> 'AllSelection':
        if False:
            for i in range(10):
                print('nop')
        'Returns a selection that includes all assets and asset checks.'
        return AllSelection()

    @public
    @staticmethod
    def all_asset_checks() -> 'AllAssetCheckSelection':
        if False:
            print('Hello World!')
        'Returns a selection that includes all asset checks.'
        return AllAssetCheckSelection()

    @public
    @staticmethod
    def assets(*assets_defs: AssetsDefinition) -> 'KeysAssetSelection':
        if False:
            print('Hello World!')
        'Returns a selection that includes all of the provided assets and asset checks that target them.'
        return KeysAssetSelection(*(key for assets_def in assets_defs for key in assets_def.keys))

    @public
    @staticmethod
    def keys(*asset_keys: CoercibleToAssetKey) -> 'KeysAssetSelection':
        if False:
            i = 10
            return i + 15
        'Returns a selection that includes assets with any of the provided keys and all asset checks that target them.\n\n        Examples:\n            .. code-block:: python\n\n                AssetSelection.keys(AssetKey(["a"]))\n\n                AssetSelection.keys("a")\n\n                AssetSelection.keys(AssetKey(["a"]), AssetKey(["b"]))\n\n                AssetSelection.keys("a", "b")\n\n                asset_key_list = [AssetKey(["a"]), AssetKey(["b"])]\n                AssetSelection.keys(*asset_key_list)\n        '
        _asset_keys = [AssetKey.from_user_string(key) if isinstance(key, str) else AssetKey.from_coercible(key) for key in asset_keys]
        return KeysAssetSelection(*_asset_keys)

    @public
    @staticmethod
    def key_prefixes(*key_prefixes: CoercibleToAssetKeyPrefix, include_sources: bool=False) -> 'KeyPrefixesAssetSelection':
        if False:
            while True:
                i = 10
        'Returns a selection that includes assets that match any of the provided key prefixes and all the asset checks that target them.\n\n        Args:\n            include_sources (bool): If True, then include source assets matching the key prefix(es)\n                in the selection.\n\n        Examples:\n            .. code-block:: python\n\n              # match any asset key where the first segment is equal to "a" or "b"\n              # e.g. AssetKey(["a", "b", "c"]) would match, but AssetKey(["abc"]) would not.\n              AssetSelection.key_prefixes("a", "b")\n\n              # match any asset key where the first two segments are ["a", "b"] or ["a", "c"]\n              AssetSelection.key_prefixes(["a", "b"], ["a", "c"])\n        '
        _asset_key_prefixes = [key_prefix_from_coercible(key_prefix) for key_prefix in key_prefixes]
        return KeyPrefixesAssetSelection(*_asset_key_prefixes, include_sources=include_sources)

    @public
    @staticmethod
    def groups(*group_strs, include_sources: bool=False) -> 'GroupsAssetSelection':
        if False:
            print('Hello World!')
        'Returns a selection that includes materializable assets that belong to any of the\n        provided groups and all the asset checks that target them.\n\n        Args:\n            include_sources (bool): If True, then include source assets matching the group in the\n                selection.\n        '
        check.tuple_param(group_strs, 'group_strs', of_type=str)
        return GroupsAssetSelection(*group_strs, include_sources=include_sources)

    @public
    @staticmethod
    def checks_for_assets(*assets_defs: AssetsDefinition) -> 'AssetChecksForAssetKeys':
        if False:
            i = 10
            return i + 15
        'Returns a selection with the asset checks that target the provided assets.'
        return AssetChecksForAssetKeys([key for assets_def in assets_defs for key in assets_def.keys])

    @public
    @staticmethod
    def checks(*asset_checks: AssetChecksDefinition) -> 'AssetChecksForHandles':
        if False:
            print('Hello World!')
        'Returns a selection that includes all of the provided asset checks.'
        return AssetChecksForHandles([AssetCheckKey(asset_key=AssetKey.from_coercible(spec.asset_key), name=spec.name) for checks_def in asset_checks for spec in checks_def.specs])

    @public
    def downstream(self, depth: Optional[int]=None, include_self: bool=True) -> 'DownstreamAssetSelection':
        if False:
            print('Hello World!')
        'Returns a selection that includes all assets that are downstream of any of the assets in\n        this selection, selecting the assets in this selection by default. Includes the asset checks targeting the returned assets. Iterates through each\n        asset in this selection and returns the union of all downstream assets.\n\n        depth (Optional[int]): If provided, then only include assets to the given depth. A depth\n            of 2 means all assets that are children or grandchildren of the assets in this\n            selection.\n        include_self (bool): If True, then include the assets in this selection in the result.\n            If the include_self flag is False, return each downstream asset that is not part of the\n            original selection. By default, set to True.\n        '
        check.opt_int_param(depth, 'depth')
        check.opt_bool_param(include_self, 'include_self')
        return DownstreamAssetSelection(self, depth=depth, include_self=include_self)

    @public
    def upstream(self, depth: Optional[int]=None, include_self: bool=True) -> 'UpstreamAssetSelection':
        if False:
            i = 10
            return i + 15
        'Returns a selection that includes all materializable assets that are upstream of any of\n        the assets in this selection, selecting the assets in this selection by default. Includes the asset checks targeting the returned assets. Iterates\n        through each asset in this selection and returns the union of all upstream assets.\n\n        Because mixed selections of source and materializable assets are currently not supported,\n        keys corresponding to `SourceAssets` will not be included as upstream of regular assets.\n\n        Args:\n            depth (Optional[int]): If provided, then only include assets to the given depth. A depth\n                of 2 means all assets that are parents or grandparents of the assets in this\n                selection.\n            include_self (bool): If True, then include the assets in this selection in the result.\n                If the include_self flag is False, return each upstream asset that is not part of the\n                original selection. By default, set to True.\n        '
        check.opt_int_param(depth, 'depth')
        check.opt_bool_param(include_self, 'include_self')
        return UpstreamAssetSelection(self, depth=depth, include_self=include_self)

    @public
    def sinks(self) -> 'SinkAssetSelection':
        if False:
            while True:
                i = 10
        'Given an asset selection, returns a new asset selection that contains all of the sink\n        assets within the original asset selection. Includes the asset checks targeting the returned assets.\n\n        A sink asset is an asset that has no downstream dependencies within the asset selection.\n        The sink asset can have downstream dependencies outside of the asset selection.\n        '
        return SinkAssetSelection(self)

    @public
    def required_multi_asset_neighbors(self) -> 'RequiredNeighborsAssetSelection':
        if False:
            return 10
        'Given an asset selection in which some assets are output from a multi-asset compute op\n        which cannot be subset, returns a new asset selection that contains all of the assets\n        required to execute the original asset selection. Includes the asset checks targeting the returned assets.\n        '
        return RequiredNeighborsAssetSelection(self)

    @public
    def roots(self) -> 'RootAssetSelection':
        if False:
            while True:
                i = 10
        'Given an asset selection, returns a new asset selection that contains all of the root\n        assets within the original asset selection. Includes the asset checks targeting the returned assets.\n\n        A root asset is an asset that has no upstream dependencies within the asset selection.\n        The root asset can have downstream dependencies outside of the asset selection.\n\n        Because mixed selections of source and materializable assets are currently not supported,\n        keys corresponding to `SourceAssets` will not be included as roots. To select source assets,\n        use the `upstream_source_assets` method.\n        '
        return RootAssetSelection(self)

    @public
    @deprecated(breaking_version='2.0', additional_warn_text='Use AssetSelection.roots instead.')
    def sources(self) -> 'RootAssetSelection':
        if False:
            while True:
                i = 10
        'Given an asset selection, returns a new asset selection that contains all of the root\n        assets within the original asset selection. Includes the asset checks targeting the returned assets.\n\n        A root asset is a materializable asset that has no upstream dependencies within the asset\n        selection. The root asset can have downstream dependencies outside of the asset selection.\n\n        Because mixed selections of source and materializable assets are currently not supported,\n        keys corresponding to `SourceAssets` will not be included as roots. To select source assets,\n        use the `upstream_source_assets` method.\n        '
        return self.roots()

    @public
    def upstream_source_assets(self) -> 'SourceAssetSelection':
        if False:
            i = 10
            return i + 15
        'Given an asset selection, returns a new asset selection that contains all of the source\n        assets upstream of assets in the original selection. Includes the asset checks targeting the returned assets.\n        '
        return SourceAssetSelection(self)

    @public
    def without_checks(self) -> 'AssetSelection':
        if False:
            return 10
        'Removes all asset checks in the selection.'
        return self - AssetSelection.all_asset_checks()

    def __or__(self, other: 'AssetSelection') -> 'OrAssetSelection':
        if False:
            return 10
        check.inst_param(other, 'other', AssetSelection)
        return OrAssetSelection(self, other)

    def __and__(self, other: 'AssetSelection') -> 'AndAssetSelection':
        if False:
            for i in range(10):
                print('nop')
        check.inst_param(other, 'other', AssetSelection)
        return AndAssetSelection(self, other)

    def __sub__(self, other: 'AssetSelection') -> 'SubAssetSelection':
        if False:
            while True:
                i = 10
        check.inst_param(other, 'other', AssetSelection)
        return SubAssetSelection(self, other)

    def resolve(self, all_assets: Union[Iterable[Union[AssetsDefinition, SourceAsset]], AssetGraph]) -> AbstractSet[AssetKey]:
        if False:
            return 10
        if isinstance(all_assets, AssetGraph):
            asset_graph = all_assets
        else:
            check.iterable_param(all_assets, 'all_assets', (AssetsDefinition, SourceAsset))
            asset_graph = AssetGraph.from_assets(all_assets)
        resolved = self.resolve_inner(asset_graph)
        resolved_source_assets = asset_graph.source_asset_keys & resolved
        resolved_regular_assets = resolved - asset_graph.source_asset_keys
        check.invariant(not (len(resolved_source_assets) > 0 and len(resolved_regular_assets) > 0), 'Asset selection specified both regular assets and source assets. This is not currently supported. Selections must be all regular assets or all source assets.')
        return resolved

    @abstractmethod
    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def resolve_checks(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            while True:
                i = 10
        "We don't need this method currently, but it makes things consistent with resolve_inner. Currently\n        we don't store checks in the ExternalAssetGraph, so we only support InternalAssetGraph.\n        "
        return self.resolve_checks_inner(asset_graph)

    def resolve_checks_inner(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            print('Hello World!')
        'By default, resolve to checks that target the selected assets. This is overriden for particular selections.'
        asset_keys = self.resolve(asset_graph)
        return {handle for handle in asset_graph.asset_check_keys if handle.asset_key in asset_keys}

    @staticmethod
    def _selection_from_string(string: str) -> 'AssetSelection':
        if False:
            while True:
                i = 10
        from dagster._core.definitions import AssetSelection
        if string == '*':
            return AssetSelection.all()
        parts = parse_clause(string)
        if not parts:
            check.failed(f'Invalid selection string: {string}')
        (u, item, d) = parts
        selection: AssetSelection = AssetSelection.keys(item)
        if u:
            selection = selection.upstream(u)
        if d:
            selection = selection.downstream(d)
        return selection

    @classmethod
    def from_coercible(cls, selection: CoercibleToAssetSelection) -> 'AssetSelection':
        if False:
            i = 10
            return i + 15
        if isinstance(selection, str):
            return cls._selection_from_string(selection)
        elif isinstance(selection, AssetSelection):
            return selection
        elif isinstance(selection, collections.abc.Sequence) and all((isinstance(el, str) for el in selection)):
            return reduce(operator.or_, [cls._selection_from_string(cast(str, s)) for s in selection])
        elif isinstance(selection, collections.abc.Sequence) and all((isinstance(el, (AssetsDefinition, SourceAsset)) for el in selection)):
            return AssetSelection.keys(*(key for el in selection for key in (el.keys if isinstance(el, AssetsDefinition) else [cast(SourceAsset, el).key])))
        elif isinstance(selection, collections.abc.Sequence) and all((isinstance(el, AssetKey) for el in selection)):
            return cls.keys(*cast(Sequence[AssetKey], selection))
        else:
            check.failed(f'selection argument must be one of str, Sequence[str], Sequence[AssetKey], Sequence[AssetsDefinition], Sequence[SourceAsset], AssetSelection. Was {type(selection)}.')

class AllSelection(AssetSelection):

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            return 10
        return asset_graph.materializable_asset_keys

class AllAssetCheckSelection(AssetSelection):

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            i = 10
            return i + 15
        return set()

    def resolve_checks_inner(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            print('Hello World!')
        return asset_graph.asset_check_keys

class AssetChecksForAssetKeys(AssetSelection):

    def __init__(self, keys: Sequence[AssetKey]):
        if False:
            for i in range(10):
                print('nop')
        self._keys = keys

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            print('Hello World!')
        return set()

    def resolve_checks_inner(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            print('Hello World!')
        return {handle for handle in asset_graph.asset_check_keys if handle.asset_key in self._keys}

class AssetChecksForHandles(AssetSelection):

    def __init__(self, asset_check_keys: Sequence[AssetCheckKey]):
        if False:
            while True:
                i = 10
        self._asset_check_keys = asset_check_keys

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        return set()

    def resolve_checks_inner(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            for i in range(10):
                print('nop')
        return {handle for handle in asset_graph.asset_check_keys if handle in self._asset_check_keys}

class AndAssetSelection(AssetSelection):

    def __init__(self, left: AssetSelection, right: AssetSelection):
        if False:
            while True:
                i = 10
        self._left = left
        self._right = right

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            return 10
        return self._left.resolve_inner(asset_graph) & self._right.resolve_inner(asset_graph)

    def resolve_checks_inner(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            for i in range(10):
                print('nop')
        return self._left.resolve_checks_inner(asset_graph) & self._right.resolve_checks_inner(asset_graph)

class SubAssetSelection(AssetSelection):

    def __init__(self, left: AssetSelection, right: AssetSelection):
        if False:
            i = 10
            return i + 15
        self._left = left
        self._right = right

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        return self._left.resolve_inner(asset_graph) - self._right.resolve_inner(asset_graph)

    def resolve_checks_inner(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            print('Hello World!')
        return self._left.resolve_checks_inner(asset_graph) - self._right.resolve_checks_inner(asset_graph)

class SinkAssetSelection(AssetSelection):

    def __init__(self, child: AssetSelection):
        if False:
            for i in range(10):
                print('nop')
        self._child = child

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        selection = self._child.resolve_inner(asset_graph)
        return fetch_sinks(asset_graph.asset_dep_graph, selection)

class RequiredNeighborsAssetSelection(AssetSelection):

    def __init__(self, child: AssetSelection):
        if False:
            return 10
        self._child = child

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            while True:
                i = 10
        selection = self._child.resolve_inner(asset_graph)
        output = set(selection)
        for asset_key in selection:
            output.update(asset_graph.get_required_multi_asset_keys(asset_key))
        return output

class RootAssetSelection(AssetSelection):

    def __init__(self, child: AssetSelection):
        if False:
            print('Hello World!')
        self._child = child

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        selection = self._child.resolve_inner(asset_graph)
        return fetch_sources(asset_graph.asset_dep_graph, selection)

class DownstreamAssetSelection(AssetSelection):

    def __init__(self, child: AssetSelection, *, depth: Optional[int]=None, include_self: Optional[bool]=True):
        if False:
            print('Hello World!')
        self._child = child
        self.depth = depth
        self.include_self = include_self

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            print('Hello World!')
        selection = self._child.resolve_inner(asset_graph)
        return operator.sub(reduce(operator.or_, [{asset_key} | fetch_connected(item=asset_key, graph=asset_graph.asset_dep_graph, direction='downstream', depth=self.depth) for asset_key in selection]), selection if not self.include_self else set())

class GroupsAssetSelection(AssetSelection):

    def __init__(self, *groups: str, include_sources: bool):
        if False:
            for i in range(10):
                print('nop')
        self._groups = groups
        self._include_sources = include_sources

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            while True:
                i = 10
        base_set = asset_graph.all_asset_keys if self._include_sources else asset_graph.materializable_asset_keys
        return {asset_key for (asset_key, group) in asset_graph.group_names_by_key.items() if group in self._groups and asset_key in base_set}

class KeysAssetSelection(AssetSelection):

    def __init__(self, *keys: AssetKey):
        if False:
            while True:
                i = 10
        self._keys = keys

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            while True:
                i = 10
        specified_keys = set(self._keys)
        invalid_keys = {key for key in specified_keys if key not in asset_graph.all_asset_keys}
        if invalid_keys:
            raise DagsterInvalidSubsetError(f'AssetKey(s) {invalid_keys} were selected, but no AssetsDefinition objects supply these keys. Make sure all keys are spelled correctly, and all AssetsDefinitions are correctly added to the `Definitions`.')
        return specified_keys

class KeyPrefixesAssetSelection(AssetSelection):

    def __init__(self, *key_prefixes: Sequence[str], include_sources: bool):
        if False:
            while True:
                i = 10
        self._key_prefixes = key_prefixes
        self._include_sources = include_sources

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        base_set = asset_graph.all_asset_keys if self._include_sources else asset_graph.materializable_asset_keys
        return {key for key in base_set if any((key.has_prefix(prefix) for prefix in self._key_prefixes))}

class OrAssetSelection(AssetSelection):

    def __init__(self, left: AssetSelection, right: AssetSelection):
        if False:
            return 10
        self._left = left
        self._right = right

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            i = 10
            return i + 15
        return self._left.resolve_inner(asset_graph) | self._right.resolve_inner(asset_graph)

    def resolve_checks_inner(self, asset_graph: InternalAssetGraph) -> AbstractSet[AssetCheckKey]:
        if False:
            print('Hello World!')
        return self._left.resolve_checks_inner(asset_graph) | self._right.resolve_checks_inner(asset_graph)

def _fetch_all_upstream(selection: AbstractSet[AssetKey], asset_graph: AssetGraph, depth: Optional[int]=None, include_self: bool=True) -> AbstractSet[AssetKey]:
    if False:
        for i in range(10):
            print('nop')
    return operator.sub(reduce(operator.or_, [{asset_key} | fetch_connected(item=asset_key, graph=asset_graph.asset_dep_graph, direction='upstream', depth=depth) for asset_key in selection], set()), selection if not include_self else set())

class UpstreamAssetSelection(AssetSelection):

    def __init__(self, child: AssetSelection, *, depth: Optional[int]=None, include_self: bool=True):
        if False:
            while True:
                i = 10
        self._child = child
        self.depth = depth
        self.include_self = include_self

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            print('Hello World!')
        selection = self._child.resolve_inner(asset_graph)
        if len(selection) == 0:
            return selection
        all_upstream = _fetch_all_upstream(selection, asset_graph, self.depth, self.include_self)
        return {key for key in all_upstream if key not in asset_graph.source_asset_keys}

class SourceAssetSelection(AssetSelection):

    def __init__(self, child: AssetSelection):
        if False:
            while True:
                i = 10
        self._child = child

    def resolve_inner(self, asset_graph: AssetGraph) -> AbstractSet[AssetKey]:
        if False:
            return 10
        selection = self._child.resolve_inner(asset_graph)
        if len(selection) == 0:
            return selection
        all_upstream = _fetch_all_upstream(selection, asset_graph)
        return {key for key in all_upstream if key in asset_graph.source_asset_keys}