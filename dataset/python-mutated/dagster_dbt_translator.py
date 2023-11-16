from dataclasses import dataclass
from typing import Any, Mapping, Optional
from dagster import AssetKey, AutoMaterializePolicy, FreshnessPolicy, PartitionMapping, _check as check
from dagster._annotations import public
from dagster._core.definitions.events import CoercibleToAssetKeyPrefix, check_opt_coercible_to_asset_key_prefix_param
from .asset_utils import default_asset_key_fn, default_auto_materialize_policy_fn, default_description_fn, default_freshness_policy_fn, default_group_from_dbt_resource_props, default_metadata_from_dbt_resource_props

@dataclass(frozen=True)
class DagsterDbtTranslatorSettings:
    """Settings to enable Dagster features for your dbt project.

    Args:
        enable_asset_checks (bool): Whether to load dbt tests as Dagster asset checks.
            Defaults to False.
    """
    enable_asset_checks: bool = False

class DagsterDbtTranslator:
    """Holds a set of methods that derive Dagster asset definition metadata given a representation
    of a dbt resource (models, tests, sources, etc).

    This class is exposed so that methods can be overriden to customize how Dagster asset metadata
    is derived.
    """

    def __init__(self, settings: Optional[DagsterDbtTranslatorSettings]=None):
        if False:
            i = 10
            return i + 15
        'Initialize the translator.\n\n        Args:\n            settings (Optional[DagsterDbtTranslatorSettings]): Settings for the translator.\n        '
        self._settings = settings or DagsterDbtTranslatorSettings()

    @property
    def settings(self) -> DagsterDbtTranslatorSettings:
        if False:
            print('Hello World!')
        if not hasattr(self, '_settings'):
            self._settings = DagsterDbtTranslatorSettings()
        return self._settings

    @classmethod
    @public
    def get_asset_key(cls, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
        if False:
            while True:
                i = 10
        'A function that takes a dictionary representing properties of a dbt resource, and\n        returns the Dagster asset key that represents that resource.\n\n        Note that a dbt resource is unrelated to Dagster\'s resource concept, and simply represents\n        a model, seed, snapshot or source in a given dbt project. You can learn more about dbt\n        resources and the properties available in this dictionary here:\n        https://docs.getdbt.com/reference/artifacts/manifest-json#resource-details\n\n        This method can be overridden to provide a custom asset key for a dbt resource.\n\n        Args:\n            dbt_resource_props (Mapping[str, Any]): A dictionary representing the dbt resource.\n\n        Returns:\n            AssetKey: The Dagster asset key for the dbt resource.\n\n        Examples:\n            Adding a prefix to the default asset key generated for each dbt resource:\n\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster import AssetKey\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_asset_key(cls, dbt_resource_props: Mapping[str, Any]) -> AssetKey:\n                        return super().get_asset_key(dbt_resource_props).with_prefix("prefix")\n\n            Adding a prefix to the default asset key generated for each dbt resource, but only for dbt sources:\n\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster import AssetKey\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_asset_key(cls, dbt_resource_props: Mapping[str, Any]) -> AssetKey:\n                        asset_key = super().get_asset_key(dbt_resource_props)\n\n                        if dbt_resource_props["resource_type"] == "source":\n                            asset_key = asset_key.with_prefix("my_prefix")\n\n                        return asset_key\n        '
        return default_asset_key_fn(dbt_resource_props)

    @classmethod
    @public
    def get_partition_mapping(cls, dbt_resource_props: Mapping[str, Any], dbt_parent_resource_props: Mapping[str, Any]) -> Optional[PartitionMapping]:
        if False:
            for i in range(10):
                print('nop')
        "A function that takes two dictionaries: the first, representing properties of a dbt\n        resource; and the second, representing the properties of a parent dependency to the first\n        dbt resource. The function returns the Dagster partition mapping for the dbt dependency.\n\n        Note that a dbt resource is unrelated to Dagster's resource concept, and simply represents\n        a model, seed, snapshot or source in a given dbt project. You can learn more about dbt\n        resources and the properties available in this dictionary here:\n        https://docs.getdbt.com/reference/artifacts/manifest-json#resource-details\n\n        This method can be overridden to provide a custom partition mapping for a dbt dependency.\n\n        Args:\n            dbt_resource_props (Mapping[str, Any]):\n                A dictionary representing the dbt child resource.\n            dbt_parent_resource_props (Mapping[str, Any]):\n                A dictionary representing the dbt parent resource, in relationship to the child.\n\n        Returns:\n            Optional[PartitionMapping]:\n                The Dagster partition mapping for the dbt resource. If None is returned, the\n                default partition mapping will be used.\n        "
        return None

    @classmethod
    @public
    def get_description(cls, dbt_resource_props: Mapping[str, Any]) -> str:
        if False:
            i = 10
            return i + 15
        'A function that takes a dictionary representing properties of a dbt resource, and\n        returns the Dagster description for that resource.\n\n        Note that a dbt resource is unrelated to Dagster\'s resource concept, and simply represents\n        a model, seed, snapshot or source in a given dbt project. You can learn more about dbt\n        resources and the properties available in this dictionary here:\n        https://docs.getdbt.com/reference/artifacts/manifest-json#resource-details\n\n        This method can be overridden to provide a custom description for a dbt resource.\n\n        Args:\n            dbt_resource_props (Mapping[str, Any]): A dictionary representing the dbt resource.\n\n        Returns:\n            str: The description for the dbt resource.\n\n        Examples:\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_description(cls, dbt_resource_props: Mapping[str, Any]) -> str:\n                        return "custom description"\n        '
        return default_description_fn(dbt_resource_props)

    @classmethod
    @public
    def get_metadata(cls, dbt_resource_props: Mapping[str, Any]) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        'A function that takes a dictionary representing properties of a dbt resource, and\n        returns the Dagster metadata for that resource.\n\n        Note that a dbt resource is unrelated to Dagster\'s resource concept, and simply represents\n        a model, seed, snapshot or source in a given dbt project. You can learn more about dbt\n        resources and the properties available in this dictionary here:\n        https://docs.getdbt.com/reference/artifacts/manifest-json#resource-details\n\n        This method can be overridden to provide a custom metadata for a dbt resource.\n\n        Args:\n            dbt_resource_props (Mapping[str, Any]): A dictionary representing the dbt resource.\n\n        Returns:\n            Mapping[str, Any]: A dictionary representing the Dagster metadata for the dbt resource.\n\n        Examples:\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_metadata(cls, dbt_resource_props: Mapping[str, Any]) -> Mapping[str, Any]:\n                        return {"custom": "metadata"}\n        '
        return default_metadata_from_dbt_resource_props(dbt_resource_props)

    @classmethod
    @public
    def get_group_name(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'A function that takes a dictionary representing properties of a dbt resource, and\n        returns the Dagster group name for that resource.\n\n        Note that a dbt resource is unrelated to Dagster\'s resource concept, and simply represents\n        a model, seed, snapshot or source in a given dbt project. You can learn more about dbt\n        resources and the properties available in this dictionary here:\n        https://docs.getdbt.com/reference/artifacts/manifest-json#resource-details\n\n        This method can be overridden to provide a custom group name for a dbt resource.\n\n        Args:\n            dbt_resource_props (Mapping[str, Any]): A dictionary representing the dbt resource.\n\n        Returns:\n            Optional[str]: A Dagster group name.\n\n        Examples:\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_group_name(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[str]:\n                        return "custom_group_prefix" + dbt_resource_props.get("config", {}).get("group")\n        '
        return default_group_from_dbt_resource_props(dbt_resource_props)

    @classmethod
    @public
    def get_freshness_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[FreshnessPolicy]:
        if False:
            while True:
                i = 10
        'A function that takes a dictionary representing properties of a dbt resource, and\n        returns the Dagster :py:class:`dagster.FreshnessPolicy` for that resource.\n\n        Note that a dbt resource is unrelated to Dagster\'s resource concept, and simply represents\n        a model, seed, snapshot or source in a given dbt project. You can learn more about dbt\n        resources and the properties available in this dictionary here:\n        https://docs.getdbt.com/reference/artifacts/manifest-json#resource-details\n\n        This method can be overridden to provide a custom freshness policy for a dbt resource.\n\n        Args:\n            dbt_resource_props (Mapping[str, Any]): A dictionary representing the dbt resource.\n\n        Returns:\n            Optional[FreshnessPolicy]: A Dagster freshness policy.\n\n        Examples:\n            Set a custom freshness policy for all dbt resources:\n\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_freshness_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[FreshnessPolicy]:\n                        return FreshnessPolicy(maximum_lag_minutes=60)\n\n            Set a custom freshness policy for dbt resources with a specific tag:\n\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_freshness_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[FreshnessPolicy]:\n                        freshness_policy = None\n                        if "my_custom_tag" in dbt_resource_props.get("tags", []):\n                            freshness_policy = FreshnessPolicy(maximum_lag_minutes=60)\n\n                        return freshness_policy\n        '
        return default_freshness_policy_fn(dbt_resource_props)

    @classmethod
    @public
    def get_auto_materialize_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[AutoMaterializePolicy]:
        if False:
            return 10
        'A function that takes a dictionary representing properties of a dbt resource, and\n        returns the Dagster :py:class:`dagster.AutoMaterializePolicy` for that resource.\n\n        Note that a dbt resource is unrelated to Dagster\'s resource concept, and simply represents\n        a model, seed, snapshot or source in a given dbt project. You can learn more about dbt\n        resources and the properties available in this dictionary here:\n        https://docs.getdbt.com/reference/artifacts/manifest-json#resource-details\n\n        This method can be overridden to provide a custom auto-materialize policy for a dbt resource.\n\n        Args:\n            dbt_resource_props (Mapping[str, Any]): A dictionary representing the dbt resource.\n\n        Returns:\n            Optional[AutoMaterializePolicy]: A Dagster auto-materialize policy.\n\n        Examples:\n            Set a custom auto-materialize policy for all dbt resources:\n\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_auto_materialize_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[AutoMaterializePolicy]:\n                        return AutoMaterializePolicy.eager()\n\n            Set a custom auto-materialize policy for dbt resources with a specific tag:\n\n            .. code-block:: python\n\n                from typing import Any, Mapping\n\n                from dagster_dbt import DagsterDbtTranslator\n\n\n                class CustomDagsterDbtTranslator(DagsterDbtTranslator):\n                    @classmethod\n                    def get_auto_materialize_policy(cls, dbt_resource_props: Mapping[str, Any]) -> Optional[AutoMaterializePolicy]:\n                        auto_materialize_policy = None\n                        if "my_custom_tag" in dbt_resource_props.get("tags", []):\n                            auto_materialize_policy = AutoMaterializePolicy.eager()\n\n                        return auto_materialize_policy\n\n        '
        return default_auto_materialize_policy_fn(dbt_resource_props)

class KeyPrefixDagsterDbtTranslator(DagsterDbtTranslator):
    """A DagsterDbtTranslator that applies prefixes to the asset keys generated from dbt resources.

    Attributes:
        asset_key_prefix (Optional[Union[str, Sequence[str]]]): A prefix to apply to all dbt models,
            seeds, snapshots, etc. This will *not* apply to dbt sources.
        source_asset_key_prefix (Optional[Union[str, Sequence[str]]]): A prefix to apply to all dbt
            sources.
    """

    def __init__(self, asset_key_prefix: Optional[CoercibleToAssetKeyPrefix]=None, source_asset_key_prefix: Optional[CoercibleToAssetKeyPrefix]=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._asset_key_prefix = check_opt_coercible_to_asset_key_prefix_param(asset_key_prefix, 'asset_key_prefix') or []
        self._source_asset_key_prefix = check_opt_coercible_to_asset_key_prefix_param(source_asset_key_prefix, 'source_asset_key_prefix') or []
        super().__init__(*args, **kwargs)

    @public
    def get_asset_key(self, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
        if False:
            i = 10
            return i + 15
        base_key = default_asset_key_fn(dbt_resource_props)
        if dbt_resource_props['resource_type'] == 'source':
            return base_key.with_prefix(self._source_asset_key_prefix)
        else:
            return base_key.with_prefix(self._asset_key_prefix)

@dataclass
class DbtManifestWrapper:
    manifest: Mapping[str, Any]

def validate_translator(dagster_dbt_translator: DagsterDbtTranslator) -> DagsterDbtTranslator:
    if False:
        while True:
            i = 10
    return check.inst_param(dagster_dbt_translator, 'dagster_dbt_translator', DagsterDbtTranslator, additional_message='Ensure that the argument is an instantiated class that subclasses DagsterDbtTranslator.')

def validate_opt_translator(dagster_dbt_translator: Optional[DagsterDbtTranslator]) -> Optional[DagsterDbtTranslator]:
    if False:
        for i in range(10):
            print('nop')
    return check.opt_inst_param(dagster_dbt_translator, 'dagster_dbt_translator', DagsterDbtTranslator, additional_message='Ensure that the argument is an instantiated class that subclasses DagsterDbtTranslator.')