from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.api_connexion import security
from airflow.api_connexion.parameters import check_limit, format_parameters
from airflow.api_connexion.schemas.plugin_schema import PluginCollection, plugin_collection_schema
from airflow.auth.managers.models.resource_details import AccessView
from airflow.plugins_manager import get_plugin_info
if TYPE_CHECKING:
    from airflow.api_connexion.types import APIResponse

@security.requires_access_view(AccessView.PLUGINS)
@format_parameters({'limit': check_limit})
def get_plugins(*, limit: int, offset: int=0) -> APIResponse:
    if False:
        while True:
            i = 10
    'Get plugins endpoint.'
    plugins_info = get_plugin_info()
    collection = PluginCollection(plugins=plugins_info[offset:][:limit], total_entries=len(plugins_info))
    return plugin_collection_schema.dump(collection)