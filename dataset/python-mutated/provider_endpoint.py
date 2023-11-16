from __future__ import annotations
from typing import TYPE_CHECKING
import re2
from airflow.api_connexion import security
from airflow.api_connexion.schemas.provider_schema import Provider, ProviderCollection, provider_collection_schema
from airflow.auth.managers.models.resource_details import AccessView
from airflow.providers_manager import ProvidersManager
if TYPE_CHECKING:
    from airflow.api_connexion.types import APIResponse
    from airflow.providers_manager import ProviderInfo

def _remove_rst_syntax(value: str) -> str:
    if False:
        return 10
    return re2.sub('[`_<>]', '', value.strip(' \n.'))

def _provider_mapper(provider: ProviderInfo) -> Provider:
    if False:
        i = 10
        return i + 15
    return Provider(package_name=provider.data['package-name'], description=_remove_rst_syntax(provider.data['description']), version=provider.version)

@security.requires_access_view(AccessView.PROVIDERS)
def get_providers() -> APIResponse:
    if False:
        print('Hello World!')
    'Get providers.'
    providers = [_provider_mapper(d) for d in ProvidersManager().providers.values()]
    total_entries = len(providers)
    return provider_collection_schema.dump(ProviderCollection(providers=providers, total_entries=total_entries))