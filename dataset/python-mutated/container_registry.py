"""Hook for Azure Container Registry."""
from __future__ import annotations
from functools import cached_property
from typing import Any
from azure.mgmt.containerinstance.models import ImageRegistryCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from airflow.hooks.base import BaseHook
from airflow.providers.microsoft.azure.utils import add_managed_identity_connection_widgets, get_field, get_sync_default_azure_credential

class AzureContainerRegistryHook(BaseHook):
    """
    A hook to communicate with a Azure Container Registry.

    :param conn_id: :ref:`Azure Container Registry connection id<howto/connection:acr>`
        of a service principal which will be used to start the container instance

    """
    conn_name_attr = 'azure_container_registry_conn_id'
    default_conn_name = 'azure_container_registry_default'
    conn_type = 'azure_container_registry'
    hook_name = 'Azure Container Registry'

    @staticmethod
    @add_managed_identity_connection_widgets
    def get_connection_form_widgets() -> dict[str, Any]:
        if False:
            return 10
        'Returns connection widgets to add to connection form.'
        from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import StringField
        return {'subscription_id': StringField(lazy_gettext('Subscription ID (optional)'), widget=BS3TextFieldWidget()), 'resource_group': StringField(lazy_gettext('Resource group name (optional)'), widget=BS3TextFieldWidget())}

    @classmethod
    def get_ui_field_behaviour(cls) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns custom field behaviour.'
        return {'hidden_fields': ['schema', 'port', 'extra'], 'relabeling': {'login': 'Registry Username', 'password': 'Registry Password', 'host': 'Registry Server'}, 'placeholders': {'login': 'private registry username', 'password': 'private registry password', 'host': 'docker image registry server', 'subscription_id': 'Subscription id (required for Azure AD authentication)', 'resource_group': 'Resource group name (required for Azure AD authentication)'}}

    def __init__(self, conn_id: str='azure_registry') -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.conn_id = conn_id

    def _get_field(self, extras, name):
        if False:
            print('Hello World!')
        return get_field(conn_id=self.conn_id, conn_type=self.conn_type, extras=extras, field_name=name)

    @cached_property
    def connection(self) -> ImageRegistryCredential:
        if False:
            for i in range(10):
                print('nop')
        return self.get_conn()

    def get_conn(self) -> ImageRegistryCredential:
        if False:
            for i in range(10):
                print('nop')
        conn = self.get_connection(self.conn_id)
        password = conn.password
        if not password:
            extras = conn.extra_dejson
            subscription_id = self._get_field(extras, 'subscription_id')
            resource_group = self._get_field(extras, 'resource_group')
            managed_identity_client_id = self._get_field(extras, 'managed_identity_client_id')
            workload_identity_tenant_id = self._get_field(extras, 'workload_identity_tenant_id')
            credential = get_sync_default_azure_credential(managed_identity_client_id=managed_identity_client_id, workload_identity_tenant_id=workload_identity_tenant_id)
            client = ContainerRegistryManagementClient(credential=credential, subscription_id=subscription_id)
            credentials = client.registries.list_credentials(resource_group, conn.login).as_dict()
            password = credentials['passwords'][0]['value']
        return ImageRegistryCredential(server=conn.host, username=conn.login, password=password)