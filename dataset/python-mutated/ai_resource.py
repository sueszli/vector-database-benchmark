from typing import Dict, Optional
from azure.ai.ml.entities import CustomerManagedKey, WorkspaceHub
from azure.ai.ml.entities._credentials import IdentityConfiguration
from azure.ai.ml.entities._workspace.networking import ManagedNetwork
from azure.ai.ml.entities._workspace_hub.workspace_hub_config import WorkspaceHubConfig

class AIResource:
    """An AI Resource, which serves as a container for projects and other AI-related objects"""

    def __init__(self, *, name: str, description: Optional[str]=None, tags: Optional[Dict[str, str]]=None, display_name: Optional[str]=None, location: Optional[str]=None, resource_group: Optional[str]=None, managed_network: Optional[ManagedNetwork]=None, storage_account: Optional[str]=None, customer_managed_key: Optional[CustomerManagedKey]=None, public_network_access: Optional[str]=None, identity: Optional[IdentityConfiguration]=None, primary_user_assigned_identity: Optional[str]=None, default_workspace_resource_group: Optional[str]=None, **kwargs):
        if False:
            return 10
        self._workspace_hub = WorkspaceHub(name=name, description=description, tags=tags, display_name=display_name, location=location, resource_group=resource_group, managed_network=managed_network, storage_account=storage_account, customer_managed_key=customer_managed_key, public_network_access=public_network_access, identity=identity, primary_user_assigned_identity=primary_user_assigned_identity, workspace_hub_config=WorkspaceHubConfig(additional_workspace_storage_accounts=[], default_workspace_resource_group=default_workspace_resource_group), **kwargs)

    @classmethod
    def _from_v2_workspace_hub(cls, workspace_hub: WorkspaceHub) -> 'AIResource':
        if False:
            for i in range(10):
                print('nop')
        'Create a connection from a v2 AML SDK workspace hub. For internal use.\n\n        :param workspace_hub: The workspace connection object to convert into a workspace.\n        :type workspace_hub: ~azure.ai.ml.entities.WorkspaceConnection\n\n        :return: The converted AI resource.\n        :rtype: ~azure.ai.resources.entities.AIResource\n        '
        resource = cls(name='a')
        resource._workspace_hub = workspace_hub
        return resource

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        'The name of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.name

    @name.setter
    def name(self, value: str):
        if False:
            i = 10
            return i + 15
        'Set the name of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.name = value

    @property
    def description(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The description of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.description

    @description.setter
    def description(self, value: str):
        if False:
            for i in range(10):
                print('nop')
        'Set the description of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.description = value

    @property
    def tags(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The tags of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.tags

    @tags.setter
    def tags(self, value: str):
        if False:
            while True:
                i = 10
        'Set the tags of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.tags = value

    @property
    def display_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The display_name of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.display_name

    @display_name.setter
    def display_name(self, value: str):
        if False:
            i = 10
            return i + 15
        'Set the display_name of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.display_name = value

    @property
    def location(self) -> str:
        if False:
            print('Hello World!')
        'The location of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.location

    @location.setter
    def location(self, value: str):
        if False:
            print('Hello World!')
        'Set the location of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.location = value

    @property
    def resource_group(self) -> str:
        if False:
            i = 10
            return i + 15
        'The resource_group of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.resource_group

    @resource_group.setter
    def resource_group(self, value: str):
        if False:
            print('Hello World!')
        'Set the resource_group of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.resource_group = value

    @property
    def managed_network(self) -> str:
        if False:
            i = 10
            return i + 15
        'The managed_network of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.managed_network

    @managed_network.setter
    def managed_network(self, value: str):
        if False:
            return 10
        'Set the managed_network of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.managed_network = value

    @property
    def storage_account(self) -> str:
        if False:
            i = 10
            return i + 15
        'The storage_account of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.storage_account

    @storage_account.setter
    def storage_account(self, value: str):
        if False:
            return 10
        'Set the storage_account of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.storage_account = value

    @property
    def existing_workspaces(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The existing_workspaces of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.existing_workspaces

    @property
    def customer_managed_key(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The customer_managed_key of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.customer_managed_key

    @customer_managed_key.setter
    def customer_managed_key(self, value: str):
        if False:
            return 10
        'Set the customer_managed_key of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.customer_managed_key = value

    @property
    def public_network_access(self) -> str:
        if False:
            print('Hello World!')
        'The public_network_access of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.public_network_access

    @public_network_access.setter
    def public_network_access(self, value: str):
        if False:
            for i in range(10):
                print('nop')
        'Set the public_network_access of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.public_network_access = value

    @property
    def identity(self) -> str:
        if False:
            return 10
        'The identity of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.identity

    @identity.setter
    def identity(self, value: str):
        if False:
            print('Hello World!')
        'Set the identity of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.identity = value

    @property
    def primary_user_assigned_identity(self) -> str:
        if False:
            print('Hello World!')
        'The primary_user_assigned_identity of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.primary_user_assigned_identity

    @primary_user_assigned_identity.setter
    def primary_user_assigned_identity(self, value: str):
        if False:
            i = 10
            return i + 15
        'Set the primary_user_assigned_identity of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.primary_user_assigned_identity = value

    @property
    def enable_data_isolation(self) -> str:
        if False:
            i = 10
            return i + 15
        'The enable_data_isolation of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.enable_data_isolation

    @property
    def default_workspace_resource_group(self) -> str:
        if False:
            while True:
                i = 10
        'The default_workspace_resource_group of the resource.\n\n        :return: Name of the resource.\n        :rtype: str\n        '
        return self._workspace_hub.workspace_hub_config.default_workspace_resource_group

    @default_workspace_resource_group.setter
    def default_workspace_resource_group(self, value: str):
        if False:
            print('Hello World!')
        'Set the default_workspace_resource_group of the resource.\n\n        :param value: The new type to assign to the resource.\n        :type value: str\n        '
        if not value:
            return
        self._workspace_hub.workspace_hub_config.default_workspace_resource_group = value