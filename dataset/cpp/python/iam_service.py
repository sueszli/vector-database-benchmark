from dataclasses import dataclass

from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.authorization.v2022_04_01.models import Permission

from prowler.lib.logger import logger
from prowler.providers.azure.lib.service.service import AzureService


########################## IAM
class IAM(AzureService):
    def __init__(self, audit_info):
        super().__init__(AuthorizationManagementClient, audit_info)
        self.roles = self.__get_roles__()

    def __get_roles__(self):
        logger.info("IAM - Getting roles...")
        roles = {}
        for subscription, client in self.clients.items():
            try:
                roles.update({subscription: []})
                for role in client.role_definitions.list(
                    scope=f"/subscriptions/{self.subscriptions[subscription]}",
                    filter="type eq 'CustomRole'",
                ):
                    roles[subscription].append(
                        Role(
                            id=role.id,
                            name=role.role_name,
                            type=role.role_type,
                            assignable_scopes=role.assignable_scopes,
                            permissions=role.permissions,
                        )
                    )
            except Exception as error:
                logger.error(f"Subscription name: {subscription}")
                logger.error(
                    f"{error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}"
                )
        return roles


@dataclass
class Role:
    id: str
    name: str
    type: str
    assignable_scopes: list[str]
    permissions: list[Permission]

    def __init__(self, id, name, type, assignable_scopes, permissions):
        self.id = id
        self.name = name
        self.type = type
        self.assignable_scopes = assignable_scopes
        self.permissions = permissions
