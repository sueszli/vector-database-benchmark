from __future__ import annotations
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Container, Literal
from sqlalchemy import select
from airflow.auth.managers.models.resource_details import DagDetails
from airflow.exceptions import AirflowException
from airflow.models import DagModel
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.session import NEW_SESSION, provide_session
if TYPE_CHECKING:
    from connexion import FlaskApi
    from flask import Flask
    from sqlalchemy.orm import Session
    from airflow.auth.managers.models.base_user import BaseUser
    from airflow.auth.managers.models.resource_details import AccessView, ConfigurationDetails, ConnectionDetails, DagAccessEntity, DatasetDetails, PoolDetails, VariableDetails
    from airflow.cli.cli_config import CLICommand
    from airflow.www.extensions.init_appbuilder import AirflowAppBuilder
    from airflow.www.security_manager import AirflowSecurityManagerV2
ResourceMethod = Literal['GET', 'POST', 'PUT', 'DELETE']

class BaseAuthManager(LoggingMixin):
    """
    Class to derive in order to implement concrete auth managers.

    Auth managers are responsible for any user management related operation such as login, logout, authz, ...
    """

    def __init__(self, app: Flask, appbuilder: AirflowAppBuilder) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.app = app
        self.appbuilder = appbuilder

    @staticmethod
    def get_cli_commands() -> list[CLICommand]:
        if False:
            return 10
        'Vends CLI commands to be included in Airflow CLI.\n\n        Override this method to expose commands via Airflow CLI to manage this auth manager.\n        '
        return []

    def get_api_endpoints(self) -> None | FlaskApi:
        if False:
            return 10
        'Return API endpoint(s) definition for the auth manager.'
        return None

    @abstractmethod
    def get_user_name(self) -> str:
        if False:
            print('Hello World!')
        'Return the username associated to the user in session.'

    @abstractmethod
    def get_user_display_name(self) -> str:
        if False:
            while True:
                i = 10
        "Return the user's display name associated to the user in session."

    @abstractmethod
    def get_user(self) -> BaseUser:
        if False:
            for i in range(10):
                print('nop')
        'Return the user associated to the user in session.'

    @abstractmethod
    def get_user_id(self) -> str:
        if False:
            print('Hello World!')
        'Return the user ID associated to the user in session.'

    def init(self) -> None:
        if False:
            return 10
        '\n        Run operations when Airflow is initializing.\n\n        By default, do nothing.\n        '

    @abstractmethod
    def is_logged_in(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return whether the user is logged in.'

    @abstractmethod
    def is_authorized_configuration(self, *, method: ResourceMethod, details: ConfigurationDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether the user is authorized to perform a given action on configuration.\n\n        :param method: the method to perform\n        :param details: optional details about the configuration\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    @abstractmethod
    def is_authorized_cluster_activity(self, *, method: ResourceMethod, user: BaseUser | None=None) -> bool:
        if False:
            return 10
        '\n        Return whether the user is authorized to perform a given action on the cluster activity.\n\n        :param method: the method to perform\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    @abstractmethod
    def is_authorized_connection(self, *, method: ResourceMethod, details: ConnectionDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether the user is authorized to perform a given action on a connection.\n\n        :param method: the method to perform\n        :param details: optional details about the connection\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    @abstractmethod
    def is_authorized_dag(self, *, method: ResourceMethod, access_entity: DagAccessEntity | None=None, details: DagDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return whether the user is authorized to perform a given action on a DAG.\n\n        :param method: the method to perform\n        :param access_entity: the kind of DAG information the authorization request is about.\n            If not provided, the authorization request is about the DAG itself\n        :param details: optional details about the DAG\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    @abstractmethod
    def is_authorized_dataset(self, *, method: ResourceMethod, details: DatasetDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return whether the user is authorized to perform a given action on a dataset.\n\n        :param method: the method to perform\n        :param details: optional details about the dataset\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    @abstractmethod
    def is_authorized_pool(self, *, method: ResourceMethod, details: PoolDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            while True:
                i = 10
        '\n        Return whether the user is authorized to perform a given action on a pool.\n\n        :param method: the method to perform\n        :param details: optional details about the pool\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    @abstractmethod
    def is_authorized_variable(self, *, method: ResourceMethod, details: VariableDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            return 10
        '\n        Return whether the user is authorized to perform a given action on a variable.\n\n        :param method: the method to perform\n        :param details: optional details about the variable\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    @abstractmethod
    def is_authorized_view(self, *, access_view: AccessView, user: BaseUser | None=None) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether the user is authorized to access a read-only state of the installation.\n\n        :param access_view: the specific read-only view/state the authorization request is about.\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '

    def is_authorized_custom_view(self, *, fab_action_name: str, fab_resource_name: str, user: BaseUser | None=None):
        if False:
            i = 10
            return i + 15
        '\n        Return whether the user is authorized to perform a given action on a custom view.\n\n        A custom view is a view defined as part of the auth manager. This view is then only available when\n        the auth manager is used as part of the environment.\n\n        By default, it throws an exception because auth managers do not define custom views by default.\n        If an auth manager defines some custom views, it needs to override this method.\n\n        :param fab_action_name: the name of the FAB action defined in the view in ``base_permissions``\n        :param fab_resource_name: the name of the FAB resource defined in the view in\n            ``class_permission_name``\n        :param user: the user to perform the action on. If not provided (or None), it uses the current user\n        '
        raise AirflowException(f'The resource `{fab_resource_name}` does not exist in the environment.')

    @provide_session
    def get_permitted_dag_ids(self, *, methods: Container[ResourceMethod] | None=None, user=None, session: Session=NEW_SESSION) -> set[str]:
        if False:
            while True:
                i = 10
        '\n        Get readable or writable DAGs for user.\n\n        By default, reads all the DAGs and check individually if the user has permissions to access the DAG.\n        Can lead to some poor performance. It is recommended to override this method in the auth manager\n        implementation to provide a more efficient implementation.\n        '
        if not methods:
            methods = ['PUT', 'GET']
        dag_ids = {dag.dag_id for dag in session.execute(select(DagModel.dag_id))}
        if 'GET' in methods and self.is_authorized_dag(method='GET', user=user) or ('PUT' in methods and self.is_authorized_dag(method='PUT', user=user)):
            return dag_ids

        def _is_permitted_dag_id(method: ResourceMethod, methods: Container[ResourceMethod], dag_id: str):
            if False:
                return 10
            return method in methods and self.is_authorized_dag(method=method, details=DagDetails(id=dag_id), user=user)
        return {dag_id for dag_id in dag_ids if _is_permitted_dag_id('GET', methods, dag_id) or _is_permitted_dag_id('PUT', methods, dag_id)}

    @abstractmethod
    def get_url_login(self, **kwargs) -> str:
        if False:
            return 10
        'Return the login page url.'

    @abstractmethod
    def get_url_logout(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the logout page url.'

    @abstractmethod
    def get_url_user_profile(self) -> str | None:
        if False:
            while True:
                i = 10
        'Return the url to a page displaying info about the current user.'

    @cached_property
    def security_manager(self) -> AirflowSecurityManagerV2:
        if False:
            while True:
                i = 10
        '\n        Return the security manager.\n\n        By default, Airflow comes with the default security manager\n        airflow.www.security_manager.AirflowSecurityManagerV2. The auth manager might need to extend this\n        default security manager for its own purposes.\n\n        By default, return the default AirflowSecurityManagerV2.\n        '
        from airflow.www.security_manager import AirflowSecurityManagerV2
        return AirflowSecurityManagerV2(self.appbuilder)