from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
if TYPE_CHECKING:
    from flask import Flask
    from airflow.auth.managers.base_auth_manager import BaseAuthManager
    from airflow.www.extensions.init_appbuilder import AirflowAppBuilder
auth_manager: BaseAuthManager | None = None

def get_auth_manager_cls() -> type[BaseAuthManager]:
    if False:
        while True:
            i = 10
    '\n    Return just the auth manager class without initializing it.\n\n    Useful to save execution time if only static methods need to be called.\n    '
    auth_manager_cls = conf.getimport(section='core', key='auth_manager')
    if not auth_manager_cls:
        raise AirflowConfigException('No auth manager defined in the config. Please specify one using section/key [core/auth_manager].')
    return auth_manager_cls

def init_auth_manager(app: Flask, appbuilder: AirflowAppBuilder) -> BaseAuthManager:
    if False:
        return 10
    '\n    Initialize the auth manager.\n\n    Import the user manager class and instantiate it.\n    '
    global auth_manager
    auth_manager_cls = get_auth_manager_cls()
    auth_manager = auth_manager_cls(app, appbuilder)
    return auth_manager

def get_auth_manager() -> BaseAuthManager:
    if False:
        i = 10
        return i + 15
    "Return the auth manager, provided it's been initialized before."
    if auth_manager is None:
        raise Exception('Auth Manager has not been initialized yet. The `init_auth_manager` method needs to be called first.')
    return auth_manager