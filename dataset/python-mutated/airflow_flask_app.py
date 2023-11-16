from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
from flask import Flask
if TYPE_CHECKING:
    from airflow.models.dagbag import DagBag
    from airflow.www.extensions.init_appbuilder import AirflowAppBuilder

class AirflowApp(Flask):
    """Airflow Flask Application."""
    appbuilder: AirflowAppBuilder
    dag_bag: DagBag
    api_auth: list[Any]

def get_airflow_app() -> AirflowApp:
    if False:
        i = 10
        return i + 15
    from flask import current_app
    return cast(AirflowApp, current_app)