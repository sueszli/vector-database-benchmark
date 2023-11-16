from __future__ import annotations
from typing import TYPE_CHECKING
from flask import g
from sqlalchemy import select
from airflow.api_connexion import security
from airflow.api_connexion.parameters import apply_sorting, check_limit, format_parameters
from airflow.api_connexion.schemas.dag_warning_schema import DagWarningCollection, dag_warning_collection_schema
from airflow.auth.managers.models.resource_details import DagAccessEntity
from airflow.models.dagwarning import DagWarning as DagWarningModel
from airflow.utils.airflow_flask_app import get_airflow_app
from airflow.utils.db import get_query_count
from airflow.utils.session import NEW_SESSION, provide_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.api_connexion.types import APIResponse

@security.requires_access_dag('GET', DagAccessEntity.WARNING)
@format_parameters({'limit': check_limit})
@provide_session
def get_dag_warnings(*, limit: int, dag_id: str | None=None, warning_type: str | None=None, offset: int | None=None, order_by: str='timestamp', session: Session=NEW_SESSION) -> APIResponse:
    if False:
        return 10
    'Get DAG warnings.\n\n    :param dag_id: the dag_id to optionally filter by\n    :param warning_type: the warning type to optionally filter by\n    '
    allowed_filter_attrs = ['dag_id', 'warning_type', 'message', 'timestamp']
    query = select(DagWarningModel)
    if dag_id:
        query = query.where(DagWarningModel.dag_id == dag_id)
    else:
        readable_dags = get_airflow_app().appbuilder.sm.get_accessible_dag_ids(g.user)
        query = query.where(DagWarningModel.dag_id.in_(readable_dags))
    if warning_type:
        query = query.where(DagWarningModel.warning_type == warning_type)
    total_entries = get_query_count(query, session=session)
    query = apply_sorting(query=query, order_by=order_by, allowed_attrs=allowed_filter_attrs)
    dag_warnings = session.scalars(query.offset(offset).limit(limit)).all()
    return dag_warning_collection_schema.dump(DagWarningCollection(dag_warnings=dag_warnings, total_entries=total_entries))