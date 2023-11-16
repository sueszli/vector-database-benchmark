from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy import func, select
from airflow.api_connexion import security
from airflow.api_connexion.exceptions import NotFound
from airflow.api_connexion.parameters import apply_sorting, check_limit, format_parameters
from airflow.api_connexion.schemas.error_schema import ImportErrorCollection, import_error_collection_schema, import_error_schema
from airflow.auth.managers.models.resource_details import DagAccessEntity
from airflow.models.errors import ImportError as ImportErrorModel
from airflow.utils.session import NEW_SESSION, provide_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.api_connexion.types import APIResponse

@security.requires_access_dag('GET', DagAccessEntity.IMPORT_ERRORS)
@provide_session
def get_import_error(*, import_error_id: int, session: Session=NEW_SESSION) -> APIResponse:
    if False:
        while True:
            i = 10
    'Get an import error.'
    error = session.get(ImportErrorModel, import_error_id)
    if error is None:
        raise NotFound('Import error not found', detail=f'The ImportError with import_error_id: `{import_error_id}` was not found')
    return import_error_schema.dump(error)

@security.requires_access_dag('GET', DagAccessEntity.IMPORT_ERRORS)
@format_parameters({'limit': check_limit})
@provide_session
def get_import_errors(*, limit: int, offset: int | None=None, order_by: str='import_error_id', session: Session=NEW_SESSION) -> APIResponse:
    if False:
        return 10
    'Get all import errors.'
    to_replace = {'import_error_id': 'id'}
    allowed_filter_attrs = ['import_error_id', 'timestamp', 'filename']
    total_entries = session.scalars(func.count(ImportErrorModel.id)).one()
    query = select(ImportErrorModel)
    query = apply_sorting(query, order_by, to_replace, allowed_filter_attrs)
    import_errors = session.scalars(query.offset(offset).limit(limit)).all()
    return import_error_collection_schema.dump(ImportErrorCollection(import_errors=import_errors, total_entries=total_entries))