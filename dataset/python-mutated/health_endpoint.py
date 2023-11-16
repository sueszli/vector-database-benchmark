from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.api.common.airflow_health import get_airflow_health
from airflow.api_connexion.schemas.health_schema import health_schema
if TYPE_CHECKING:
    from airflow.api_connexion.types import APIResponse

def get_health() -> APIResponse:
    if False:
        while True:
            i = 10
    'Return the health of the airflow scheduler, metadatabase and triggerer.'
    airflow_health_status = get_airflow_health()
    return health_schema.dump(airflow_health_status)