from __future__ import annotations
from typing import Any, Mapping, Sequence
from airflow.api_connexion.exceptions import BadRequest

def extract_update_mask_data(update_mask: Sequence[str], non_update_fields: list[str], data: Mapping[str, Any]) -> Mapping[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    extracted_data = {}
    for field in update_mask:
        field = field.strip()
        if field in data and field not in non_update_fields:
            extracted_data[field] = data[field]
        else:
            raise BadRequest(detail=f"'{field}' is unknown or cannot be updated.")
    return extracted_data