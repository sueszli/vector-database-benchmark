from __future__ import annotations
from typing import Any, Mapping, cast

def get_json_request_dict() -> Mapping[str, Any]:
    if False:
        return 10
    'Cast request dictionary to JSON.'
    from flask import request
    return cast(Mapping[str, Any], request.get_json())