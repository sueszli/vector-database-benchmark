from __future__ import annotations
from typing import Any
custom_data_store: dict[int, Any] = {}

def ExtensionCustomAction(data, keep_app_open=False):
    if False:
        i = 10
        return i + 15
    ref = id(data)
    custom_data_store[ref] = data
    return {'type': 'event:activate_custom', 'ref': ref, 'keep_app_open': keep_app_open}