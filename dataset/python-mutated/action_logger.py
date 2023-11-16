from __future__ import annotations

def action_event_from_permission(prefix: str, permission: str) -> str:
    if False:
        while True:
            i = 10
    if permission.startswith('can_'):
        permission = permission[4:]
    if prefix:
        return f'{prefix}.{permission}'
    return permission