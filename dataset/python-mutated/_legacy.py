from collections.abc import Mapping, Sequence
from pyramid.threadlocal import get_current_request

def _adapt_v1(data: Mapping) -> Sequence | None:
    if False:
        return 10
    permissions = data.get('permissions')
    if permissions is None:
        return None
    if permissions == 'user':
        request = get_current_request()
        if request is None:
            return None
        if request.user is None:
            return None
        return [3, str(request.user.id)]
    elif isinstance(permissions, Mapping) and 'projects' in permissions:
        return [1, permissions['projects']]
    return None

def _adapt_expiry(data: Mapping) -> Sequence | None:
    if False:
        while True:
            i = 10
    return [0, data['exp'], data['nbf']]

def _adapt_project_ids(data: Mapping) -> Sequence | None:
    if False:
        i = 10
        return i + 15
    return [2, data['project_ids']]

def adapt(data: Mapping) -> Sequence | None:
    if False:
        while True:
            i = 10
    if data.get('version') == 1:
        return _adapt_v1(data)
    if 'exp' in data and 'nbf' in data:
        return _adapt_expiry(data)
    if 'project_ids' in data:
        return _adapt_project_ids(data)
    return None