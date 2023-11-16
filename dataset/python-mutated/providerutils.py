"""Utilities for providers."""
import logging
logger = logging.getLogger(__name__)

def filter_backends(backends, filters=None, **kwargs):
    if False:
        return 10
    'Return the backends matching the specified filtering.\n\n    Filter the `backends` list by their `configuration` or `status`\n    attributes, or from a boolean callable. The criteria for filtering can\n    be specified via `**kwargs` or as a callable via `filters`, and the\n    backends must fulfill all specified conditions.\n\n    Args:\n        backends (list[Backend]): list of backends.\n        filters (callable): filtering conditions as a callable.\n        **kwargs: dict of criteria.\n\n    Returns:\n        list[Backend]: a list of backend instances matching the\n            conditions.\n    '

    def _match_all(obj, criteria):
        if False:
            return 10
        'Return True if all items in criteria matches items in obj.'
        return all((getattr(obj, key_, None) == value_ for (key_, value_) in criteria.items()))
    configuration_filters = {}
    status_filters = {}
    for (key, value) in kwargs.items():
        if all((key in backend.configuration() for backend in backends)):
            configuration_filters[key] = value
        else:
            status_filters[key] = value
    if configuration_filters:
        backends = [b for b in backends if _match_all(b.configuration(), configuration_filters)]
    if status_filters:
        backends = [b for b in backends if _match_all(b.status(), status_filters)]
    backends = list(filter(filters, backends))
    return backends

def resolve_backend_name(name, backends, deprecated, aliased):
    if False:
        for i in range(10):
            print('nop')
    'Resolve backend name from a deprecated name or an alias.\n\n    A group will be resolved in order of member priorities, depending on\n    availability.\n\n    Args:\n        name (str): name of backend to resolve\n        backends (list[Backend]): list of available backends.\n        deprecated (dict[str: str]): dict of deprecated names.\n        aliased (dict[str: list[str]]): dict of aliased names.\n\n    Returns:\n        str: resolved name (name of an available backend)\n\n    Raises:\n        LookupError: if name cannot be resolved through regular available\n            names, nor deprecated, nor alias names.\n    '
    available = [backend.name() for backend in backends]
    resolved_name = deprecated.get(name, aliased.get(name, name))
    if isinstance(resolved_name, list):
        resolved_name = next((b for b in resolved_name if b in available), '')
    if resolved_name not in available:
        raise LookupError(f"backend '{name}' not found.")
    if name in deprecated:
        logger.warning("Backend '%s' is deprecated. Use '%s'.", name, resolved_name)
    return resolved_name