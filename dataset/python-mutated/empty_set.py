from __future__ import annotations

def _get_empty_set_for_configuration() -> set[tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    '\n    Retrieve an empty_set_for_configuration.\n\n    This method is only needed because configuration module has a deprecated method called set, and it\n    confuses mypy. This method will be removed when we remove the deprecated method.\n\n    :meta private:\n    :return: empty set\n    '
    return set()