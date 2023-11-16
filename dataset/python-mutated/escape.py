from typing import Any
from clickhouse_driver.connection import ServerInfo
from clickhouse_driver.context import Context
from clickhouse_driver.util.escape import escape_param

def substitute_params(query, params):
    if False:
        while True:
            i = 10
    "\n    This is a copy of clickhouse-driver's `substitute_params` function without\n    the dependency that you need to connect to the server before you can escape\n    params. There was a bug in which we were trying to substitute params before\n    the connection was established, which caused the query to fail. Presumably\n    this was on initial worker startup only.\n\n    It seems somewhat unusual that you need to connect to the server before\n    you can escape params, so we're just going to copy the function here\n    and remove that dependency.\n\n    See\n    https://github.com/mymarilyn/clickhouse-driver/blob/87090902f0270ed51a0b6754d5cbf0dc8544ec4b/clickhouse_driver/client.py#L593\n    for the original function.\n    "
    if not isinstance(params, dict):
        raise ValueError('Parameters are expected in dict form')
    escaped = escape_params(params)
    return query % escaped

def escape_params(params):
    if False:
        while True:
            i = 10
    "\n    This is a copy of clickhouse-driver's `escape_params` function without the\n    dependency that you need to connect to the server before you can escape\n    params.\n\n    See\n    https://github.com/mymarilyn/clickhouse-driver/blob/87090902f0270ed51a0b6754d5cbf0dc8544ec4b/clickhouse_driver/util/escape.py#L60\n    for the original function.\n    "
    escaped = {}
    for (key, value) in params.items():
        escaped[key] = escape_param_for_clickhouse(value)
    return escaped

def escape_param_for_clickhouse(param: Any) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    This is a wrapper around the `escape_param` function from the\n    `clickhouse-driver` package, but passes a placeholder `Context` object to it\n    just such that it can run. The only value that the real `escape_param` uses\n    from the context is the server timezone. We assume that the server timezone\n    is UTC.\n\n    See\n    https://github.com/mymarilyn/clickhouse-driver/blob/87090902f0270ed51a0b6754d5cbf0dc8544ec4b/clickhouse_driver/util/escape.py#L31\n    for the wrapped function.\n    '
    context = Context()
    context.server_info = ServerInfo(name='placeholder server_info value', version_major='placeholder server_info value', version_minor='placeholder server_info value', version_patch='placeholder server_info value', revision='placeholder server_info value', display_name='placeholder server_info value', timezone='UTC')
    return escape_param(param, context=context)