from django.conf import settings

def dictionary_source_clickhouse(table: str) -> str:
    if False:
        return 10
    '\n    Returns\n    '
    connection_settings = f"TABLE {table} DB '{settings.CLICKHOUSE_DATABASE}'"
    if settings.CLICKHOUSE_USER:
        connection_settings += f" USER '{settings.CLICKHOUSE_USER}'"
    if settings.CLICKHOUSE_PASSWORD:
        connection_settings += f" PASSWORD '{settings.CLICKHOUSE_PASSWORD}'"
    return f'SOURCE(CLICKHOUSE({connection_settings}))'