import datetime
from django.db import connection
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Literal
from zilencer.models import RemoteInstallationCount, RemoteZulipServer

class MissingDataError(Exception):
    pass

def compute_max_monthly_messages(remote_server: RemoteZulipServer) -> int:
    if False:
        return 10
    if not RemoteInstallationCount.objects.filter(server=remote_server, property='active_users_audit:is_bot:day', end_time__lte=timezone_now() - datetime.timedelta(days=3)).exists():
        raise MissingDataError
    query = SQL("\n    WITH server_message_stats_daily AS -- Up to 4 rows per day for different subgroups\n    (\n        SELECT\n            r.end_time,\n            r.value AS message_count\n        FROM\n            zilencer_remoteinstallationcount r\n        WHERE\n            r.property = 'messages_sent:message_type:day'\n            AND end_time >= CURRENT_TIMESTAMP(0) - INTERVAL '90 days'\n            AND r.server_id = {server_id}\n    ),\n    server_message_stats_monthly AS (\n        SELECT\n            CASE\n                WHEN current_timestamp(0) - end_time <= INTERVAL '30 days' THEN 0\n                WHEN current_timestamp(0) - end_time <= INTERVAL '60 days' THEN 1\n                WHEN current_timestamp(0) - end_time <= INTERVAL '90 days' THEN 2\n            END AS billing_month,\n            SUM(message_count) AS message_count\n        FROM\n            server_message_stats_daily\n        GROUP BY\n            1\n    ),\n    server_max_monthly_messages AS (\n        SELECT\n            MAX(message_count) AS message_count\n        FROM\n            server_message_stats_monthly\n        WHERE\n            billing_month IS NOT NULL\n    )\n    SELECT\n        -- Return zeros, rather than nulls,\n        -- for reporting servers with zero messages.\n        COALESCE(server_max_monthly_messages.message_count, 0) AS message_count\n    FROM\n        server_max_monthly_messages;\n        ").format(server_id=Literal(remote_server.id))
    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()[0]
    return int(result)