from posthog.client import sync_execute
from posthog.settings import CLICKHOUSE_DATABASE

def analyze_enough_disk_space_free_for_table(table_name: str, required_ratio: float):
    if False:
        while True:
            i = 10
    "\n    Analyzes whether there's enough disk space free for given async migration operation.\n\n    This is done by checking whether there's at least ratio times space free to resize table_name with.\n    "
    (current_ratio, _, required_space_pretty) = sync_execute(f"\n        WITH (\n            SELECT free_space\n            FROM system.disks WHERE name = 'default'\n        ) AS free_disk_space,(\n            SELECT total_space\n            FROM system.disks WHERE name = 'default'\n        ) AS total_disk_space,(\n            SELECT sum(bytes) as size\n            FROM system.parts\n            WHERE table = %(table_name)s AND database = %(database)s\n        ) AS table_size\n        SELECT\n            free_disk_space / greatest(table_size, 1),\n            total_disk_space - (free_disk_space - %(ratio)s * table_size) AS required,\n            formatReadableSize(required)\n        ", {'database': CLICKHOUSE_DATABASE, 'table_name': table_name, 'ratio': required_ratio})[0]
    if current_ratio >= required_ratio:
        return (True, None)
    else:
        return (False, f'Upgrade your ClickHouse storage to at least {required_space_pretty}.')