from posthog.clickhouse.client import sync_execute

def remove_deleted_person_data(mutations_sync=False):
    if False:
        for i in range(10):
            print('nop')
    settings = {'mutations_sync': 1 if mutations_sync else 0}
    sync_execute('\n        ALTER TABLE person\n        DELETE WHERE id IN (SELECT id FROM person WHERE is_deleted > 0)\n        ', settings=settings)