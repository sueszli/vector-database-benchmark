import json
from dagster import AssetKey, DagsterEventType, EventRecordsFilter, RunRequest, SensorDefinition, sensor

def make_hn_tables_updated_sensor(job) -> SensorDefinition:
    if False:
        for i in range(10):
            print('nop')
    'Returns a sensor that launches the given job when the HN "comments" and "stories" tables have\n    both been updated.\n    '

    @sensor(name=f'{job.name}_on_hn_tables_updated', job=job)
    def hn_tables_updated_sensor(context):
        if False:
            i = 10
            return i + 15
        cursor_dict = json.loads(context.cursor) if context.cursor else {}
        comments_cursor = cursor_dict.get('comments')
        stories_cursor = cursor_dict.get('stories')
        comments_event_records = context.instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION, asset_key=AssetKey(['snowflake', 'core', 'comments']), after_cursor=comments_cursor), ascending=False, limit=1)
        stories_event_records = context.instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION, asset_key=AssetKey(['snowflake', 'core', 'stories']), after_cursor=stories_cursor), ascending=False, limit=1)
        if not comments_event_records or not stories_event_records:
            return
        yield RunRequest(run_key=None)
        context.update_cursor(json.dumps({'comments': comments_event_records[0].storage_id, 'stories': stories_event_records[0].storage_id}))
    return hn_tables_updated_sensor