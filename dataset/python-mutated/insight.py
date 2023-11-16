from typing import Optional
from posthog.clickhouse.query_tagging import tag_queries
from posthog.client import query_with_columns, sync_execute
from posthog.types import FilterType

def insight_sync_execute(query, args=None, *, team_id: int, query_type: str, filter: Optional['FilterType']=None, **kwargs):
    if False:
        print('Hello World!')
    tag_queries(team_id=team_id)
    _tag_query(query, query_type, filter)
    return sync_execute(query, args=args, team_id=team_id, **kwargs)

def insight_query_with_columns(query, args=None, *, query_type: str, filter: Optional['FilterType']=None, team_id: int, **kwargs):
    if False:
        i = 10
        return i + 15
    _tag_query(query, query_type, filter)
    return query_with_columns(query, args=args, team_id=team_id, **kwargs)

def _tag_query(query, query_type, filter: Optional['FilterType']):
    if False:
        i = 10
        return i + 15
    tag_queries(query_type=query_type, has_joins='JOIN' in query, has_json_operations='JSONExtract' in query or 'JSONHas' in query)
    if filter is not None:
        tag_queries(filter=filter.to_dict(), **filter.query_tags())