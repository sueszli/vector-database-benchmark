import schemas
from chalicelib.core import sessions_mobs, sessions_legacy as sessions_search, events
from chalicelib.utils import pg_client, helper
SESSION_PROJECTION_COLS = 's.project_id,\ns.session_id::text AS session_id,\ns.user_uuid,\ns.user_id,\ns.user_os,\ns.user_browser,\ns.user_device,\ns.user_device_type,\ns.user_country,\ns.start_ts,\ns.duration,\ns.events_count,\ns.pages_count,\ns.errors_count,\ns.user_anonymous_id,\ns.platform,\ns.issue_score,\nto_jsonb(s.issue_types) AS issue_types,\nfavorite_sessions.session_id NOTNULL            AS favorite,\nCOALESCE((SELECT TRUE\n FROM public.user_viewed_sessions AS fs\n WHERE s.session_id = fs.session_id\n   AND fs.user_id = %(userId)s LIMIT 1), FALSE) AS viewed '

def search_short_session(data: schemas.ClickMapSessionsSearch, project_id, user_id, include_mobs: bool=True):
    if False:
        print('Hello World!')
    no_platform = True
    for f in data.filters:
        if f.type == schemas.FilterType.platform:
            no_platform = False
            break
    if no_platform:
        data.filters.append(schemas.SessionSearchFilterSchema(type=schemas.FilterType.platform, value=[schemas.PlatformType.desktop], operator=schemas.SearchEventOperator._is))
    (full_args, query_part) = sessions_search.search_query_parts(data=data, error_status=None, errors_only=False, favorite_only=data.bookmarked, issue=None, project_id=project_id, user_id=user_id)
    with pg_client.PostgresClient() as cur:
        data.order = schemas.SortOrderType.desc
        data.sort = 'duration'
        meta_keys = []
        main_query = cur.mogrify(f"""SELECT {SESSION_PROJECTION_COLS}\n                                                {(',' if len(meta_keys) > 0 else '')}{','.join([f"metadata_{m['index']}" for m in meta_keys])}\n                                     {query_part}\n                                     ORDER BY {data.sort} {data.order.value}\n                                     LIMIT 1;""", full_args)
        try:
            cur.execute(main_query)
        except Exception as err:
            print('--------- CLICK MAP SHORT SESSION SEARCH QUERY EXCEPTION -----------')
            print(main_query.decode('UTF-8'))
            print('--------- PAYLOAD -----------')
            print(data.model_dump_json())
            print('--------------------')
            raise err
        session = cur.fetchone()
    if session:
        if include_mobs:
            session['domURL'] = sessions_mobs.get_urls(session_id=session['session_id'], project_id=project_id)
            session['mobsUrl'] = sessions_mobs.get_urls_depercated(session_id=session['session_id'])
        session['events'] = events.get_by_session_id(project_id=project_id, session_id=session['session_id'], event_type=schemas.EventType.location)
    return helper.dict_to_camel_case(session)