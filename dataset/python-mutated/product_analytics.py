from typing import List
import schemas
from chalicelib.core import metadata
from chalicelib.core.metrics import __get_constraints, __get_constraint_values
from chalicelib.utils import helper, dev
from chalicelib.utils import pg_client
from chalicelib.utils.TimeUTC import TimeUTC
from chalicelib.utils import sql_helper as sh
from time import time
import logging
logger = logging.getLogger(__name__)

def __transform_journey(rows, reverse_path=False):
    if False:
        i = 10
        return i + 15
    total_100p = 0
    number_of_step1 = 0
    for r in rows:
        if r['event_number_in_session'] > 1:
            break
        number_of_step1 += 1
        total_100p += r['sessions_count']
    for i in range(len(rows)):
        rows[i]['value'] = rows[i]['sessions_count'] * 100 / total_100p
    nodes = []
    nodes_values = []
    links = []
    for r in rows:
        source = f"{r['event_number_in_session']}_{r['event_type']}_{r['e_value']}"
        if source not in nodes:
            nodes.append(source)
            nodes_values.append({'name': r['e_value'], 'eventType': r['event_type'], 'avgTimeFromPrevious': 0, 'sessionsCount': 0})
        if r['next_value']:
            target = f"{r['event_number_in_session'] + 1}_{r['next_type']}_{r['next_value']}"
            if target not in nodes:
                nodes.append(target)
                nodes_values.append({'name': r['next_value'], 'eventType': r['next_type'], 'avgTimeFromPrevious': 0, 'sessionsCount': 0})
            sr_idx = nodes.index(source)
            tg_idx = nodes.index(target)
            if r['avg_time_from_previous'] is not None:
                nodes_values[tg_idx]['avgTimeFromPrevious'] += r['avg_time_from_previous'] * r['sessions_count']
                nodes_values[tg_idx]['sessionsCount'] += r['sessions_count']
            link = {'eventType': r['event_type'], 'sessionsCount': r['sessions_count'], 'value': r['value'], 'avgTimeFromPrevious': r['avg_time_from_previous']}
            if not reverse_path:
                link['source'] = sr_idx
                link['target'] = tg_idx
            else:
                link['source'] = tg_idx
                link['target'] = sr_idx
            links.append(link)
    for n in nodes_values:
        if n['sessionsCount'] > 0:
            n['avgTimeFromPrevious'] = n['avgTimeFromPrevious'] / n['sessionsCount']
        else:
            n['avgTimeFromPrevious'] = None
        n.pop('sessionsCount')
    return {'nodes': nodes_values, 'links': sorted(links, key=lambda x: (x['source'], x['target']), reverse=False)}
JOURNEY_TYPES = {schemas.ProductAnalyticsSelectedEventType.location: {'table': 'events.pages', 'column': 'path'}, schemas.ProductAnalyticsSelectedEventType.click: {'table': 'events.clicks', 'column': 'label'}, schemas.ProductAnalyticsSelectedEventType.input: {'table': 'events.inputs', 'column': 'label'}, schemas.ProductAnalyticsSelectedEventType.custom_event: {'table': 'events_common.customs', 'column': 'name'}}

def path_analysis(project_id: int, data: schemas.CardPathAnalysis):
    if False:
        for i in range(10):
            print('nop')
    sub_events = []
    start_points_from = 'pre_ranked_events'
    sub_sessions_extra_projection = ''
    start_points_conditions = []
    sessions_conditions = ['start_ts>=%(startTimestamp)s', 'start_ts<%(endTimestamp)s', 'project_id=%(project_id)s', 'events_count > 1', 'duration>0']
    if len(data.metric_value) == 0:
        data.metric_value.append(schemas.ProductAnalyticsSelectedEventType.location)
        sub_events.append({'table': JOURNEY_TYPES[schemas.ProductAnalyticsSelectedEventType.location]['table'], 'column': JOURNEY_TYPES[schemas.ProductAnalyticsSelectedEventType.location]['column'], 'eventType': schemas.ProductAnalyticsSelectedEventType.location.value})
    else:
        for v in data.metric_value:
            if JOURNEY_TYPES.get(v):
                sub_events.append({'table': JOURNEY_TYPES[v]['table'], 'column': JOURNEY_TYPES[v]['column'], 'eventType': v})
    extra_values = {}
    start_join = []
    reverse = data.start_type == 'end'
    for (i, sf) in enumerate(data.start_point):
        f_k = f'start_point_{i}'
        op = sh.get_sql_operator(sf.operator)
        sf.value = helper.values_for_operator(value=sf.value, op=sf.operator)
        is_not = sh.is_negation_operator(sf.operator)
        extra_values = {**extra_values, **sh.multi_values(sf.value, value_key=f_k)}
        start_points_conditions.append(f"(event_type='{sf.type}' AND " + sh.multi_conditions(f'e_value {op} %({f_k})s', sf.value, is_not=is_not, value_key=f_k) + ')')
        main_column = JOURNEY_TYPES[sf.type]['column']
        sessions_conditions.append(sh.multi_conditions(f'{main_column} {op} %({f_k})s', sf.value, is_not=is_not, value_key=f_k))
        sessions_conditions += ['timestamp>=%(startTimestamp)s', 'timestamp<%(endTimestamp)s']
        start_join.append(f"INNER JOIN {JOURNEY_TYPES[sf.type]['table']} USING (session_id)")
    exclusions = {}
    for (i, ef) in enumerate(data.excludes):
        if len(ef.value) == 0:
            continue
        if ef.type in data.metric_value:
            f_k = f'exclude_{i}'
            extra_values = {**extra_values, **sh.multi_values(ef.value, value_key=f_k)}
            exclusions[ef.type] = [sh.multi_conditions(f"{JOURNEY_TYPES[ef.type]['column']} != %({f_k})s", ef.value, is_not=True, value_key=f_k)]
    meta_keys = None
    for (i, f) in enumerate(data.series[0].filter.filters):
        op = sh.get_sql_operator(f.operator)
        is_any = sh.isAny_opreator(f.operator)
        is_not = sh.is_negation_operator(f.operator)
        is_undefined = sh.isUndefined_operator(f.operator)
        f_k = f'f_value_{i}'
        extra_values = {**extra_values, **sh.multi_values(f.value, value_key=f_k)}
        if not is_any and len(f.value) == 0:
            continue
        if f.type == schemas.FilterType.user_browser:
            if is_any:
                sessions_conditions.append('user_browser IS NOT NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'user_browser {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.user_os]:
            if is_any:
                sessions_conditions.append('user_os IS NOT NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'user_os {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.user_device]:
            if is_any:
                sessions_conditions.append('user_device IS NOT NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'user_device {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.user_country]:
            if is_any:
                sessions_conditions.append('user_country IS NOT NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'user_country {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type == schemas.FilterType.user_city:
            if is_any:
                sessions_conditions.append('user_city IS NOT NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'user_city {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type == schemas.FilterType.user_state:
            if is_any:
                sessions_conditions.append('user_state IS NOT NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'user_state {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.utm_source]:
            if is_any:
                sessions_conditions.append('utm_source IS NOT NULL')
            elif is_undefined:
                sessions_conditions.append('utm_source IS NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'utm_source {op} %({f_k})s::text', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.utm_medium]:
            if is_any:
                sessions_conditions.append('utm_medium IS NOT NULL')
            elif is_undefined:
                sessions_conditions.append('utm_medium IS NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'utm_medium {op} %({f_k})s::text', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.utm_campaign]:
            if is_any:
                sessions_conditions.append('utm_campaign IS NOT NULL')
            elif is_undefined:
                sessions_conditions.append('utm_campaign IS NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'utm_campaign {op} %({f_k})s::text', f.value, is_not=is_not, value_key=f_k))
        elif f.type == schemas.FilterType.duration:
            if len(f.value) > 0 and f.value[0] is not None:
                sessions_conditions.append('duration >= %(minDuration)s')
                extra_values['minDuration'] = f.value[0]
            if len(f.value) > 1 and f.value[1] is not None and (int(f.value[1]) > 0):
                sessions_conditions.append('duration <= %(maxDuration)s')
                extra_values['maxDuration'] = f.value[1]
        elif f.type == schemas.FilterType.referrer:
            if is_any:
                sessions_conditions.append('base_referrer IS NOT NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'base_referrer {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type == schemas.FilterType.metadata:
            if meta_keys is None:
                meta_keys = metadata.get(project_id=project_id)
                meta_keys = {m['key']: m['index'] for m in meta_keys}
            if f.source in meta_keys.keys():
                if is_any:
                    sessions_conditions.append(f'{metadata.index_to_colname(meta_keys[f.source])} IS NOT NULL')
                elif is_undefined:
                    sessions_conditions.append(f'{metadata.index_to_colname(meta_keys[f.source])} IS NULL')
                else:
                    sessions_conditions.append(sh.multi_conditions(f'{metadata.index_to_colname(meta_keys[f.source])} {op} %({f_k})s::text', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.user_id, schemas.FilterType.user_id_ios]:
            if is_any:
                sessions_conditions.append('user_id IS NOT NULL')
            elif is_undefined:
                sessions_conditions.append('user_id IS NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f's.user_id {op} %({f_k})s::text', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.user_anonymous_id, schemas.FilterType.user_anonymous_id_ios]:
            if is_any:
                sessions_conditions.append('user_anonymous_id IS NOT NULL')
            elif is_undefined:
                sessions_conditions.append('user_anonymous_id IS NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'user_anonymous_id {op} %({f_k})s::text', f.value, is_not=is_not, value_key=f_k))
        elif f.type in [schemas.FilterType.rev_id, schemas.FilterType.rev_id_ios]:
            if is_any:
                sessions_conditions.append('rev_id IS NOT NULL')
            elif is_undefined:
                sessions_conditions.append('rev_id IS NULL')
            else:
                sessions_conditions.append(sh.multi_conditions(f'rev_id {op} %({f_k})s::text', f.value, is_not=is_not, value_key=f_k))
        elif f.type == schemas.FilterType.platform:
            sessions_conditions.append(sh.multi_conditions(f'user_device_type {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
        elif f.type == schemas.FilterType.issue:
            if is_any:
                sessions_conditions.append('array_length(issue_types, 1) > 0')
            else:
                sessions_conditions.append(sh.multi_conditions(f'%({f_k})s {op} ANY (issue_types)', f.value, is_not=is_not, value_key=f_k))
        elif f.type == schemas.FilterType.events_count:
            sessions_conditions.append(sh.multi_conditions(f'events_count {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
    events_subquery = []
    for t in sub_events:
        sub_events_conditions = ['e.timestamp >= %(startTimestamp)s', 'e.timestamp < %(endTimestamp)s'] + exclusions.get(t['eventType'], [])
        if len(start_points_conditions) > 0:
            sub_events_conditions.append('e.timestamp >= sub_sessions.start_event_timestamp')
        events_subquery.append(f"                   SELECT session_id, {t['column']} AS e_value, timestamp, '{t['eventType']}' AS event_type\n                   FROM {t['table']} AS e INNER JOIN sub_sessions USING (session_id)\n                   WHERE {' AND '.join(sub_events_conditions)}")
    events_subquery = '\n UNION ALL \n'.join(events_subquery)
    if reverse:
        path_direction = 'DESC'
    else:
        path_direction = ''
    if len(start_points_conditions) == 0:
        start_points_from = '(SELECT event_type, e_value\n                                FROM pre_ranked_events\n                                WHERE event_number_in_session = 1\n                                GROUP BY event_type, e_value\n                                ORDER BY count(1) DESC\n                                LIMIT 1) AS top_start_events\n                                   INNER JOIN pre_ranked_events\n                                              USING (event_type, e_value)'
    else:
        sub_sessions_extra_projection = ', MIN(timestamp) AS start_event_timestamp'
        start_points_conditions = ['(' + ' OR '.join(start_points_conditions) + ')']
    start_points_conditions.append('event_number_in_session = 1')
    steps_query = ['n1 AS (SELECT event_number_in_session,\n                                    event_type,\n                                    e_value,\n                                    next_type,\n                                    next_value,\n                                    AVG(time_from_previous) AS avg_time_from_previous,\n                                    COUNT(1) AS sessions_count\n                             FROM ranked_events INNER JOIN start_points USING (session_id)\n                             WHERE event_number_in_session = 1 \n                                AND next_value IS NOT NULL\n                             GROUP BY event_number_in_session, event_type, e_value, next_type, next_value\n                             ORDER BY sessions_count DESC\n                             LIMIT %(eventThresholdNumberInGroup)s)']
    projection_query = ['(SELECT event_number_in_session,\n                                   event_type,\n                                   e_value,\n                                   next_type,\n                                   next_value,\n                                   sessions_count,\n                                   avg_time_from_previous\n                           FROM n1)']
    for i in range(2, data.density + 1):
        steps_query.append(f'n{i} AS (SELECT *\n                                      FROM (SELECT re.event_number_in_session,\n                                                   re.event_type,\n                                                   re.e_value,\n                                                   re.next_type,\n                                                   re.next_value,\n                                                   AVG(re.time_from_previous) AS avg_time_from_previous,\n                                                   COUNT(1) AS sessions_count\n                                            FROM ranked_events AS re\n                                                     INNER JOIN n{i - 1} ON (n{i - 1}.next_value = re.e_value)\n                                            WHERE re.event_number_in_session = {i}\n                                            GROUP BY re.event_number_in_session, re.event_type, re.e_value, re.next_type, re.next_value) AS sub_level\n                                      ORDER BY sessions_count DESC\n                                      LIMIT %(eventThresholdNumberInGroup)s)')
        projection_query.append(f'(SELECT event_number_in_session,\n                                            event_type,\n                                            e_value,\n                                            next_type,\n                                            next_value,\n                                            sessions_count,\n                                            avg_time_from_previous\n                                     FROM n{i})')
    with pg_client.PostgresClient() as cur:
        pg_query = f"WITH sub_sessions AS (SELECT session_id {sub_sessions_extra_projection}\n                      FROM public.sessions {' '.join(start_join)}\n                      WHERE {' AND '.join(sessions_conditions)}\n                      {('GROUP BY session_id' if len(start_points_conditions) > 0 else '')}),\n     sub_events AS ({events_subquery}),\n     pre_ranked_events AS (SELECT *\n                           FROM (SELECT session_id,\n                                        event_type,\n                                        e_value,\n                                        timestamp,\n                                        row_number() OVER (PARTITION BY session_id ORDER BY timestamp {path_direction}) AS event_number_in_session\n                                 FROM sub_events\n                                 ORDER BY session_id) AS full_ranked_events\n                           WHERE event_number_in_session <= %(density)s),\n     start_points AS (SELECT session_id\n                      FROM {start_points_from}\n                      WHERE {' AND '.join(start_points_conditions)}),\n     ranked_events AS (SELECT *,\n                              LEAD(e_value, 1) OVER (PARTITION BY session_id ORDER BY timestamp {path_direction})    AS next_value,\n                              LEAD(event_type, 1) OVER (PARTITION BY session_id ORDER BY timestamp {path_direction}) AS next_type,\n                              abs(LAG(timestamp, 1) OVER (PARTITION BY session_id ORDER BY timestamp {path_direction}) -\n                                  timestamp)                                                         AS time_from_previous\n                       FROM pre_ranked_events INNER JOIN start_points USING (session_id)),\n     {','.join(steps_query)}\n{'UNION ALL'.join(projection_query)};"
        params = {'project_id': project_id, 'startTimestamp': data.startTimestamp, 'endTimestamp': data.endTimestamp, 'density': data.density, 'eventThresholdNumberInGroup': 4 if data.hide_excess else 8, **extra_values}
        query = cur.mogrify(pg_query, params)
        _now = time()
        cur.execute(query)
        if time() - _now > 2:
            logger.warning(f'>>>>>>>>>PathAnalysis long query ({int(time() - _now)}s)<<<<<<<<<')
            logger.warning('----------------------')
            logger.warning(query)
            logger.warning('----------------------')
        rows = cur.fetchall()
    return __transform_journey(rows=rows, reverse_path=reverse)