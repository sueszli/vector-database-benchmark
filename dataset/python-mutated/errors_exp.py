import json
import schemas
from chalicelib.core import metrics, metadata
from chalicelib.core import errors_legacy
from chalicelib.core import sourcemaps, sessions
from chalicelib.utils import ch_client, metrics_helper, exp_ch_helper
from chalicelib.utils import pg_client, helper
from chalicelib.utils.TimeUTC import TimeUTC
from decouple import config

def _multiple_values(values, value_key='value'):
    if False:
        return 10
    query_values = {}
    if values is not None and isinstance(values, list):
        for i in range(len(values)):
            k = f'{value_key}_{i}'
            query_values[k] = values[i]
    return query_values

def __get_sql_operator(op: schemas.SearchEventOperator):
    if False:
        for i in range(10):
            print('nop')
    return {schemas.SearchEventOperator._is: '=', schemas.SearchEventOperator._is_any: 'IN', schemas.SearchEventOperator._on: '=', schemas.SearchEventOperator._on_any: 'IN', schemas.SearchEventOperator._is_not: '!=', schemas.SearchEventOperator._not_on: '!=', schemas.SearchEventOperator._contains: 'ILIKE', schemas.SearchEventOperator._not_contains: 'NOT ILIKE', schemas.SearchEventOperator._starts_with: 'ILIKE', schemas.SearchEventOperator._ends_with: 'ILIKE'}.get(op, '=')

def _isAny_opreator(op: schemas.SearchEventOperator):
    if False:
        return 10
    return op in [schemas.SearchEventOperator._on_any, schemas.SearchEventOperator._is_any]

def _isUndefined_operator(op: schemas.SearchEventOperator):
    if False:
        print('Hello World!')
    return op in [schemas.SearchEventOperator._is_undefined]

def __is_negation_operator(op: schemas.SearchEventOperator):
    if False:
        return 10
    return op in [schemas.SearchEventOperator._is_not, schemas.SearchEventOperator._not_on, schemas.SearchEventOperator._not_contains]

def _multiple_conditions(condition, values, value_key='value', is_not=False):
    if False:
        for i in range(10):
            print('nop')
    query = []
    for i in range(len(values)):
        k = f'{value_key}_{i}'
        query.append(condition.replace(value_key, k))
    return '(' + (' AND ' if is_not else ' OR ').join(query) + ')'

def get(error_id, family=False):
    if False:
        i = 10
        return i + 15
    if family:
        return get_batch([error_id])
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify('SELECT * FROM events.errors AS e INNER JOIN public.errors AS re USING(error_id) WHERE error_id = %(error_id)s;', {'error_id': error_id})
        cur.execute(query=query)
        result = cur.fetchone()
        if result is not None:
            result['stacktrace_parsed_at'] = TimeUTC.datetime_to_timestamp(result['stacktrace_parsed_at'])
        return helper.dict_to_camel_case(result)

def get_batch(error_ids):
    if False:
        i = 10
        return i + 15
    if len(error_ids) == 0:
        return []
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify('\n            WITH RECURSIVE error_family AS (\n                SELECT *\n                FROM public.errors\n                WHERE error_id IN %(error_ids)s\n                UNION\n                SELECT child_errors.*\n                FROM public.errors AS child_errors\n                         INNER JOIN error_family ON error_family.error_id = child_errors.parent_error_id OR error_family.parent_error_id = child_errors.error_id\n            )\n            SELECT *\n            FROM error_family;', {'error_ids': tuple(error_ids)})
        cur.execute(query=query)
        errors = cur.fetchall()
        for e in errors:
            e['stacktrace_parsed_at'] = TimeUTC.datetime_to_timestamp(e['stacktrace_parsed_at'])
        return helper.list_to_camel_case(errors)

def __flatten_sort_key_count_version(data, merge_nested=False):
    if False:
        print('Hello World!')
    if data is None:
        return []
    return sorted([{'name': f'{o[0][0][0]}@{v[0]}', 'count': v[1]} for o in data for v in o[2]], key=lambda o: o['count'], reverse=True) if merge_nested else [{'name': o[0][0][0], 'count': o[1][0][0]} for o in data]

def __transform_map_to_tag(data, key1, key2, requested_key):
    if False:
        for i in range(10):
            print('nop')
    result = []
    for i in data:
        if requested_key == 0 and i.get(key1) is None and (i.get(key2) is None):
            result.append({'name': 'all', 'count': int(i.get('count'))})
        elif requested_key == 1 and i.get(key1) is not None and (i.get(key2) is None):
            result.append({'name': i.get(key1), 'count': int(i.get('count'))})
        elif requested_key == 2 and i.get(key1) is not None and (i.get(key2) is not None):
            result.append({'name': i.get(key2), 'count': int(i.get('count'))})
    return result

def __flatten_sort_key_count(data):
    if False:
        i = 10
        return i + 15
    if data is None:
        return []
    return [{'name': o[0][0][0], 'count': o[1][0][0]} for o in data]

def __rearrange_chart_details(start_at, end_at, density, chart):
    if False:
        for i in range(10):
            print('nop')
    chart = list(chart)
    for i in range(len(chart)):
        chart[i] = {'timestamp': chart[i][0], 'count': chart[i][1]}
    chart = metrics.__complete_missing_steps(rows=chart, start_time=start_at, end_time=end_at, density=density, neutral={'count': 0})
    return chart

def __process_tags(row):
    if False:
        i = 10
        return i + 15
    return [{'name': 'browser', 'partitions': __flatten_sort_key_count_version(data=row.get('browsers_partition'))}, {'name': 'browser.ver', 'partitions': __flatten_sort_key_count_version(data=row.pop('browsers_partition'), merge_nested=True)}, {'name': 'OS', 'partitions': __flatten_sort_key_count_version(data=row.get('os_partition'))}, {'name': 'OS.ver', 'partitions': __flatten_sort_key_count_version(data=row.pop('os_partition'), merge_nested=True)}, {'name': 'device.family', 'partitions': __flatten_sort_key_count_version(data=row.get('device_partition'))}, {'name': 'device', 'partitions': __flatten_sort_key_count_version(data=row.pop('device_partition'), merge_nested=True)}, {'name': 'country', 'partitions': __flatten_sort_key_count(data=row.pop('country_partition'))}]

def __process_tags_map(row):
    if False:
        i = 10
        return i + 15
    browsers_partition = row.pop('browsers_partition')
    os_partition = row.pop('os_partition')
    device_partition = row.pop('device_partition')
    country_partition = row.pop('country_partition')
    return [{'name': 'browser', 'partitions': __transform_map_to_tag(data=browsers_partition, key1='browser', key2='browser_version', requested_key=1)}, {'name': 'browser.ver', 'partitions': __transform_map_to_tag(data=browsers_partition, key1='browser', key2='browser_version', requested_key=2)}, {'name': 'OS', 'partitions': __transform_map_to_tag(data=os_partition, key1='os', key2='os_version', requested_key=1)}, {'name': 'OS.ver', 'partitions': __transform_map_to_tag(data=os_partition, key1='os', key2='os_version', requested_key=2)}, {'name': 'device.family', 'partitions': __transform_map_to_tag(data=device_partition, key1='device_type', key2='device', requested_key=1)}, {'name': 'device', 'partitions': __transform_map_to_tag(data=device_partition, key1='device_type', key2='device', requested_key=2)}, {'name': 'country', 'partitions': __transform_map_to_tag(data=country_partition, key1='country', key2='', requested_key=1)}]

def get_details_deprecated(project_id, error_id, user_id, **data):
    if False:
        print('Hello World!')
    if not config('EXP_ERRORS_GET', cast=bool, default=False):
        return errors_legacy.get_details(project_id, error_id, user_id, **data)
    MAIN_SESSIONS_TABLE = exp_ch_helper.get_main_sessions_table(0)
    MAIN_EVENTS_TABLE = exp_ch_helper.get_main_events_table(0)
    MAIN_EVENTS_TABLE_24 = exp_ch_helper.get_main_events_table(TimeUTC.now())
    ch_sub_query24 = __get_basic_constraints(startTime_arg_name='startDate24', endTime_arg_name='endDate24')
    ch_sub_query24.append('error_id = %(error_id)s')
    pg_sub_query30_err = __get_basic_constraints(time_constraint=True, startTime_arg_name='startDate30', endTime_arg_name='endDate30', project_key='errors.project_id', table_name='errors')
    pg_sub_query30_err.append('sessions.project_id = toUInt16(%(project_id)s)')
    pg_sub_query30_err.append('sessions.datetime >= toDateTime(%(startDate30)s/1000)')
    pg_sub_query30_err.append('sessions.datetime <= toDateTime(%(endDate30)s/1000)')
    pg_sub_query30_err.append('error_id = %(error_id)s')
    pg_sub_query30_err.append("source ='js_exception'")
    ch_sub_query30 = __get_basic_constraints(startTime_arg_name='startDate30', endTime_arg_name='endDate30', project_key='errors.project_id')
    ch_sub_query30.append('error_id = %(error_id)s')
    ch_basic_query = __get_basic_constraints(time_constraint=False)
    ch_basic_query.append('error_id = %(error_id)s')
    ch_basic_query_session = ch_basic_query[:]
    ch_basic_query_session.append('sessions.project_id = toUInt16(%(project_id)s)')
    with ch_client.ClickHouseClient() as ch:
        data['startDate24'] = TimeUTC.now(-1)
        data['endDate24'] = TimeUTC.now()
        data['startDate30'] = TimeUTC.now(-30)
        data['endDate30'] = TimeUTC.now()
        density24 = int(data.get('density24', 24))
        step_size24 = __get_step_size(data['startDate24'], data['endDate24'], density24)
        density30 = int(data.get('density30', 30))
        step_size30 = __get_step_size(data['startDate30'], data['endDate30'], density30)
        params = {'startDate24': data['startDate24'], 'endDate24': data['endDate24'], 'startDate30': data['startDate30'], 'endDate30': data['endDate30'], 'project_id': project_id, 'userId': user_id, 'step_size24': step_size24, 'step_size30': step_size30, 'error_id': error_id}
        main_ch_query = f"        SELECT details.error_id AS error_id,\n               name,\n               message,\n               users,\n               sessions,\n               last_occurrence,\n               first_occurrence,\n               last_session_id,\n               browsers_partition,\n               os_partition,\n               device_partition,\n               country_partition,\n               chart24,\n               chart30\n        FROM (SELECT error_id,\n                     name,\n                     message,\n                     COUNT(DISTINCT user_uuid)  AS users,\n                     COUNT(DISTINCT session_id) AS sessions\n              FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n              WHERE {' AND '.join(pg_sub_query30_err)}\n              GROUP BY error_id, name, message) AS details\n                 INNER JOIN (SELECT error_id,\n                                    toUnixTimestamp(max(datetime)) * 1000 AS last_occurrence,\n                                    toUnixTimestamp(min(datetime)) * 1000 AS first_occurrence\n                             FROM {MAIN_EVENTS_TABLE} AS errors\n                             WHERE {' AND '.join(ch_basic_query)}\n                             GROUP BY error_id) AS time_details\n                            ON details.error_id = time_details.error_id\n                 INNER JOIN (SELECT error_id, session_id AS last_session_id, user_os, user_os_version, user_browser, user_browser_version, user_device, user_device_type, user_uuid\n                             FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                             WHERE {' AND '.join(ch_basic_query_session)}\n                             ORDER BY errors.datetime DESC\n                             LIMIT 1) AS last_session_details ON last_session_details.error_id = details.error_id\n                 INNER JOIN (SELECT %(error_id)s AS error_id,\n                                    groupArray([[[user_browser]], [[toString(count_per_browser)]],versions_partition]) AS browsers_partition\n                             FROM (SELECT user_browser,\n                                          COUNT(session_id) AS count_per_browser\n                                   FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                                   WHERE {' AND '.join(pg_sub_query30_err)}\n                                   GROUP BY user_browser\n                                   ORDER BY count_per_browser DESC) AS count_per_browser_query\n                                      INNER JOIN (SELECT user_browser,\n                                                         groupArray([user_browser_version, toString(count_per_version)]) AS versions_partition\n                                                  FROM (SELECT user_browser,\n                                                               user_browser_version,\n                                                               COUNT(session_id) AS count_per_version\n                                                        FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                                                        WHERE {' AND '.join(pg_sub_query30_err)}\n                                                        GROUP BY user_browser, user_browser_version\n                                                        ORDER BY count_per_version DESC) AS version_details\n                                                  GROUP BY user_browser ) AS browser_version_details USING (user_browser)) AS browser_details\n                            ON browser_details.error_id = details.error_id\n                 INNER JOIN (SELECT %(error_id)s AS error_id,\n                                    groupArray([[[user_os]], [[toString(count_per_os)]],versions_partition]) AS os_partition\n                             FROM (SELECT user_os,\n                                          COUNT(session_id) AS count_per_os\n                                   FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                                   WHERE {' AND '.join(pg_sub_query30_err)}\n                                   GROUP BY user_os\n                                   ORDER BY count_per_os DESC) AS count_per_os_details\n                                      INNER JOIN (SELECT user_os,\n                                                         groupArray([user_os_version, toString(count_per_version)]) AS versions_partition\n                                                  FROM (SELECT user_os, user_os_version, COUNT(session_id) AS count_per_version\n                                                        FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                                                        WHERE {' AND '.join(pg_sub_query30_err)}\n                                                        GROUP BY user_os, user_os_version\n                                                        ORDER BY count_per_version DESC) AS count_per_version_details\n                                                  GROUP BY user_os ) AS os_version_details USING (user_os)) AS os_details\n                            ON os_details.error_id = details.error_id\n                 INNER JOIN (SELECT %(error_id)s AS error_id,\n                                    groupArray([[[toString(user_device_type)]], [[toString(count_per_device)]],versions_partition]) AS device_partition\n                             FROM (SELECT user_device_type,\n                                          COUNT(session_id) AS count_per_device\n                                   FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                                   WHERE {' AND '.join(pg_sub_query30_err)}\n                                   GROUP BY user_device_type\n                                   ORDER BY count_per_device DESC) AS count_per_device_details\n                                      INNER JOIN (SELECT user_device_type,\n                                                         groupArray([user_device, toString(count_per_device)]) AS versions_partition\n                                                  FROM (SELECT user_device_type,\n                                                               coalesce(user_device,'unknown') AS user_device,\n                                                               COUNT(session_id) AS count_per_device\n                                                        FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                                                        WHERE {' AND '.join(pg_sub_query30_err)}\n                                                        GROUP BY user_device_type, user_device\n                                                        ORDER BY count_per_device DESC) AS count_per_device_details\n                                                  GROUP BY user_device_type ) AS device_version_details USING (user_device_type)) AS device_details\n                            ON device_details.error_id = details.error_id\n                 INNER JOIN (SELECT %(error_id)s AS error_id,\n                                    groupArray([[[toString(user_country)]], [[toString(count_per_country)]]]) AS country_partition\n                             FROM (SELECT user_country,\n                                          COUNT(session_id) AS count_per_country\n                                   FROM {MAIN_EVENTS_TABLE} AS errors INNER JOIN {MAIN_SESSIONS_TABLE} AS sessions USING (session_id)\n                                   WHERE {' AND '.join(pg_sub_query30_err)}\n                                   GROUP BY user_country\n                                   ORDER BY count_per_country DESC) AS count_per_country_details) AS country_details\n                            ON country_details.error_id = details.error_id\n                 INNER JOIN (SELECT %(error_id)s AS error_id, groupArray([timestamp, count]) AS chart24\n                             FROM (SELECT toUnixTimestamp(toStartOfInterval(datetime, INTERVAL %(step_size24)s second)) * 1000 AS timestamp,\n                                          COUNT(DISTINCT session_id) AS count\n                                   FROM {MAIN_EVENTS_TABLE_24} AS errors\n                                   WHERE {' AND '.join(ch_sub_query24)}\n                                   GROUP BY timestamp\n                                   ORDER BY timestamp) AS chart_details) AS chart_details24\n                            ON details.error_id = chart_details24.error_id\n                 INNER JOIN (SELECT %(error_id)s AS error_id, groupArray([timestamp, count]) AS chart30\n                             FROM (SELECT toUnixTimestamp(toStartOfInterval(datetime, INTERVAL %(step_size30)s second)) * 1000 AS timestamp,\n                                          COUNT(DISTINCT session_id) AS count\n                                   FROM {MAIN_EVENTS_TABLE} AS errors\n                                   WHERE {' AND '.join(ch_sub_query30)}\n                                   GROUP BY timestamp\n                                   ORDER BY timestamp) AS chart_details) AS chart_details30\n                            ON details.error_id = chart_details30.error_id;"
        row = ch.execute(query=main_ch_query, params=params)
    if len(row) == 0:
        return {'errors': ['error not found']}
    row = row[0]
    row['tags'] = __process_tags(row)
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(f'SELECT error_id, status, session_id, start_ts, \n                        parent_error_id, user_anonymous_id,\n                        user_id, user_uuid, user_browser, user_browser_version, \n                        user_os, user_os_version, user_device, payload,\n                                    FALSE AS favorite,\n                                       True AS viewed\n                                FROM public.errors AS pe\n                                         INNER JOIN events.errors AS ee USING (error_id)\n                                         INNER JOIN public.sessions USING (session_id)\n                                WHERE pe.project_id = %(project_id)s\n                                  AND sessions.project_id = %(project_id)s\n                                  AND error_id = %(error_id)s\n                                ORDER BY start_ts DESC\n                                LIMIT 1;', {'project_id': project_id, 'error_id': error_id, 'userId': user_id})
        cur.execute(query=query)
        status = cur.fetchone()
    if status is not None:
        row['stack'] = format_first_stack_frame(status).pop('stack')
        row['status'] = status.pop('status')
        row['parent_error_id'] = status.pop('parent_error_id')
        row['favorite'] = status.pop('favorite')
        row['viewed'] = status.pop('viewed')
        row['last_hydrated_session'] = status
    else:
        row['stack'] = []
        row['last_hydrated_session'] = None
        row['status'] = 'untracked'
        row['parent_error_id'] = None
        row['favorite'] = False
        row['viewed'] = False
    row['chart24'] = __rearrange_chart_details(start_at=data['startDate24'], end_at=data['endDate24'], density=density24, chart=row['chart24'])
    row['chart30'] = __rearrange_chart_details(start_at=data['startDate30'], end_at=data['endDate30'], density=density30, chart=row['chart30'])
    return {'data': helper.dict_to_camel_case(row)}

def get_details(project_id, error_id, user_id, **data):
    if False:
        return 10
    if not config('EXP_ERRORS_GET', cast=bool, default=False):
        return errors_legacy.get_details(project_id, error_id, user_id, **data)
    MAIN_SESSIONS_TABLE = exp_ch_helper.get_main_sessions_table(0)
    MAIN_ERR_SESS_TABLE = exp_ch_helper.get_main_js_errors_sessions_table(0)
    MAIN_EVENTS_TABLE = exp_ch_helper.get_main_events_table(0)
    MAIN_EVENTS_TABLE_24 = exp_ch_helper.get_main_events_table(TimeUTC.now())
    ch_sub_query24 = __get_basic_constraints(startTime_arg_name='startDate24', endTime_arg_name='endDate24')
    ch_sub_query24.append('error_id = %(error_id)s')
    ch_sub_query30 = __get_basic_constraints(startTime_arg_name='startDate30', endTime_arg_name='endDate30', project_key='errors.project_id')
    ch_sub_query30.append('error_id = %(error_id)s')
    ch_basic_query = __get_basic_constraints(time_constraint=False)
    ch_basic_query.append('error_id = %(error_id)s')
    with ch_client.ClickHouseClient() as ch:
        data['startDate24'] = TimeUTC.now(-1)
        data['endDate24'] = TimeUTC.now()
        data['startDate30'] = TimeUTC.now(-30)
        data['endDate30'] = TimeUTC.now()
        density24 = int(data.get('density24', 24))
        step_size24 = __get_step_size(data['startDate24'], data['endDate24'], density24)
        density30 = int(data.get('density30', 30))
        step_size30 = __get_step_size(data['startDate30'], data['endDate30'], density30)
        params = {'startDate24': data['startDate24'], 'endDate24': data['endDate24'], 'startDate30': data['startDate30'], 'endDate30': data['endDate30'], 'project_id': project_id, 'userId': user_id, 'step_size24': step_size24, 'step_size30': step_size30, 'error_id': error_id}
        main_ch_query = f"        WITH pre_processed AS (SELECT error_id,\n                                      name,\n                                      message,\n                                      session_id,\n                                      datetime,\n                                      user_id,\n                                      user_browser,\n                                      user_browser_version,\n                                      user_os,\n                                      user_os_version,\n                                      user_device_type,\n                                      user_device,\n                                      user_country,\n                                      error_tags_keys, \n                                      error_tags_values\n                               FROM {MAIN_ERR_SESS_TABLE} AS errors\n                               WHERE {' AND '.join(ch_basic_query)}\n                               )\n        SELECT %(error_id)s AS error_id, name, message,users,\n                first_occurrence,last_occurrence,last_session_id,\n                sessions,browsers_partition,os_partition,device_partition,\n                country_partition,chart24,chart30,custom_tags\n        FROM (SELECT error_id,\n                     name,\n                     message\n              FROM pre_processed\n              LIMIT 1) AS details\n                  INNER JOIN (SELECT COUNT(DISTINCT user_id)    AS users,\n                                     COUNT(DISTINCT session_id) AS sessions\n                              FROM pre_processed\n                              WHERE datetime >= toDateTime(%(startDate30)s / 1000)\n                                AND datetime <= toDateTime(%(endDate30)s / 1000)\n                              ) AS last_month_stats ON TRUE\n                  INNER JOIN (SELECT toUnixTimestamp(max(datetime)) * 1000 AS last_occurrence,\n                                     toUnixTimestamp(min(datetime)) * 1000 AS first_occurrence\n                              FROM pre_processed) AS time_details ON TRUE\n                  INNER JOIN (SELECT session_id AS last_session_id,\n                                    arrayMap((key, value)->(map(key, value)), error_tags_keys, error_tags_values) AS custom_tags\n                              FROM pre_processed\n                              ORDER BY datetime DESC\n                              LIMIT 1) AS last_session_details ON TRUE\n                  INNER JOIN (SELECT groupArray(details) AS browsers_partition\n                              FROM (SELECT COUNT(1)                                              AS count,\n                                           coalesce(nullIf(user_browser,''),toNullable('unknown')) AS browser,\n                                           coalesce(nullIf(user_browser_version,''),toNullable('unknown')) AS browser_version,\n                                           map('browser', browser,\n                                               'browser_version', browser_version,\n                                               'count', toString(count)) AS details\n                                    FROM pre_processed\n                                    GROUP BY ROLLUP(browser, browser_version)\n                                    ORDER BY browser nulls first, browser_version nulls first, count DESC) AS mapped_browser_details\n                 ) AS browser_details ON TRUE\n                 INNER JOIN (SELECT groupArray(details) AS os_partition\n                             FROM (SELECT COUNT(1)                                    AS count,\n                                          coalesce(nullIf(user_os,''),toNullable('unknown')) AS os,\n                                          coalesce(nullIf(user_os_version,''),toNullable('unknown')) AS os_version,\n                                          map('os', os,\n                                              'os_version', os_version,\n                                              'count', toString(count)) AS details\n                                   FROM pre_processed\n                                   GROUP BY ROLLUP(os, os_version)\n                                   ORDER BY os nulls first, os_version nulls first, count DESC) AS mapped_os_details\n                    ) AS os_details ON TRUE\n                 INNER JOIN (SELECT groupArray(details) AS device_partition\n                             FROM (SELECT COUNT(1)                                            AS count,\n                                          coalesce(nullIf(user_device,''),toNullable('unknown')) AS user_device,\n                                          map('device_type', toString(user_device_type),\n                                              'device', user_device,\n                                              'count', toString(count)) AS details\n                                   FROM pre_processed\n                                   GROUP BY ROLLUP(user_device_type, user_device)\n                                   ORDER BY user_device_type nulls first, user_device nulls first, count DESC\n                                      ) AS count_per_device_details\n                            ) AS mapped_device_details ON TRUE\n                 INNER JOIN (SELECT groupArray(details) AS country_partition\n                             FROM (SELECT COUNT(1)  AS count,\n                                          map('country', toString(user_country),\n                                              'count', toString(count)) AS details\n                                   FROM pre_processed\n                                   GROUP BY user_country\n                                   ORDER BY count DESC) AS count_per_country_details\n                            ) AS mapped_country_details ON TRUE\n                 INNER JOIN (SELECT groupArray(map('timestamp', timestamp, 'count', count)) AS chart24\n                             FROM (SELECT toUnixTimestamp(toStartOfInterval(datetime, INTERVAL 3756 second)) *\n                                          1000                       AS timestamp,\n                                          COUNT(DISTINCT session_id) AS count\n                                   FROM {MAIN_EVENTS_TABLE} AS errors\n                                   WHERE {' AND '.join(ch_sub_query24)}\n                                   GROUP BY timestamp\n                                   ORDER BY timestamp) AS chart_details\n                            ) AS chart_details24 ON TRUE\n                 INNER JOIN (SELECT groupArray(map('timestamp', timestamp, 'count', count)) AS chart30\n                             FROM (SELECT toUnixTimestamp(toStartOfInterval(datetime, INTERVAL 3724 second)) *\n                                          1000                       AS timestamp,\n                                          COUNT(DISTINCT session_id) AS count\n                                   FROM {MAIN_EVENTS_TABLE} AS errors\n                                   WHERE {' AND '.join(ch_sub_query30)}\n                                   GROUP BY timestamp\n                                   ORDER BY timestamp) AS chart_details\n                            ) AS chart_details30 ON TRUE;"
        row = ch.execute(query=main_ch_query, params=params)
        if len(row) == 0:
            return {'errors': ['error not found']}
        row = row[0]
        row['tags'] = __process_tags_map(row)
        query = f'SELECT session_id, toUnixTimestamp(datetime) * 1000 AS start_ts,\n                         user_anonymous_id,user_id, user_uuid, user_browser, user_browser_version,\n                        user_os, user_os_version, user_device, FALSE AS favorite, True AS viewed\n                    FROM {MAIN_SESSIONS_TABLE} AS sessions\n                    WHERE project_id = toUInt16(%(project_id)s)\n                      AND session_id = %(session_id)s\n                    ORDER BY datetime DESC\n                    LIMIT 1;'
        params = {'project_id': project_id, 'session_id': row['last_session_id'], 'userId': user_id}
        status = ch.execute(query=query, params=params)
    if status is not None:
        status = status[0]
        row['favorite'] = status.pop('favorite')
        row['viewed'] = status.pop('viewed')
        row['last_hydrated_session'] = status
    else:
        row['last_hydrated_session'] = None
        row['favorite'] = False
        row['viewed'] = False
    row['chart24'] = metrics.__complete_missing_steps(start_time=data['startDate24'], end_time=data['endDate24'], density=density24, rows=row['chart24'], neutral={'count': 0})
    row['chart30'] = metrics.__complete_missing_steps(start_time=data['startDate30'], end_time=data['endDate30'], density=density30, rows=row['chart30'], neutral={'count': 0})
    return {'data': helper.dict_to_camel_case(row)}

def get_details_chart(project_id, error_id, user_id, **data):
    if False:
        print('Hello World!')
    ch_sub_query = __get_basic_constraints()
    ch_sub_query.append('error_id = %(error_id)s')
    with ch_client.ClickHouseClient() as ch:
        if data.get('startDate') is None:
            data['startDate'] = TimeUTC.now(-7)
        else:
            data['startDate'] = int(data['startDate'])
        if data.get('endDate') is None:
            data['endDate'] = TimeUTC.now()
        else:
            data['endDate'] = int(data['endDate'])
        density = int(data.get('density', 7))
        step_size = __get_step_size(data['startDate'], data['endDate'], density)
        params = {'startDate': data['startDate'], 'endDate': data['endDate'], 'project_id': project_id, 'userId': user_id, 'step_size': step_size, 'error_id': error_id}
        main_ch_query = f"        SELECT browser_details.error_id AS error_id,\n               browsers_partition,\n               os_partition,\n               device_partition,\n               country_partition,\n               chart\n        FROM (SELECT %(error_id)s                                             AS error_id,\n                     groupArray([[[user_browser]], [[toString(count_per_browser)]],versions_partition]) AS browsers_partition\n              FROM (SELECT user_browser,\n                           COUNT(session_id) AS count_per_browser\n                    FROM errors\n                    WHERE {' AND '.join(ch_sub_query)}\n                    GROUP BY user_browser\n                    ORDER BY count_per_browser DESC) AS count_per_browser_query\n                       INNER JOIN (SELECT user_browser,\n                                          groupArray([user_browser_version, toString(count_per_version)]) AS versions_partition\n                                   FROM (SELECT user_browser,\n                                                user_browser_version,\n                                                COUNT(session_id) AS count_per_version\n                                         FROM errors\n                                         WHERE {' AND '.join(ch_sub_query)}\n                                         GROUP BY user_browser, user_browser_version\n                                         ORDER BY count_per_version DESC) AS count_per_version_details\n                                   GROUP BY user_browser ) AS browesr_version_details USING (user_browser)) AS browser_details\n                 INNER JOIN (SELECT %(error_id)s                                   AS error_id,\n                                    groupArray(\n                                            [[[user_os]], [[toString(count_per_os)]],versions_partition]) AS os_partition\n                             FROM (SELECT user_os,\n                                          COUNT(session_id) AS count_per_os\n                                   FROM errors\n                                   WHERE {' AND '.join(ch_sub_query)}\n                                   GROUP BY user_os\n                                   ORDER BY count_per_os DESC) AS count_per_os_details\n                                      INNER JOIN (SELECT user_os,\n                                                         groupArray([user_os_version, toString(count_per_version)]) AS versions_partition\n                                                  FROM (SELECT user_os, user_os_version, COUNT(session_id) AS count_per_version\n                                                        FROM errors\n                                                        WHERE {' AND '.join(ch_sub_query)}\n                                                        GROUP BY user_os, user_os_version\n                                                        ORDER BY count_per_version DESC) AS count_per_version_query\n                                                  GROUP BY user_os ) AS os_version_query USING (user_os)) AS os_details\n                            ON os_details.error_id = browser_details.error_id\n                 INNER JOIN (SELECT %(error_id)s                                                          AS error_id,\n                                    groupArray(\n                                            [[[toString(user_device_type)]], [[toString(count_per_device)]],versions_partition]) AS device_partition\n                             FROM (SELECT user_device_type,\n                                          COUNT(session_id) AS count_per_device\n                                   FROM errors\n                                   WHERE {' AND '.join(ch_sub_query)}\n                                   GROUP BY user_device_type\n                                   ORDER BY count_per_device DESC) AS count_per_device_details\n                                      INNER JOIN (SELECT user_device_type,\n                                                         groupArray([user_device, toString(count_per_device)]) AS versions_partition\n                                                  FROM (SELECT user_device_type,\n                                                               coalesce(user_device,'unknown') AS user_device,\n                                                               COUNT(session_id) AS count_per_device\n                                                        FROM errors\n                                                        WHERE {' AND '.join(ch_sub_query)}\n                                                        GROUP BY user_device_type, user_device\n                                                        ORDER BY count_per_device DESC) AS count_per_device_details\n                                                  GROUP BY user_device_type ) AS device_version_details USING (user_device_type)) AS device_details\n                            ON device_details.error_id = os_details.error_id\n                 INNER JOIN (SELECT %(error_id)s                                    AS error_id,\n                                    groupArray(\n                                            [[[toString(user_country)]], [[toString(count_per_country)]]]) AS country_partition\n                             FROM (SELECT user_country,\n                                          COUNT(session_id) AS count_per_country\n                                   FROM errors\n                                   WHERE {' AND '.join(ch_sub_query)}\n                                   GROUP BY user_country\n                                   ORDER BY count_per_country DESC) AS count_per_country_details) AS country_details\n                            ON country_details.error_id = device_details.error_id\n                 INNER JOIN (SELECT %(error_id)s AS error_id, groupArray([timestamp, count]) AS chart\n                             FROM (SELECT toUnixTimestamp(toStartOfInterval(datetime, INTERVAL %(step_size)s second)) * 1000                       AS timestamp,\n                                          COUNT(DISTINCT session_id) AS count\n                                   FROM errors\n                                   WHERE {' AND '.join(ch_sub_query)}\n                                   GROUP BY timestamp\n                                   ORDER BY timestamp) AS chart_details) AS chart_details\n                            ON country_details.error_id = chart_details.error_id;"
        row = ch.execute(query=main_ch_query, params=params)
    if len(row) == 0:
        return {'errors': ['error not found']}
    row = row[0]
    row['tags'] = __process_tags(row)
    row['chart'] = __rearrange_chart_details(start_at=data['startDate'], end_at=data['endDate'], density=density, chart=row['chart'])
    return {'data': helper.dict_to_camel_case(row)}

def __get_basic_constraints(platform=None, time_constraint=True, startTime_arg_name='startDate', endTime_arg_name='endDate', type_condition=True, project_key='project_id', table_name=None):
    if False:
        for i in range(10):
            print('nop')
    ch_sub_query = [f'{project_key} =toUInt16(%(project_id)s)']
    if table_name is not None:
        table_name = table_name + '.'
    else:
        table_name = ''
    if type_condition:
        ch_sub_query.append(f"{table_name}EventType='ERROR'")
    if time_constraint:
        ch_sub_query += [f'{table_name}datetime >= toDateTime(%({startTime_arg_name})s/1000)', f'{table_name}datetime < toDateTime(%({endTime_arg_name})s/1000)']
    if platform == schemas.PlatformType.mobile:
        ch_sub_query.append("user_device_type = 'mobile'")
    elif platform == schemas.PlatformType.desktop:
        ch_sub_query.append("user_device_type = 'desktop'")
    return ch_sub_query

def __get_step_size(startTimestamp, endTimestamp, density):
    if False:
        print('Hello World!')
    step_size = (int(endTimestamp) // 1000 - int(startTimestamp) // 1000) // (int(density) - 1)
    return step_size

def __get_sort_key(key):
    if False:
        print('Hello World!')
    return {schemas.ErrorSort.occurrence: 'max_datetime', schemas.ErrorSort.users_count: 'users', schemas.ErrorSort.sessions_count: 'sessions'}.get(key, 'max_datetime')

def __get_basic_constraints_pg(platform=None, time_constraint=True, startTime_arg_name='startDate', endTime_arg_name='endDate', chart=False, step_size_name='step_size', project_key='project_id'):
    if False:
        i = 10
        return i + 15
    if project_key is None:
        ch_sub_query = []
    else:
        ch_sub_query = [f'{project_key} =%(project_id)s']
    if time_constraint:
        ch_sub_query += [f'timestamp >= %({startTime_arg_name})s', f'timestamp < %({endTime_arg_name})s']
    if chart:
        ch_sub_query += [f'timestamp >=  generated_timestamp', f'timestamp <  generated_timestamp + %({step_size_name})s']
    if platform == schemas.PlatformType.mobile:
        ch_sub_query.append("user_device_type = 'mobile'")
    elif platform == schemas.PlatformType.desktop:
        ch_sub_query.append("user_device_type = 'desktop'")
    return ch_sub_query

def search(data: schemas.SearchErrorsSchema, project_id, user_id):
    if False:
        while True:
            i = 10
    MAIN_EVENTS_TABLE = exp_ch_helper.get_main_events_table(data.startDate)
    MAIN_SESSIONS_TABLE = exp_ch_helper.get_main_sessions_table(data.startDate)
    platform = None
    for f in data.filters:
        if f.type == schemas.FilterType.platform and len(f.value) > 0:
            platform = f.value[0]
    ch_sessions_sub_query = __get_basic_constraints(platform, type_condition=False)
    ch_sub_query = __get_basic_constraints(platform, type_condition=True)
    ch_sub_query.append("source ='js_exception'")
    ch_sub_query.append("message!='Script error.'")
    error_ids = None
    if data.startDate is None:
        data.startDate = TimeUTC.now(-7)
    if data.endDate is None:
        data.endDate = TimeUTC.now(1)
    subquery_part = ''
    params = {}
    if len(data.events) > 0:
        errors_condition_count = 0
        for (i, e) in enumerate(data.events):
            if e.type == schemas.EventType.error:
                errors_condition_count += 1
                is_any = _isAny_opreator(e.operator)
                op = __get_sql_operator(e.operator)
                e_k = f'e_value{i}'
                params = {**params, **_multiple_values(e.value, value_key=e_k)}
                if not is_any and len(e.value) > 0 and (e.value[1] not in [None, '*', '']):
                    ch_sub_query.append(_multiple_conditions(f'(message {op} %({e_k})s OR name {op} %({e_k})s)', e.value, value_key=e_k))
        if len(data.events) > errors_condition_count:
            (subquery_part_args, subquery_part) = sessions.search_query_parts_ch(data=data, error_status=data.status, errors_only=True, project_id=project_id, user_id=user_id, issue=None, favorite_only=False)
            subquery_part = f'INNER JOIN {subquery_part} USING(session_id)'
            params = {**params, **subquery_part_args}
    if len(data.filters) > 0:
        meta_keys = None
        for (i, f) in enumerate(data.filters):
            if not isinstance(f.value, list):
                f.value = [f.value]
            filter_type = f.type
            f.value = helper.values_for_operator(value=f.value, op=f.operator)
            f_k = f'f_value{i}'
            params = {**params, f_k: f.value, **_multiple_values(f.value, value_key=f_k)}
            op = __get_sql_operator(f.operator) if filter_type not in [schemas.FilterType.events_count] else f.operator
            is_any = _isAny_opreator(f.operator)
            is_undefined = _isUndefined_operator(f.operator)
            if not is_any and (not is_undefined) and (len(f.value) == 0):
                continue
            is_not = False
            if __is_negation_operator(f.operator):
                is_not = True
            if filter_type == schemas.FilterType.user_browser:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.user_browser)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.user_browser {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.user_os, schemas.FilterType.user_os_ios]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.user_os)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.user_os {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.user_device, schemas.FilterType.user_device_ios]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.user_device)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.user_device {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.user_country, schemas.FilterType.user_country_ios]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.user_country)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.user_country {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.utm_source]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.utm_source)')
                elif is_undefined:
                    ch_sessions_sub_query.append('isNull(s.utm_source)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.utm_source {op} toString(%({f_k})s)', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.utm_medium]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.utm_medium)')
                elif is_undefined:
                    ch_sessions_sub_query.append('isNull(s.utm_medium)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.utm_medium {op} toString(%({f_k})s)', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.utm_campaign]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.utm_campaign)')
                elif is_undefined:
                    ch_sessions_sub_query.append('isNull(s.utm_campaign)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.utm_campaign {op} toString(%({f_k})s)', f.value, is_not=is_not, value_key=f_k))
            elif filter_type == schemas.FilterType.duration:
                if len(f.value) > 0 and f.value[0] is not None:
                    ch_sessions_sub_query.append('s.duration >= %(minDuration)s')
                    params['minDuration'] = f.value[0]
                if len(f.value) > 1 and f.value[1] is not None and (int(f.value[1]) > 0):
                    ch_sessions_sub_query.append('s.duration <= %(maxDuration)s')
                    params['maxDuration'] = f.value[1]
            elif filter_type == schemas.FilterType.referrer:
                if is_any:
                    referrer_constraint = 'isNotNull(s.base_referrer)'
                else:
                    referrer_constraint = _multiple_conditions(f's.base_referrer {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k)
            elif filter_type == schemas.FilterType.metadata:
                if meta_keys is None:
                    meta_keys = metadata.get(project_id=project_id)
                    meta_keys = {m['key']: m['index'] for m in meta_keys}
                if f.source in meta_keys.keys():
                    if is_any:
                        ch_sessions_sub_query.append(f'isNotNull(s.{metadata.index_to_colname(meta_keys[f.source])})')
                    elif is_undefined:
                        ch_sessions_sub_query.append(f'isNull(s.{metadata.index_to_colname(meta_keys[f.source])})')
                    else:
                        ch_sessions_sub_query.append(_multiple_conditions(f's.{metadata.index_to_colname(meta_keys[f.source])} {op} toString(%({f_k})s)', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.user_id, schemas.FilterType.user_id_ios]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.user_id)')
                elif is_undefined:
                    ch_sessions_sub_query.append('isNull(s.user_id)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.user_id {op} toString(%({f_k})s)', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.user_anonymous_id, schemas.FilterType.user_anonymous_id_ios]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.user_anonymous_id)')
                elif is_undefined:
                    ch_sessions_sub_query.append('isNull(s.user_anonymous_id)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.user_anonymous_id {op} toString(%({f_k})s)', f.value, is_not=is_not, value_key=f_k))
            elif filter_type in [schemas.FilterType.rev_id, schemas.FilterType.rev_id_ios]:
                if is_any:
                    ch_sessions_sub_query.append('isNotNull(s.rev_id)')
                elif is_undefined:
                    ch_sessions_sub_query.append('isNull(s.rev_id)')
                else:
                    ch_sessions_sub_query.append(_multiple_conditions(f's.rev_id {op} toString(%({f_k})s)', f.value, is_not=is_not, value_key=f_k))
            elif filter_type == schemas.FilterType.platform:
                ch_sessions_sub_query.append(_multiple_conditions(f's.user_device_type {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
            elif filter_type == schemas.FilterType.events_count:
                ch_sessions_sub_query.append(_multiple_conditions(f's.events_count {op} %({f_k})s', f.value, is_not=is_not, value_key=f_k))
    with ch_client.ClickHouseClient() as ch:
        step_size = __get_step_size(data.startDate, data.endDate, data.density)
        sort = __get_sort_key('datetime')
        if data.sort is not None:
            sort = __get_sort_key(data.sort)
        order = 'DESC'
        if data.order is not None:
            order = data.order
        params = {**params, 'startDate': data.startDate, 'endDate': data.endDate, 'project_id': project_id, 'userId': user_id, 'step_size': step_size}
        if data.limit is not None and data.page is not None:
            params['errors_offset'] = (data.page - 1) * data.limit
            params['errors_limit'] = data.limit
        else:
            params['errors_offset'] = 0
            params['errors_limit'] = 200
        if error_ids is not None:
            params['error_ids'] = tuple(error_ids)
            ch_sub_query.append('error_id IN %(error_ids)s')
        main_ch_query = f"                SELECT details.error_id AS error_id, \n                        name, message, users, total, viewed,\n                        sessions, last_occurrence, first_occurrence, chart\n                FROM (SELECT error_id,\n                             name,\n                             message,\n                             COUNT(DISTINCT user_id)  AS users,\n                             COUNT(DISTINCT events.session_id) AS sessions,\n                             MAX(datetime)              AS max_datetime,\n                             MIN(datetime)              AS min_datetime,\n                             COUNT(DISTINCT events.error_id) OVER() AS total,\n                             any(isNotNull(viewed_error_id)) AS viewed\n                      FROM {MAIN_EVENTS_TABLE} AS events\n                            LEFT JOIN (SELECT error_id AS viewed_error_id\n                                        FROM {exp_ch_helper.get_user_viewed_errors_table()}\n                                        WHERE project_id=%(project_id)s\n                                            AND user_id=%(userId)s) AS viewed_errors ON(events.error_id=viewed_errors.viewed_error_id)\n                            INNER JOIN (SELECT session_id, coalesce(user_id,toString(user_uuid)) AS user_id \n                                        FROM {MAIN_SESSIONS_TABLE} AS s\n                                                {subquery_part}\n                                        WHERE {' AND '.join(ch_sessions_sub_query)}) AS sessions \n                                                                                    ON (events.session_id = sessions.session_id)\n                      WHERE {' AND '.join(ch_sub_query)}\n                      GROUP BY error_id, name, message\n                      ORDER BY {sort} {order}\n                      LIMIT %(errors_limit)s OFFSET %(errors_offset)s) AS details \n                        INNER JOIN (SELECT error_id AS error_id, \n                                            toUnixTimestamp(MAX(datetime))*1000 AS last_occurrence, \n                                            toUnixTimestamp(MIN(datetime))*1000 AS first_occurrence\n                                     FROM {MAIN_EVENTS_TABLE}\n                                     WHERE project_id=%(project_id)s\n                                        AND EventType='ERROR'\n                                     GROUP BY error_id) AS time_details\n                ON details.error_id=time_details.error_id\n                    INNER JOIN (SELECT error_id, groupArray([timestamp, count]) AS chart\n                    FROM (SELECT error_id, toUnixTimestamp(toStartOfInterval(datetime, INTERVAL %(step_size)s second)) * 1000 AS timestamp,\n                            COUNT(DISTINCT session_id) AS count\n                            FROM {MAIN_EVENTS_TABLE}\n                            WHERE {' AND '.join(ch_sub_query)}\n                            GROUP BY error_id, timestamp\n                            ORDER BY timestamp) AS sub_table\n                            GROUP BY error_id) AS chart_details ON details.error_id=chart_details.error_id;"
        rows = ch.execute(query=main_ch_query, params=params)
        total = rows[0]['total'] if len(rows) > 0 else 0
    for r in rows:
        r['chart'] = list(r['chart'])
        for i in range(len(r['chart'])):
            r['chart'][i] = {'timestamp': r['chart'][i][0], 'count': r['chart'][i][1]}
        r['chart'] = metrics.__complete_missing_steps(rows=r['chart'], start_time=data.startDate, end_time=data.endDate, density=data.density, neutral={'count': 0})
    return {'total': total, 'errors': helper.list_to_camel_case(rows)}

def __save_stacktrace(error_id, data):
    if False:
        i = 10
        return i + 15
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify("UPDATE public.errors \n                SET stacktrace=%(data)s::jsonb, stacktrace_parsed_at=timezone('utc'::text, now())\n                WHERE error_id = %(error_id)s;", {'error_id': error_id, 'data': json.dumps(data)})
        cur.execute(query=query)

def get_trace(project_id, error_id):
    if False:
        while True:
            i = 10
    error = get(error_id=error_id, family=False)
    if error is None:
        return {'errors': ['error not found']}
    if error.get('source', '') != 'js_exception':
        return {'errors': ["this source of errors doesn't have a sourcemap"]}
    if error.get('payload') is None:
        return {'errors': ['null payload']}
    if error.get('stacktrace') is not None:
        return {'sourcemapUploaded': True, 'trace': error.get('stacktrace'), 'preparsed': True}
    (trace, all_exists) = sourcemaps.get_traces_group(project_id=project_id, payload=error['payload'])
    if all_exists:
        __save_stacktrace(error_id=error_id, data=trace)
    return {'sourcemapUploaded': all_exists, 'trace': trace, 'preparsed': False}

def get_sessions(start_date, end_date, project_id, user_id, error_id):
    if False:
        while True:
            i = 10
    extra_constraints = ['s.project_id = %(project_id)s', 's.start_ts >= %(startDate)s', 's.start_ts <= %(endDate)s', 'e.error_id = %(error_id)s']
    if start_date is None:
        start_date = TimeUTC.now(-7)
    if end_date is None:
        end_date = TimeUTC.now()
    params = {'startDate': start_date, 'endDate': end_date, 'project_id': project_id, 'userId': user_id, 'error_id': error_id}
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(f"SELECT s.project_id,\n                       s.session_id::text AS session_id,\n                       s.user_uuid,\n                       s.user_id,\n                       s.user_agent,\n                       s.user_os,\n                       s.user_browser,\n                       s.user_device,\n                       s.user_country,\n                       s.start_ts,\n                       s.duration,\n                       s.events_count,\n                       s.pages_count,\n                       s.errors_count,\n                       s.issue_types,\n                        coalesce((SELECT TRUE\n                         FROM public.user_favorite_sessions AS fs\n                         WHERE s.session_id = fs.session_id\n                           AND fs.user_id = %(userId)s LIMIT 1), FALSE) AS favorite,\n                        coalesce((SELECT TRUE\n                         FROM public.user_viewed_sessions AS fs\n                         WHERE s.session_id = fs.session_id\n                           AND fs.user_id = %(userId)s LIMIT 1), FALSE) AS viewed\n                FROM public.sessions AS s INNER JOIN events.errors AS e USING (session_id)\n                WHERE {' AND '.join(extra_constraints)}\n                ORDER BY s.start_ts DESC;", params)
        cur.execute(query=query)
        sessions_list = []
        total = cur.rowcount
        row = cur.fetchone()
        while row is not None and len(sessions_list) < 100:
            sessions_list.append(row)
            row = cur.fetchone()
    return {'total': total, 'sessions': helper.list_to_camel_case(sessions_list)}
ACTION_STATE = {'unsolve': 'unresolved', 'solve': 'resolved', 'ignore': 'ignored'}

def change_state(project_id, user_id, error_id, action):
    if False:
        return 10
    errors = get(error_id, family=True)
    print(len(errors))
    status = ACTION_STATE.get(action)
    if errors is None or len(errors) == 0:
        return {'errors': ['error not found']}
    if errors[0]['status'] == status:
        return {'errors': [f'error is already {status}']}
    if errors[0]['status'] == ACTION_STATE['solve'] and status == ACTION_STATE['ignore']:
        return {'errors': [f"state transition not permitted {errors[0]['status']} -> {status}"]}
    params = {'userId': user_id, 'error_ids': tuple([e['errorId'] for e in errors]), 'status': status}
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify('UPDATE public.errors\n                SET status = %(status)s\n                WHERE error_id IN %(error_ids)s\n                RETURNING status', params)
        cur.execute(query=query)
        row = cur.fetchone()
    if row is not None:
        for e in errors:
            e['status'] = row['status']
    return {'data': errors}
MAX_RANK = 2

def __status_rank(status):
    if False:
        return 10
    return {'unresolved': MAX_RANK - 2, 'ignored': MAX_RANK - 1, 'resolved': MAX_RANK}.get(status)

def merge(error_ids):
    if False:
        while True:
            i = 10
    error_ids = list(set(error_ids))
    errors = get_batch(error_ids)
    if len(error_ids) <= 1 or len(error_ids) > len(errors):
        return {'errors': ['invalid list of ids']}
    error_ids = [e['errorId'] for e in errors]
    parent_error_id = error_ids[0]
    status = 'unresolved'
    for e in errors:
        if __status_rank(status) < __status_rank(e['status']):
            status = e['status']
            if __status_rank(status) == MAX_RANK:
                break
    params = {'error_ids': tuple(error_ids), 'parent_error_id': parent_error_id, 'status': status}
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify('UPDATE public.errors\n                SET parent_error_id = %(parent_error_id)s, status = %(status)s\n                WHERE error_id IN %(error_ids)s OR parent_error_id IN %(error_ids)s;', params)
        cur.execute(query=query)
    return {'data': 'success'}

def format_first_stack_frame(error):
    if False:
        return 10
    error['stack'] = sourcemaps.format_payload(error.pop('payload'), truncate_to_first=True)
    for s in error['stack']:
        for c in s.get('context', []):
            for (sci, sc) in enumerate(c):
                if isinstance(sc, str) and len(sc) > 1000:
                    c[sci] = sc[:1000]
        if isinstance(s['filename'], bytes):
            s['filename'] = s['filename'].decode('utf-8')
    return error

def stats(project_id, user_id, startTimestamp=TimeUTC.now(delta_days=-7), endTimestamp=TimeUTC.now()):
    if False:
        i = 10
        return i + 15
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify("WITH user_viewed AS (SELECT error_id FROM public.user_viewed_errors WHERE user_id = %(userId)s)\n                SELECT COUNT(timed_errors.*) AS unresolved_and_unviewed\n                FROM (SELECT root_error.error_id\n                      FROM events.errors\n                               INNER JOIN public.errors AS root_error USING (error_id)\n                               LEFT JOIN user_viewed USING (error_id)\n                      WHERE project_id = %(project_id)s\n                        AND timestamp >= %(startTimestamp)s\n                        AND timestamp <= %(endTimestamp)s\n                        AND source = 'js_exception'\n                        AND root_error.status = 'unresolved'\n                        AND user_viewed.error_id ISNULL\n                      LIMIT 1\n                     ) AS timed_errors;", {'project_id': project_id, 'userId': user_id, 'startTimestamp': startTimestamp, 'endTimestamp': endTimestamp})
        cur.execute(query=query)
        row = cur.fetchone()
    return {'data': helper.dict_to_camel_case(row)}