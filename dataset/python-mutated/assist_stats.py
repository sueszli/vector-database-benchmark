import logging
from datetime import datetime
from fastapi import HTTPException
from chalicelib.utils import pg_client, helper
from schemas import AssistStatsSessionsRequest, AssistStatsSessionsResponse, AssistStatsTopMembersResponse
event_type_mapping = {'sessionsAssisted': 'assist', 'assistDuration': 'assist', 'callDuration': 'call', 'controlDuration': 'control'}

def insert_aggregated_data():
    if False:
        print('Hello World!')
    try:
        logging.info('Assist Stats: Inserting aggregated data')
        end_timestamp = int(datetime.timestamp(datetime.now())) * 1000
        start_timestamp = __last_run_end_timestamp_from_aggregates()
        if start_timestamp is None:
            logging.info('Assist Stats: First run, inserting data for last 7 days')
            start_timestamp = end_timestamp - 7 * 24 * 60 * 60 * 1000
        offset = 0
        chunk_size = 1000
        while True:
            constraints = ['timestamp BETWEEN %(start_timestamp)s AND %(end_timestamp)s']
            params = {'limit': chunk_size, 'offset': offset, 'start_timestamp': start_timestamp + 1, 'end_timestamp': end_timestamp, 'step_size': f'{60} seconds'}
            logging.info(f'Assist Stats: Fetching data from {start_timestamp} to {end_timestamp}')
            aggregated_data = __get_all_events_hourly_averages(constraints, params)
            if not aggregated_data:
                logging.info('Assist Stats: No more data to insert')
                break
            logging.info(f'Assist Stats: Inserting {len(aggregated_data)} rows')
            for data in aggregated_data:
                sql = '\n                    INSERT INTO assist_events_aggregates \n                    (timestamp, project_id, agent_id, assist_avg, call_avg, control_avg, assist_total, call_total, control_total)\n                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '
                params = (data['time'], data['project_id'], data['agent_id'], data['assist_avg'], data['call_avg'], data['control_avg'], data['assist_total'], data['call_total'], data['control_total'])
                with pg_client.PostgresClient() as cur:
                    cur.execute(sql, params)
            offset += chunk_size
        sql = f'\n            SELECT MAX(timestamp) as first_timestamp\n                FROM assist_events\n            WHERE timestamp > %(start_timestamp)s AND duration > 0\n            GROUP BY timestamp\n            ORDER BY timestamp DESC LIMIT 1\n        '
        with pg_client.PostgresClient() as cur:
            cur.execute(sql, params)
            result = cur.fetchone()
            first_timestamp = result['first_timestamp'] if result else None
        if first_timestamp is not None:
            sql = 'INSERT INTO assist_events_aggregates_logs (time) VALUES (%s)'
            params = (first_timestamp,)
            with pg_client.PostgresClient() as cur:
                cur.execute(sql, params)
    except Exception as e:
        logging.error(f'Error inserting aggregated data -: {e}')

def __last_run_end_timestamp_from_aggregates():
    if False:
        print('Hello World!')
    sql = 'SELECT MAX(time) as last_run_time FROM assist_events_aggregates_logs;'
    with pg_client.PostgresClient() as cur:
        cur.execute(sql)
        result = cur.fetchone()
        last_run_time = result['last_run_time'] if result else None
    if last_run_time is None:
        sql = 'SELECT MIN(timestamp) as last_timestamp FROM assist_events;'
        with pg_client.PostgresClient() as cur:
            cur.execute(sql)
            result = cur.fetchone()
            last_run_time = result['last_timestamp'] if result else None
    return last_run_time

def __get_all_events_hourly_averages(constraints, params):
    if False:
        for i in range(10):
            print('nop')
    sql = f"\n        WITH time_series AS (\n            SELECT\n                EXTRACT(epoch FROM generate_series(\n                    date_trunc('hour', to_timestamp(%(start_timestamp)s/1000)),\n                    date_trunc('hour', to_timestamp(%(end_timestamp)s/1000)) + interval '1 hour',\n                    interval %(step_size)s\n                ))::bigint as unix_time\n        )\n        SELECT\n            time_series.unix_time * 1000 as time,\n            project_id,\n            agent_id,\n            ROUND(AVG(CASE WHEN event_type = 'assist' THEN duration ELSE 0 END)) as assist_avg,\n            ROUND(AVG(CASE WHEN event_type = 'call' THEN duration ELSE 0 END)) as call_avg,\n            ROUND(AVG(CASE WHEN event_type = 'control' THEN duration ELSE 0 END)) as control_avg,\n            ROUND(SUM(CASE WHEN event_type = 'assist' THEN duration ELSE 0 END)) as assist_total,\n            ROUND(SUM(CASE WHEN event_type = 'call' THEN duration ELSE 0 END)) as call_total,\n            ROUND(SUM(CASE WHEN event_type = 'control' THEN duration ELSE 0 END)) as control_total\n        FROM\n            time_series\n            LEFT JOIN assist_events ON time_series.unix_time = EXTRACT(epoch FROM DATE_TRUNC('hour', to_timestamp(assist_events.timestamp/1000)))\n        WHERE\n            {' AND '.join((f'{constraint}' for constraint in constraints))}\n        GROUP BY time, project_id, agent_id\n        ORDER BY time\n        LIMIT %(limit)s OFFSET %(offset)s;\n    "
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(sql, params)
        cur.execute(query)
        rows = cur.fetchall()
    return rows

def get_averages(project_id: int, start_timestamp: int, end_timestamp: int, user_id: int=None):
    if False:
        while True:
            i = 10
    constraints = ['project_id = %(project_id)s', 'timestamp BETWEEN %(start_timestamp)s AND %(end_timestamp)s']
    params = {'project_id': project_id, 'limit': 5, 'offset': 0, 'start_timestamp': start_timestamp, 'end_timestamp': end_timestamp, 'step_size': f'{60} seconds'}
    if user_id is not None:
        constraints.append('agent_id = %(agent_id)s')
        params['agent_id'] = user_id
    totals = __get_all_events_totals(constraints, params)
    rows = __get_all_events_averages(constraints, params)
    params['start_timestamp'] = start_timestamp - (end_timestamp - start_timestamp)
    params['end_timestamp'] = start_timestamp
    previous_totals = __get_all_events_totals(constraints, params)
    return {'currentPeriod': totals[0], 'previousPeriod': previous_totals[0], 'list': helper.list_to_camel_case(rows)}

def __get_all_events_totals(constraints, params):
    if False:
        i = 10
        return i + 15
    sql = f"\n       SELECT\n           ROUND(SUM(assist_total))  as assist_total,\n           ROUND(AVG(assist_avg))    as assist_avg,\n           ROUND(SUM(call_total))    as call_total,\n           ROUND(AVG(call_avg))      as call_avg,\n           ROUND(SUM(control_total)) as control_total,\n           ROUND(AVG(control_avg))   as control_avg\n        FROM assist_events_aggregates\n        WHERE {' AND '.join((f'{constraint}' for constraint in constraints))}\n    "
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(sql, params)
        cur.execute(query)
        rows = cur.fetchall()
    return helper.list_to_camel_case(rows)

def __get_all_events_averages(constraints, params):
    if False:
        print('Hello World!')
    sql = f"\n        SELECT\n            timestamp,\n            assist_avg,\n            call_avg,\n            control_avg,\n            assist_total,\n            call_total,\n            control_total\n        FROM assist_events_aggregates\n        WHERE\n            {' AND '.join((f'{constraint}' for constraint in constraints))}\n        ORDER BY timestamp;\n    "
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(sql, params)
        cur.execute(query)
        rows = cur.fetchall()
    return rows

def __get_all_events_averagesx(constraints, params):
    if False:
        print('Hello World!')
    sql = f"\n        WITH time_series AS (\n            SELECT\n                EXTRACT(epoch FROM generate_series(\n                    date_trunc('minute', to_timestamp(%(start_timestamp)s/1000)),\n                    date_trunc('minute', to_timestamp(%(end_timestamp)s/1000)),\n                    interval %(step_size)s\n                ))::bigint as unix_time\n        )\n        SELECT\n            time_series.unix_time as time,\n            project_id,\n            ROUND(AVG(CASE WHEN event_type = 'assist' THEN duration ELSE 0 END)) as assist_avg,\n            ROUND(AVG(CASE WHEN event_type = 'call' THEN duration ELSE 0 END)) as call_avg,\n            ROUND(AVG(CASE WHEN event_type = 'control' THEN duration ELSE 0 END)) as control_avg,\n            ROUND(SUM(CASE WHEN event_type = 'assist' THEN duration ELSE 0 END)) as assist_total,\n            ROUND(SUM(CASE WHEN event_type = 'call' THEN duration ELSE 0 END)) as call_total,\n            ROUND(SUM(CASE WHEN event_type = 'control' THEN duration ELSE 0 END)) as control_total\n        FROM\n            time_series\n            LEFT JOIN assist_events ON time_series.unix_time = EXTRACT(epoch FROM DATE_TRUNC('minute', to_timestamp(assist_events.timestamp/1000)))\n        WHERE\n            {' AND '.join((f'{constraint}' for constraint in constraints))}\n        GROUP BY time, project_id\n        ORDER BY time;\n\n    "
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(sql, params)
        cur.execute(query)
        rows = cur.fetchall()
    return rows

def get_top_members(project_id: int, start_timestamp: int, end_timestamp: int, sort_by: str, sort_order: str, user_id: int=None, page: int=0, limit: int=5) -> AssistStatsTopMembersResponse:
    if False:
        print('Hello World!')
    event_type = event_type_mapping.get(sort_by)
    if event_type is None:
        raise HTTPException(status_code=400, detail='Invalid sort option provided. Supported options are: ' + ', '.join(event_type_mapping.keys()))
    constraints = ['project_id = %(project_id)s', 'timestamp BETWEEN %(start_timestamp)s AND %(end_timestamp)s']
    params = {'project_id': project_id, 'limit': limit, 'offset': page, 'sort_by': sort_by, 'sort_order': sort_order.upper(), 'start_timestamp': start_timestamp, 'end_timestamp': end_timestamp, 'event_type': event_type}
    if user_id is not None:
        constraints.append('agent_id = %(agent_id)s')
        params['agent_id'] = user_id
    sql = f"\n        SELECT\n            COUNT(1) OVER () AS total,\n            ae.agent_id,\n            u.name AS name,\n            CASE WHEN '{sort_by}' = 'sessionsAssisted'\n                 THEN SUM(CASE WHEN ae.event_type = 'assist' THEN 1 ELSE 0 END)\n                 ELSE SUM(CASE WHEN ae.event_type = %(event_type)s THEN ae.duration ELSE 0 END)\n            END AS count,\n            SUM(CASE WHEN ae.event_type = 'assist' THEN ae.duration ELSE 0 END) AS assist_duration,\n            SUM(CASE WHEN ae.event_type = 'call' THEN ae.duration ELSE 0 END) AS call_duration,\n            SUM(CASE WHEN ae.event_type = 'control' THEN ae.duration ELSE 0 END) AS control_duration,\n            SUM(CASE WHEN ae.event_type = 'assist' THEN 1 ELSE 0 END) AS assist_count\n        FROM assist_events ae\n            JOIN users u ON u.user_id = ae.agent_id\n        WHERE {' AND '.join((f'ae.{constraint}' for constraint in constraints))}\n            AND ae.event_type = '{event_type}'\n        GROUP BY ae.agent_id, u.name\n        ORDER BY count {params['sort_order']}\n        LIMIT %(limit)s OFFSET %(offset)s\n    "
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(sql, params)
        cur.execute(query)
        rows = cur.fetchall()
    if len(rows) == 0:
        return AssistStatsTopMembersResponse(total=0, list=[])
    count = rows[0]['total']
    rows = helper.list_to_camel_case(rows)
    for row in rows:
        row.pop('total')
    return AssistStatsTopMembersResponse(total=count, list=rows)

def get_sessions(project_id: int, data: AssistStatsSessionsRequest) -> AssistStatsSessionsResponse:
    if False:
        return 10
    constraints = ['project_id = %(project_id)s', 'timestamp BETWEEN %(start_timestamp)s AND %(end_timestamp)s']
    params = {'project_id': project_id, 'limit': data.limit, 'offset': (data.page - 1) * data.limit, 'sort_by': data.sort, 'sort_order': data.order.upper(), 'start_timestamp': data.startTimestamp, 'end_timestamp': data.endTimestamp}
    if data.userId is not None:
        constraints.append('agent_id = %(agent_id)s')
        params['agent_id'] = data.userId
    sql = f"\n        SELECT\n            COUNT(1) OVER () AS count,\n            ae.session_id,\n            MIN(ae.timestamp) as timestamp,\n            SUM(CASE WHEN ae.event_type = 'call' THEN ae.duration ELSE 0 END) AS call_duration,\n            SUM(CASE WHEN ae.event_type = 'control' THEN ae.duration ELSE 0 END) AS control_duration,\n            SUM(CASE WHEN ae.event_type = 'assist' THEN ae.duration ELSE 0 END) AS assist_duration,\n            (SELECT json_agg(json_build_object('name', u.name, 'id', u.user_id))\n                    FROM users u\n                    WHERE u.user_id = ANY (array_agg(ae.agent_id)))                     AS team_members\n        FROM assist_events ae\n        WHERE {' AND '.join((f'ae.{constraint}' for constraint in constraints))}\n        GROUP BY ae.session_id\n        ORDER BY {params['sort_by']} {params['sort_order']}\n        LIMIT %(limit)s OFFSET %(offset)s\n    "
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(sql, params)
        cur.execute(query)
        rows = cur.fetchall()
    if len(rows) == 0:
        return AssistStatsSessionsResponse(total=0, page=1, list=[])
    count = rows[0]['count']
    rows = helper.list_to_camel_case(rows)
    for row in rows:
        row.pop('count')
    return AssistStatsSessionsResponse(total=count, page=data.page, list=rows)