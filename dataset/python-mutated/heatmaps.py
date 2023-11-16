from chalicelib.utils import sql_helper as sh
import schemas
from chalicelib.utils import helper, pg_client

def get_by_url(project_id, data: schemas.GetHeatmapPayloadSchema):
    if False:
        while True:
            i = 10
    args = {'startDate': data.startTimestamp, 'endDate': data.endTimestamp, 'project_id': project_id, 'url': data.url}
    constraints = ['sessions.project_id = %(project_id)s', '(url = %(url)s OR path= %(url)s)', 'clicks.timestamp >= %(startDate)s', 'clicks.timestamp <= %(endDate)s', 'start_ts >= %(startDate)s', 'start_ts <= %(endDate)s', 'duration IS NOT NULL']
    query_from = 'events.clicks INNER JOIN sessions USING (session_id)'
    q_count = 'count(1) AS count'
    has_click_rage_filter = False
    if len(data.filters) > 0:
        for (i, f) in enumerate(data.filters):
            if f.type == schemas.FilterType.issue and len(f.value) > 0:
                has_click_rage_filter = True
                q_count = 'max(real_count) AS count,TRUE AS click_rage'
                query_from += 'INNER JOIN events_common.issues USING (timestamp, session_id)\n                               INNER JOIN issues AS mis USING (issue_id)\n                               INNER JOIN LATERAL (\n                                    SELECT COUNT(1) AS real_count\n                                     FROM events.clicks AS sc\n                                              INNER JOIN sessions as ss USING (session_id)\n                                     WHERE ss.project_id = 2\n                                       AND (sc.url = %(url)s OR sc.path = %(url)s)\n                                       AND sc.timestamp >= %(startDate)s\n                                       AND sc.timestamp <= %(endDate)s\n                                       AND ss.start_ts >= %(startDate)s\n                                       AND ss.start_ts <= %(endDate)s\n                                       AND sc.selector = clicks.selector) AS r_clicks ON (TRUE)'
                constraints += ['mis.project_id = %(project_id)s', 'issues.timestamp >= %(startDate)s', 'issues.timestamp <= %(endDate)s']
                f_k = f'issue_value{i}'
                args = {**args, **sh.multi_values(f.value, value_key=f_k)}
                constraints.append(sh.multi_conditions(f'%({f_k})s = ANY (issue_types)', f.value, value_key=f_k))
                constraints.append(sh.multi_conditions(f'mis.type = %({f_k})s', f.value, value_key=f_k))
    if data.click_rage and (not has_click_rage_filter):
        constraints.append('(issues.session_id IS NULL \n                                OR (issues.timestamp >= %(startDate)s\n                                    AND issues.timestamp <= %(endDate)s\n                                    AND mis.project_id = %(project_id)s))')
        q_count += ",COALESCE(bool_or(mis.type = 'click_rage'), FALSE) AS click_rage"
        query_from += 'LEFT JOIN events_common.issues USING (timestamp, session_id)\n                       LEFT JOIN issues AS mis USING (issue_id)'
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify(f"SELECT selector, {q_count}\n                                FROM {query_from}\n                                WHERE {' AND '.join(constraints)}\n                                GROUP BY selector\n                                LIMIT 500;", args)
        try:
            cur.execute(query)
        except Exception as err:
            print('--------- HEATMAP SEARCH QUERY EXCEPTION -----------')
            print(query.decode('UTF-8'))
            print('--------- PAYLOAD -----------')
            print(data)
            print('--------------------')
            raise err
        rows = cur.fetchall()
    return helper.list_to_camel_case(rows)