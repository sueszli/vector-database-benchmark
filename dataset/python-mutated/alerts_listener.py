from chalicelib.utils import pg_client, helper

def get_all_alerts():
    if False:
        i = 10
        return i + 15
    with pg_client.PostgresClient(long_query=True) as cur:
        query = "SELECT -1 AS tenant_id,\n                           alert_id,\n                           projects.project_id,\n                           detection_method,\n                           query,\n                           options,\n                           (EXTRACT(EPOCH FROM alerts.created_at) * 1000)::BIGINT AS created_at,\n                           alerts.name,\n                           alerts.series_id,\n                           filter,\n                           change,\n                           COALESCE(metrics.name || '.' || (COALESCE(metric_series.name, 'series ' || index)) || '.count',\n                                    query ->> 'left')                             AS series_name\n                    FROM public.alerts\n                             INNER JOIN projects USING (project_id)\n                             LEFT JOIN metric_series USING (series_id)\n                             LEFT JOIN metrics USING (metric_id)\n                    WHERE alerts.deleted_at ISNULL\n                      AND alerts.active\n                      AND projects.active\n                      AND projects.deleted_at ISNULL\n                      AND (alerts.series_id ISNULL OR metric_series.deleted_at ISNULL)\n                    ORDER BY alerts.created_at;"
        cur.execute(query=query)
        all_alerts = helper.list_to_camel_case(cur.fetchall())
    return all_alerts