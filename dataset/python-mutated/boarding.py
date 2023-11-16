from chalicelib.utils import pg_client
from chalicelib.core import projects, log_tool_datadog, log_tool_stackdriver, log_tool_sentry
from chalicelib.core import users

def get_state(tenant_id):
    if False:
        for i in range(10):
            print('nop')
    pids = projects.get_projects_ids(tenant_id=tenant_id)
    with pg_client.PostgresClient() as cur:
        recorded = False
        meta = False
        if len(pids) > 0:
            cur.execute(cur.mogrify('SELECT EXISTS((  SELECT 1\n                                                FROM public.sessions AS s\n                                                WHERE s.project_id IN %(ids)s)) AS exists;', {'ids': tuple(pids)}))
            recorded = cur.fetchone()['exists']
            meta = False
            if recorded:
                cur.execute('SELECT EXISTS((SELECT 1\n                               FROM public.projects AS p\n                                        LEFT JOIN LATERAL ( SELECT 1\n                                                            FROM public.sessions\n                                                            WHERE sessions.project_id = p.project_id\n                                                              AND sessions.user_id IS NOT NULL\n                                                            LIMIT 1) AS sessions(user_id) ON (TRUE)\n                               WHERE p.deleted_at ISNULL\n                                 AND ( sessions.user_id IS NOT NULL OR p.metadata_1 IS NOT NULL\n                                       OR p.metadata_2 IS NOT NULL OR p.metadata_3 IS NOT NULL\n                                       OR p.metadata_4 IS NOT NULL OR p.metadata_5 IS NOT NULL\n                                       OR p.metadata_6 IS NOT NULL OR p.metadata_7 IS NOT NULL\n                                       OR p.metadata_8 IS NOT NULL OR p.metadata_9 IS NOT NULL\n                                       OR p.metadata_10 IS NOT NULL )\n                                   )) AS exists;')
                meta = cur.fetchone()['exists']
    return [{'task': 'Install OpenReplay', 'done': recorded, 'URL': 'https://docs.openreplay.com/getting-started/quick-start'}, {'task': 'Identify Users', 'done': meta, 'URL': 'https://docs.openreplay.com/data-privacy-security/metadata'}, {'task': 'Invite Team Members', 'done': len(users.get_members(tenant_id=tenant_id)) > 1, 'URL': 'https://app.openreplay.com/client/manage-users'}, {'task': 'Integrations', 'done': len(log_tool_datadog.get_all(tenant_id=tenant_id)) > 0 or len(log_tool_sentry.get_all(tenant_id=tenant_id)) > 0 or len(log_tool_stackdriver.get_all(tenant_id=tenant_id)) > 0, 'URL': 'https://docs.openreplay.com/integrations'}]

def get_state_installing(tenant_id):
    if False:
        print('Hello World!')
    pids = projects.get_projects_ids(tenant_id=tenant_id)
    with pg_client.PostgresClient() as cur:
        recorded = False
        if len(pids) > 0:
            cur.execute(cur.mogrify('SELECT EXISTS((  SELECT 1\n                                                FROM public.sessions AS s\n                                                WHERE s.project_id IN %(ids)s)) AS exists;', {'ids': tuple(pids)}))
            recorded = cur.fetchone()['exists']
    return {'task': 'Install OpenReplay', 'done': recorded, 'URL': 'https://docs.openreplay.com/getting-started/quick-start'}

def get_state_identify_users(tenant_id):
    if False:
        print('Hello World!')
    with pg_client.PostgresClient() as cur:
        cur.execute('SELECT EXISTS((SELECT 1\n                                       FROM public.projects AS p\n                                                LEFT JOIN LATERAL ( SELECT 1\n                                                                    FROM public.sessions\n                                                                    WHERE sessions.project_id = p.project_id\n                                                                      AND sessions.user_id IS NOT NULL\n                                                                    LIMIT 1) AS sessions(user_id) ON (TRUE)\n                                       WHERE p.deleted_at ISNULL\n                                         AND ( sessions.user_id IS NOT NULL OR p.metadata_1 IS NOT NULL\n                                               OR p.metadata_2 IS NOT NULL OR p.metadata_3 IS NOT NULL\n                                               OR p.metadata_4 IS NOT NULL OR p.metadata_5 IS NOT NULL\n                                               OR p.metadata_6 IS NOT NULL OR p.metadata_7 IS NOT NULL\n                                               OR p.metadata_8 IS NOT NULL OR p.metadata_9 IS NOT NULL\n                                               OR p.metadata_10 IS NOT NULL )\n                                           )) AS exists;')
        meta = cur.fetchone()['exists']
    return {'task': 'Identify Users', 'done': meta, 'URL': 'https://docs.openreplay.com/data-privacy-security/metadata'}

def get_state_manage_users(tenant_id):
    if False:
        return 10
    return {'task': 'Invite Team Members', 'done': len(users.get_members(tenant_id=tenant_id)) > 1, 'URL': 'https://app.openreplay.com/client/manage-users'}

def get_state_integrations(tenant_id):
    if False:
        for i in range(10):
            print('nop')
    return {'task': 'Integrations', 'done': len(log_tool_datadog.get_all(tenant_id=tenant_id)) > 0 or len(log_tool_sentry.get_all(tenant_id=tenant_id)) > 0 or len(log_tool_stackdriver.get_all(tenant_id=tenant_id)) > 0, 'URL': 'https://docs.openreplay.com/integrations'}