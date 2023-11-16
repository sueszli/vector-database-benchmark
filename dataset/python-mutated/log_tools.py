from chalicelib.utils import pg_client, helper
import json
EXCEPT = ['jira_server', 'jira_cloud']

def search(project_id):
    if False:
        return 10
    result = []
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('                SELECT supported_integrations.name,\n                       (SELECT COUNT(*)\n                        FROM public.integrations\n                                 INNER JOIN public.projects USING (project_id)\n                        WHERE provider = supported_integrations.name\n                          AND project_id = %(project_id)s\n                          AND projects.deleted_at ISNULL\n                        LIMIT 1) AS count\n                FROM unnest(enum_range(NULL::integration_provider)) AS supported_integrations(name);', {'project_id': project_id}))
        r = cur.fetchall()
        for k in r:
            if k['count'] > 0 and k['name'] not in EXCEPT:
                result.append({'value': helper.key_to_camel_case(k['name']), 'type': 'logTool'})
        return {'data': result}

def add(project_id, integration, options):
    if False:
        for i in range(10):
            print('nop')
    options = json.dumps(options)
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('                INSERT INTO public.integrations(project_id, provider, options) \n                VALUES (%(project_id)s, %(provider)s, %(options)s::jsonb)\n                RETURNING *;', {'project_id': project_id, 'provider': integration, 'options': options}))
        r = cur.fetchone()
    return helper.dict_to_camel_case(helper.flatten_nested_dicts(r))

def get(project_id, integration):
    if False:
        print('Hello World!')
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('                SELECT integrations.* \n                FROM public.integrations INNER JOIN public.projects USING(project_id)\n                WHERE provider = %(provider)s \n                    AND project_id = %(project_id)s\n                    AND projects.deleted_at ISNULL\n                LIMIT 1;', {'project_id': project_id, 'provider': integration}))
        r = cur.fetchone()
    return helper.dict_to_camel_case(helper.flatten_nested_dicts(r))

def get_all_by_type(integration):
    if False:
        while True:
            i = 10
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('                SELECT integrations.* \n                FROM public.integrations INNER JOIN public.projects USING(project_id)\n                WHERE provider = %(provider)s AND projects.deleted_at ISNULL;', {'provider': integration}))
        r = cur.fetchall()
    return helper.list_to_camel_case(r, flatten=True)

def edit(project_id, integration, changes):
    if False:
        print('Hello World!')
    if 'projectId' in changes:
        changes.pop('project_id')
    if 'integration' in changes:
        changes.pop('integration')
    if len(changes.keys()) == 0:
        return None
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('                    UPDATE public.integrations\n                    SET options=options||%(changes)s\n                    WHERE project_id =%(project_id)s AND provider = %(provider)s \n                    RETURNING *;', {'project_id': project_id, 'provider': integration, 'changes': json.dumps(changes)}))
        return helper.dict_to_camel_case(helper.flatten_nested_dicts(cur.fetchone()))

def delete(project_id, integration):
    if False:
        while True:
            i = 10
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('                    DELETE FROM public.integrations\n                    WHERE project_id=%(project_id)s AND provider=%(provider)s;', {'project_id': project_id, 'provider': integration}))
        return {'state': 'success'}

def get_all_by_tenant(tenant_id, integration):
    if False:
        while True:
            i = 10
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('SELECT integrations.* \n                    FROM public.integrations INNER JOIN public.projects USING(project_id) \n                    WHERE provider = %(provider)s \n                        AND projects.deleted_at ISNULL;', {'provider': integration}))
        r = cur.fetchall()
    return helper.list_to_camel_case(r, flatten=True)