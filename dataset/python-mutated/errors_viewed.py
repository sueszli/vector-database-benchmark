from chalicelib.utils import pg_client

def add_viewed_error(project_id, user_id, error_id):
    if False:
        i = 10
        return i + 15
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('INSERT INTO public.user_viewed_errors(user_id, error_id) \n                            VALUES (%(userId)s,%(error_id)s);', {'userId': user_id, 'error_id': error_id}))

def viewed_error_exists(user_id, error_id):
    if False:
        while True:
            i = 10
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify('SELECT \n                    errors.error_id AS hydrated,\n                    COALESCE((SELECT TRUE\n                                         FROM public.user_viewed_errors AS ve\n                                         WHERE ve.error_id = %(error_id)s\n                                           AND ve.user_id = %(userId)s LIMIT 1), FALSE) AS viewed                                                \n                FROM public.errors\n                WHERE error_id = %(error_id)s', {'userId': user_id, 'error_id': error_id})
        cur.execute(query=query)
        r = cur.fetchone()
        if r:
            return r.get('viewed')
    return True

def viewed_error(project_id, user_id, error_id):
    if False:
        print('Hello World!')
    if viewed_error_exists(user_id=user_id, error_id=error_id):
        return None
    return add_viewed_error(project_id=project_id, user_id=user_id, error_id=error_id)