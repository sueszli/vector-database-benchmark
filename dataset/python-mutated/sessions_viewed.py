from chalicelib.utils import pg_client

def view_session(project_id, user_id, session_id):
    if False:
        print('Hello World!')
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify('INSERT INTO public.user_viewed_sessions(user_id, session_id) \n                            VALUES (%(userId)s,%(session_id)s)\n                            ON CONFLICT DO NOTHING;', {'userId': user_id, 'session_id': session_id}))