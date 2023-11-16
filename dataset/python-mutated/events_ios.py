from chalicelib.utils import pg_client, helper
from chalicelib.core import events

def get_customs_by_session_id(session_id, project_id):
    if False:
        return 10
    return events.get_customs_by_session_id(session_id=session_id, project_id=project_id)

def get_by_sessionId(session_id, project_id):
    if False:
        for i in range(10):
            print('nop')
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify(f"\n            SELECT \n                c.*,\n                'TAP' AS type\n            FROM events_ios.taps AS c\n            WHERE \n              c.session_id = %(session_id)s\n            ORDER BY c.timestamp;", {'project_id': project_id, 'session_id': session_id}))
        rows = cur.fetchall()
        cur.execute(cur.mogrify(f"\n            SELECT \n                i.*,\n                'INPUT' AS type\n            FROM events_ios.inputs AS i\n            WHERE \n              i.session_id = %(session_id)s\n            ORDER BY i.timestamp;", {'project_id': project_id, 'session_id': session_id}))
        rows += cur.fetchall()
        cur.execute(cur.mogrify(f"\n            SELECT \n                v.*,\n                'VIEW' AS type\n            FROM events_ios.views AS v\n            WHERE \n              v.session_id = %(session_id)s\n            ORDER BY v.timestamp;", {'project_id': project_id, 'session_id': session_id}))
        rows += cur.fetchall()
        cur.execute(cur.mogrify(f"\n            SELECT \n                s.*,\n                'SWIPE' AS type\n            FROM events_ios.swipes AS s\n            WHERE \n              s.session_id = %(session_id)s\n            ORDER BY s.timestamp;", {'project_id': project_id, 'session_id': session_id}))
        rows += cur.fetchall()
        rows = helper.list_to_camel_case(rows)
        rows = sorted(rows, key=lambda k: k['timestamp'])
    return rows

def get_crashes_by_session_id(session_id):
    if False:
        while True:
            i = 10
    with pg_client.PostgresClient() as cur:
        cur.execute(cur.mogrify(f'\n                    SELECT cr.*,uc.*, cr.timestamp - s.start_ts AS time\n                    FROM {events.EventType.CRASH_IOS.table} AS cr \n                        INNER JOIN public.crashes_ios AS uc USING (crash_ios_id) \n                        INNER JOIN public.sessions AS s USING (session_id)\n                    WHERE\n                      cr.session_id = %(session_id)s\n                    ORDER BY timestamp;', {'session_id': session_id}))
        errors = cur.fetchall()
        return helper.list_to_camel_case(errors)