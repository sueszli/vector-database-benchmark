from chalicelib.utils import pg_client
from chalicelib.utils import helper
from decouple import config
from chalicelib.utils.TimeUTC import TimeUTC

def get_all(user_id):
    if False:
        i = 10
        return i + 15
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify("\n        SELECT a.*, u.last >= (EXTRACT(EPOCH FROM a.created_at)*1000) AS viewed\n        FROM public.announcements AS a,\n             (SELECT COALESCE(CAST(data ->> 'lastAnnouncementView' AS bigint), 0)\n              FROM public.users\n              WHERE user_id = %(userId)s\n              LIMIT 1) AS u(last)\n        ORDER BY a.created_at DESC;", {'userId': user_id})
        cur.execute(query)
        announcements = helper.list_to_camel_case(cur.fetchall())
        for a in announcements:
            a['createdAt'] = TimeUTC.datetime_to_timestamp(a['createdAt'])
            if a['imageUrl'] is not None and len(a['imageUrl']) > 0:
                a['imageUrl'] = config('announcement_url') + a['imageUrl']
        return announcements

def view(user_id):
    if False:
        for i in range(10):
            print('nop')
    with pg_client.PostgresClient() as cur:
        query = cur.mogrify('\n        UPDATE public.users\n        SET data=data ||\n                 (\'{"lastAnnouncementView":\' ||\n                  (EXTRACT(EPOCH FROM timezone(\'utc\'::text, now())) * 1000)::bigint - 20 * 000 ||\n                  \'}\')::jsonb\n        WHERE user_id = %(userId)s;', {'userId': user_id})
        cur.execute(query)
    return True