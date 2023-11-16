from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from psycopg2.sql import SQL
from analytics.views.activity_common import get_page
from zerver.decorator import require_server_admin

@require_server_admin
def get_remote_server_activity(request: HttpRequest) -> HttpResponse:
    if False:
        return 10
    title = 'Remote servers'
    query = SQL("\n        with icount as (\n            select\n                server_id,\n                max(value) as max_value,\n                max(end_time) as max_end_time\n            from zilencer_remoteinstallationcount\n            where\n                property='active_users:is_bot:day'\n                and subgroup='false'\n            group by server_id\n            ),\n        remote_push_devices as (\n            select server_id, count(distinct(user_id)) as push_user_count from zilencer_remotepushdevicetoken\n            group by server_id\n        )\n        select\n            rserver.id,\n            rserver.hostname,\n            rserver.contact_email,\n            max_value,\n            push_user_count,\n            max_end_time\n        from zilencer_remotezulipserver rserver\n        left join icount on icount.server_id = rserver.id\n        left join remote_push_devices on remote_push_devices.server_id = rserver.id\n        order by max_value DESC NULLS LAST, push_user_count DESC NULLS LAST\n    ")
    cols = ['ID', 'Hostname', 'Contact email', 'Analytics users', 'Mobile users', 'Last update time']
    remote_servers = get_page(query, cols, title, totals_columns=[3, 4])
    return render(request, 'analytics/activity_details_template.html', context=dict(data=remote_servers['content'], title=remote_servers['title'], is_home=False))