from collections import defaultdict
from typing import Dict, Optional
from django.conf import settings
from django.db import connection
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.template import loader
from django.utils.timezone import now as timezone_now
from markupsafe import Markup
from psycopg2.sql import SQL
from analytics.lib.counts import COUNT_STATS
from analytics.views.activity_common import dictfetchall, get_page, realm_activity_link, realm_stats_link, realm_support_link, realm_url_link
from analytics.views.support import get_plan_name
from zerver.decorator import require_server_admin
from zerver.lib.request import has_request_variables
from zerver.models import Realm, get_org_type_display_name
if settings.BILLING_ENABLED:
    from corporate.lib.analytics import estimate_annual_recurring_revenue_by_realm, get_realms_with_default_discount_dict

def get_realm_day_counts() -> Dict[str, Dict[str, Markup]]:
    if False:
        i = 10
        return i + 15
    query = SQL("\n        select\n            r.string_id,\n            (now()::date - (end_time - interval '1 hour')::date) age,\n            coalesce(sum(value), 0) cnt\n        from zerver_realm r\n        join analytics_realmcount rc on r.id = rc.realm_id\n        where\n            property = 'messages_sent:is_bot:hour'\n        and\n            subgroup = 'false'\n        and\n            end_time > now()::date - interval '8 day' - interval '1 hour'\n        group by\n            r.string_id,\n            age\n    ")
    cursor = connection.cursor()
    cursor.execute(query)
    rows = dictfetchall(cursor)
    cursor.close()
    counts: Dict[str, Dict[int, int]] = defaultdict(dict)
    for row in rows:
        counts[row['string_id']][row['age']] = row['cnt']

    def format_count(cnt: int, style: Optional[str]=None) -> Markup:
        if False:
            while True:
                i = 10
        if style is not None:
            good_bad = style
        elif cnt == min_cnt:
            good_bad = 'bad'
        elif cnt == max_cnt:
            good_bad = 'good'
        else:
            good_bad = 'neutral'
        return Markup('<td class="number {good_bad}">{cnt}</td>').format(good_bad=good_bad, cnt=cnt)
    result = {}
    for string_id in counts:
        raw_cnts = [counts[string_id].get(age, 0) for age in range(8)]
        min_cnt = min(raw_cnts[1:])
        max_cnt = max(raw_cnts[1:])
        cnts = format_count(raw_cnts[0], 'neutral') + Markup().join(map(format_count, raw_cnts[1:]))
        result[string_id] = dict(cnts=cnts)
    return result

def realm_summary_table() -> str:
    if False:
        while True:
            i = 10
    now = timezone_now()
    query = SQL("\n        SELECT\n            realm.string_id,\n            realm.date_created,\n            realm.plan_type,\n            realm.org_type,\n            coalesce(wau_table.value, 0) wau_count,\n            coalesce(dau_table.value, 0) dau_count,\n            coalesce(user_count_table.value, 0) user_profile_count,\n            coalesce(bot_count_table.value, 0) bot_count\n        FROM\n            zerver_realm as realm\n            LEFT OUTER JOIN (\n                SELECT\n                    value _14day_active_humans,\n                    realm_id\n                from\n                    analytics_realmcount\n                WHERE\n                    property = 'realm_active_humans::day'\n                    AND end_time = %(realm_active_humans_end_time)s\n            ) as _14day_active_humans_table ON realm.id = _14day_active_humans_table.realm_id\n            LEFT OUTER JOIN (\n                SELECT\n                    value,\n                    realm_id\n                from\n                    analytics_realmcount\n                WHERE\n                    property = '7day_actives::day'\n                    AND end_time = %(seven_day_actives_end_time)s\n            ) as wau_table ON realm.id = wau_table.realm_id\n            LEFT OUTER JOIN (\n                SELECT\n                    value,\n                    realm_id\n                from\n                    analytics_realmcount\n                WHERE\n                    property = '1day_actives::day'\n                    AND end_time = %(one_day_actives_end_time)s\n            ) as dau_table ON realm.id = dau_table.realm_id\n            LEFT OUTER JOIN (\n                SELECT\n                    value,\n                    realm_id\n                from\n                    analytics_realmcount\n                WHERE\n                    property = 'active_users_audit:is_bot:day'\n                    AND subgroup = 'false'\n                    AND end_time = %(active_users_audit_end_time)s\n            ) as user_count_table ON realm.id = user_count_table.realm_id\n            LEFT OUTER JOIN (\n                SELECT\n                    value,\n                    realm_id\n                from\n                    analytics_realmcount\n                WHERE\n                    property = 'active_users_audit:is_bot:day'\n                    AND subgroup = 'true'\n                    AND end_time = %(active_users_audit_end_time)s\n            ) as bot_count_table ON realm.id = bot_count_table.realm_id\n        WHERE\n            _14day_active_humans IS NOT NULL\n            or realm.plan_type = 3\n        ORDER BY\n            dau_count DESC,\n            string_id ASC\n    ")
    cursor = connection.cursor()
    cursor.execute(query, {'realm_active_humans_end_time': COUNT_STATS['realm_active_humans::day'].last_successful_fill(), 'seven_day_actives_end_time': COUNT_STATS['7day_actives::day'].last_successful_fill(), 'one_day_actives_end_time': COUNT_STATS['1day_actives::day'].last_successful_fill(), 'active_users_audit_end_time': COUNT_STATS['active_users_audit:is_bot:day'].last_successful_fill()})
    rows = dictfetchall(cursor)
    cursor.close()
    for row in rows:
        row['date_created_day'] = row['date_created'].strftime('%Y-%m-%d')
        row['age_days'] = int((now - row['date_created']).total_seconds() / 86400)
        row['is_new'] = row['age_days'] < 12 * 7
    counts = get_realm_day_counts()
    for row in rows:
        try:
            row['history'] = counts[row['string_id']]['cnts']
        except Exception:
            row['history'] = ''
    total_arr = 0
    if settings.BILLING_ENABLED:
        estimated_arrs = estimate_annual_recurring_revenue_by_realm()
        realms_with_default_discount = get_realms_with_default_discount_dict()
        for row in rows:
            row['plan_type_string'] = get_plan_name(row['plan_type'])
            string_id = row['string_id']
            if string_id in estimated_arrs:
                row['arr'] = estimated_arrs[string_id]
            if row['plan_type'] in [Realm.PLAN_TYPE_STANDARD, Realm.PLAN_TYPE_PLUS]:
                row['effective_rate'] = 100 - int(realms_with_default_discount.get(string_id, 0))
            elif row['plan_type'] == Realm.PLAN_TYPE_STANDARD_FREE:
                row['effective_rate'] = 0
            elif row['plan_type'] == Realm.PLAN_TYPE_LIMITED and string_id in realms_with_default_discount:
                row['effective_rate'] = 100 - int(realms_with_default_discount[string_id])
            else:
                row['effective_rate'] = ''
        total_arr += sum(estimated_arrs.values())
    for row in rows:
        row['org_type_string'] = get_org_type_display_name(row['org_type'])
    for row in rows:
        row['realm_url'] = realm_url_link(row['string_id'])
        row['stats_link'] = realm_stats_link(row['string_id'])
        row['support_link'] = realm_support_link(row['string_id'])
        row['string_id'] = realm_activity_link(row['string_id'])
    num_active_sites = sum((row['dau_count'] >= 5 for row in rows))
    total_dau_count = 0
    total_user_profile_count = 0
    total_bot_count = 0
    total_wau_count = 0
    for row in rows:
        total_dau_count += int(row['dau_count'])
        total_user_profile_count += int(row['user_profile_count'])
        total_bot_count += int(row['bot_count'])
        total_wau_count += int(row['wau_count'])
    total_row = dict(string_id='Total', plan_type_string='', org_type_string='', effective_rate='', arr=total_arr, realm_url='', stats_link='', support_link='', date_created_day='', dau_count=total_dau_count, user_profile_count=total_user_profile_count, bot_count=total_bot_count, wau_count=total_wau_count)
    rows.insert(0, total_row)
    content = loader.render_to_string('analytics/realm_summary_table.html', dict(rows=rows, num_active_sites=num_active_sites, utctime=now.strftime('%Y-%m-%d %H:%M %Z'), billing_enabled=settings.BILLING_ENABLED))
    return content

@require_server_admin
@has_request_variables
def get_installation_activity(request: HttpRequest) -> HttpResponse:
    if False:
        for i in range(10):
            print('nop')
    content: str = realm_summary_table()
    title = 'Installation activity'
    return render(request, 'analytics/activity_details_template.html', context=dict(data=content, title=title, is_home=True))

@require_server_admin
def get_integrations_activity(request: HttpRequest) -> HttpResponse:
    if False:
        while True:
            i = 10
    title = 'Integrations by client'
    query = SQL("\n        select\n            case\n                when query like '%%external%%' then split_part(query, '/', 5)\n                else client.name\n            end client_name,\n            realm.string_id,\n            sum(count) as hits,\n            max(last_visit) as last_time\n        from zerver_useractivity ua\n        join zerver_client client on client.id = ua.client_id\n        join zerver_userprofile up on up.id = ua.user_profile_id\n        join zerver_realm realm on realm.id = up.realm_id\n        where\n            (query in ('send_message_backend', '/api/v1/send_message')\n            and client.name not in ('Android', 'ZulipiOS')\n            and client.name not like 'test: Zulip%%'\n            )\n        or\n            query like '%%external%%'\n        group by client_name, string_id\n        having max(last_visit) > now() - interval '2 week'\n        order by client_name, string_id\n    ")
    cols = ['Client', 'Realm', 'Hits', 'Last time']
    integrations_activity = get_page(query, cols, title)
    return render(request, 'analytics/activity_details_template.html', context=dict(data=integrations_activity['content'], title=integrations_activity['title'], is_home=False))